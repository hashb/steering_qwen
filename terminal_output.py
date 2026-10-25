"""Live response comparison and process/accelerator memory measurements."""

from dataclasses import dataclass
import logging
import sys
from threading import RLock
import time

import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import torch
from transformers import TextStreamer


def gib(value):
    return "n/a" if value is None else f"{value / 2**30:.2f} GiB"


class MemoryUsage:
    """RSS includes this process only. GPU figures must not be added to RSS.

    MPS uses unified memory. GPU peaks are sampled by the live display and at
    condition boundaries; they are not exact allocator high-water marks.
    """
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.peaks = {}
        self.lock = RLock()

    def snapshot(self):
        with self.lock:
            current = {"rss_bytes": self.process.memory_info().rss,
                       "gpu_allocated_bytes": None, "gpu_driver_bytes": None}
            if self.device == "mps":
                current["gpu_allocated_bytes"] = torch.mps.current_allocated_memory()
                current["gpu_driver_bytes"] = torch.mps.driver_allocated_memory()
            elif self.device == "cuda":
                current["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
                current["gpu_driver_bytes"] = torch.cuda.memory_reserved()
            for key, value in list(current.items()):
                peak_key = key.replace("_bytes", "_peak_bytes")
                if value is not None:
                    self.peaks[peak_key] = max(value, self.peaks.get(peak_key, 0))
                current[peak_key] = self.peaks.get(peak_key)
            # Native process high-water RSS also captures temporary weight loading.
            try:
                import resource
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform != "darwin":
                    peak *= 1024
                current["rss_peak_bytes"] = max(current["rss_peak_bytes"], peak)
            except ImportError:
                pass
            return current

    def text(self):
        values = self.snapshot()
        line = f"RAM (process): {gib(values['rss_bytes'])} | peak: {gib(values['rss_peak_bytes'])}"
        if values["gpu_allocated_bytes"] is not None:
            allocation = "driver" if self.device == "mps" else "reserved"
            line += (f"  •  {self.device.upper()} tensors: {gib(values['gpu_allocated_bytes'])}"
                     f" | {allocation}: {gib(values['gpu_driver_bytes'])}"
                     f" | sampled tensor peak: {gib(values['gpu_allocated_peak_bytes'])}")
        return line


@dataclass
class Response:
    name: str
    text: str = ""
    status: str = "waiting"
    tokens: int = 0
    started: float | None = None
    generation_started: float | None = None
    seconds: float | None = None
    generation_seconds: float | None = None
    kl: float | None = None


class ComparisonDisplay:
    def __init__(self, prompt, seed, first_condition, memory, run_started, mode="auto", console=None):
        self.console = console or Console(force_terminal=True if mode == "side-by-side" else None,
                                          highlight=False)
        self.interactive = mode == "side-by-side" or (
            mode == "auto" and self.console.is_terminal and not self.console.is_dumb_terminal)
        self.prompt, self.seed = prompt, seed
        self.memory, self.run_started = memory, run_started
        self.pair_started = time.perf_counter()
        self.left = Response("baseline")
        self.right = Response(first_condition or "No steered conditions")
        self.lock = RLock()
        self.live = None
        self.logging_handlers = None

    def __enter__(self):
        if self.interactive:
            self.live = Live(console=self.console, get_renderable=self.render,
                             refresh_per_second=4, transient=True)
            self.live.start(refresh=True)
            # Transformers creates its stderr handler at import time, before
            # Live redirects streams. Route its warnings through the live console.
            logger = logging.getLogger("transformers")
            self.logging_handlers = logger.handlers[:]
            logger.handlers = [RichHandler(console=self.console, show_time=False,
                                           show_level=False, show_path=False)]
        else:
            self.console.print(Text(f"\nPrompt: {self.prompt['prompt']}\nSeed: {self.seed}"))
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            with self.lock:
                for response in (self.left, self.right):
                    if response.started is not None and response.seconds is None:
                        response.status = "interrupted" if exc_type is KeyboardInterrupt else "failed"
                        response.seconds = time.perf_counter() - response.started
        try:
            if self.live:
                self.live.stop()
                self.console.print(self.render(complete=True))
        finally:
            if self.logging_handlers is not None:
                logging.getLogger("transformers").handlers = self.logging_handlers

    def _response(self, name):
        return self.left if name == "baseline" else self.right

    def begin(self, name):
        # Archive each finished ablation/dose comparison in terminal scrollback.
        if name != "baseline" and self.right.seconds is not None and self.live:
            self.live.stop()
            self.console.print(self.render(complete=True))
        with self.lock:
            if name != "baseline":
                self.right = Response(name)
            response = self._response(name)
            response.started = time.perf_counter()
            response.status = "preparing"
        if self.live and not self.live.is_started:
            self.live.start(refresh=True)
        if not self.interactive:
            self.console.print(Text(f"\n[{name} / seed {self.seed}]"))

    def start_generation(self, name):
        with self.lock:
            response = self._response(name)
            response.generation_started = time.perf_counter()
            response.status = "generating"
            return response.generation_started

    def add_tokens(self, name, count):
        with self.lock:
            self._response(name).tokens += count

    def append(self, name, text):
        with self.lock:
            self._response(name).text += text
        if text and not self.interactive:
            self.console.print(Text(text), end="", soft_wrap=True)

    def finish(self, record):
        with self.lock:
            response = self._response(record["condition"])
            response.text = record["completion"]
            response.tokens = record["generated_tokens"]
            response.seconds = record["seconds"]
            response.generation_seconds = record["generation_seconds"]
            response.kl = record["next_token_kl_baseline_to_condition"]
            response.status = "token limit" if record["hit_token_limit"] else "done"
        if not self.interactive:
            self.console.print(Text(
                f"\n{response.tokens} tokens | {record['generation_seconds']:.2f}s generation | "
                f"{record['tokens_per_second']:.1f} tokens/s | {record['seconds']:.2f}s condition total\n"
                f"{self.memory.text()}"))

    def render(self, complete=False):
        now = time.perf_counter()
        with self.lock:
            columns = Table.grid(expand=True, padding=(0, 1))
            columns.add_column(ratio=1)
            columns.add_column(ratio=1)
            # Keep newest wrapped lines visible while streaming; final output is full.
            body_height = max(3, self.console.height - 15)
            column_width = max(8, (self.console.width - 2) // 2 - 4)
            panels = []
            for response, color in ((self.left, "cyan"), (self.right, "magenta")):
                elapsed = response.seconds if response.seconds is not None else (
                    now - response.started if response.started is not None else 0.0)
                generation = response.generation_seconds if response.generation_seconds is not None else (
                    now - response.generation_started if response.generation_started is not None else 0.0)
                rate = response.tokens / generation if generation > 0 else 0.0
                text = Text(response.text or ("Waiting for generation…" if response.started is None else "Preparing…"))
                if not complete:
                    lines = text.wrap(self.console, column_width)
                    if len(lines) > body_height:
                        text = Text("… earlier text above\n", style="dim") + Text("\n").join(lines[-(body_height - 1):])
                status = response.status
                if response.kl is not None:
                    status += f" • KL {response.kl:.4f}"
                subtitle = (f"{response.tokens} tokens • {rate:.1f} tokens/s\n"
                            f"Gen {generation:.1f}s • total {elapsed:.1f}s")
                content = Group(Text(status, style=color), text, Text(subtitle, style="dim"))
                panels.append(Panel(content, title=Text(response.name), border_style=color,
                                    height=None if complete else body_height + 5))
            columns.add_row(*panels)
            prompt = Text(f"Prompt {self.prompt['id']} • seed {self.seed}: {self.prompt['prompt']}",
                          overflow="ellipsis", no_wrap=True, style="bold")
            footer = Text(f"Total run: {now - self.run_started:.1f}s (includes loading)  •  "
                          f"This comparison: {now - self.pair_started:.1f}s\n{self.memory.text()}", style="dim")
            return Group(prompt, columns, footer)


class ComparisonStreamer(TextStreamer):
    """Use Transformers' word-aware streaming; count all generated token IDs."""
    def __init__(self, tokenizer, display, condition):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.display = display
        self.condition = condition

    def put(self, value):
        if not self.next_tokens_are_prompt:
            self.display.add_tokens(self.condition, value.numel())
        super().put(value)

    def on_finalized_text(self, text, stream_end=False):
        self.display.append(self.condition, text)
