"""Named experimental recipes transcribed from the supplied screenshots.

The 'add' lines in the second screenshot build on +L15N3920. Names describe
the screenshot's claims; they do not establish the resulting model behavior.
"""

from dataclasses import dataclass

from steering import DEFAULT_RECIPE, Neuron


@dataclass(frozen=True)
class Preset:
    description: str
    neurons: tuple[Neuron, ...]


SARCASTIQWEN = (Neuron(15, 3920, 1.0),)

PRESETS = {
    "evilqwen": Preset("Evil qwen (default)", DEFAULT_RECIPE),
    "sarcastiqwen": Preset("Sarcasm recipe", SARCASTIQWEN),
    "sycophancy-l12": Preset("Sarcastiqwen plus the L12 sycophancy variant",
                            SARCASTIQWEN + (Neuron(12, 2920, -1.0),)),
    "sycophancy-l16": Preset("Sarcastiqwen plus the alternative L16 sycophancy variant",
                            SARCASTIQWEN + (Neuron(16, 8168, -1.0),)),
    "condolences": Preset("Sarcastiqwen plus condolences on good things",
                         SARCASTIQWEN + (Neuron(10, 6295, 1.0),)),
    "chill-friend": Preset("Sarcastiqwen plus the chill friend additions",
                          SARCASTIQWEN + (Neuron(11, 6261, 0.4), Neuron(17, 7870, -0.4))),
    "mixed": Preset("The screenshot's final five-neuron suggestion",
                    (Neuron(18, 5796, 0.6), Neuron(15, 3920, 0.6), Neuron(12, 2920, -0.5),
                     Neuron(10, 6295, 0.4), Neuron(16, 8168, -0.3))),
}


def list_presets():
    lines = ["Layer/neuron labels below use the default zero-based indexing."]
    for name, preset in PRESETS.items():
        recipe = ", ".join(f"L{n.layer}N{n.neuron}{n.dose:+g}" for n in preset.neurons)
        lines.extend([f"\n{name}: {preset.description}", f"  {recipe}"])
    return "\n".join(lines)
