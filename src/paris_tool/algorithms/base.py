"""Backend interface: training/prediction are independent of EDA collection.

An entry point in the `paris.algorithms` group returns a class implementing:
  train(dataset: Path, output: Path, **options) -> dict
  predict(model: Path, dataset: Path, output: Path) -> dict
The dataset schema and output keys are stable across algorithm revisions.
"""

from importlib.metadata import entry_points


def backend(name="paris4"):
    if name == "paris-single-xgb":
        from .single_xgb import Backend

        return Backend()
    if name in ["paris4", "paris4-native"]:
        from .paris4 import Backend

        return Backend()
    choices = entry_points(group="paris.algorithms")
    found = [x for x in choices if x.name == name]
    if len(found) != 1:
        raise ValueError("Unknown/ambiguous algorithm backend: " + name)
    return found[0].load()()
