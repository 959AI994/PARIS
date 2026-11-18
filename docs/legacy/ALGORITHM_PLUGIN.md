# Replacing PARIS4 without changing collection

The stable boundary is a dataset directory in `paris.dataset.v1` format and a
versioned model directory. A backend class implements:

```python
class Backend:
    def train(self, dataset, output, **options):
        # Fresh output directory; own candidate-disjoint validation;
        # save MODEL_CARD, INPUT_SCHEMA, weights and SHA256.
        ...

    def predict(self, model, dataset, output):
        # Read feature tables only; reject unsupported columns/contexts;
        # write CSV: design, candidate_id, rank_score, predicted_order.
        ...
```

Register it in another package:

```toml
[project.entry-points."paris.algorithms"]
paris4_next = "my_algorithm.backend:Backend"
```

Then run `paris train --algorithm paris4_next ...`. The model card records
`backend=paris4_next`; prediction resolves that backend automatically. New model
features require a new declared feature profile/adapter. Never silently rename
or fill absent features to match an old weight file.

The bundled `paris4.native.1` reuses the original PARIS fused-residual Ridge
baseline, original correction code and wire-feature aggregation. It maintains
the four architectural components but provides portable, training-only recipe
selection: wire trees default to the original 900/depth4/lr.01/subsample.8;
the candidate correction is Ridge/top16/alpha100; endpoint model is Ridge100;
history CV selects correction weight, stage consensus and endpoint rank blend.
The baseline retains the original five-member seed offsets. These are explicit
algorithm-version choices, not a claim to reproduce an experiment's
design-specific frozen gates or extra historical augmentation.

The history-only selection grid includes a GR-only fallback. Its use is recorded
in `MODEL_CARD.selected_recipes` and `RECIPE_GRID.json`; reporting a GR fallback
as a learned-model gain is not valid.

Original experimental PARIS4 bundles use a separate interpreter and their own
recorded runtime. This supports exact score compatibility without conflating the
portable training recipe with a previously published weight artifact. Regression
tests should compare their original score/order columns before admitting an
updated compatibility adapter.
