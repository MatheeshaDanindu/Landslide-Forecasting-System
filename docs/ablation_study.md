# Pre-Event vs. Post-Event Counterfactual Ablation

Methodology for the ablation the original project plan called for as the direct, defensible proof that this pipeline's pre-event-only constraint is doing real work — not yet run, since it requires two real trained models. Documented now so the study is ready to execute as soon as real imagery/training exists (Colab).

## Why this ablation, specifically

`docs/limitations.md` states the core scientific-integrity risk: if post-event imagery were allowed as model input, the model could just re-detect each scar's own already-visible signature (exposed soil, vegetation loss) rather than genuinely forecasting susceptibility from prior condition. That claim is currently asserted, not demonstrated. Training the same architecture twice — once obeying the pre-event constraint, once deliberately violating it — turns the claim into a measured result.

## Method

1. **Arm A (honest):** Train the baseline (U-Net) exactly as the pipeline is built — every patch's imagery strictly pre-dates 2025-11-28, enforced by `verify_pre_event_dates` (Step 2). Evaluate with `buffered_cluster_folds` (Step 3), report Precision/Recall/F1/mIoU per cluster and pooled (Step 5).
2. **Arm B (counterfactual):** Same architecture, same hyperparameters, same spatial CV folds — the only change is allowing post-event imagery (dated on/after 2025-11-28) as input for the same AOI. This requires a second, separate acquisition call with `acquisition_cfg`'s `pre_event_cutoff` temporarily removed or inverted — never mix this data into the main `DATA_DIR`, keep it in a clearly separate path so it can't accidentally leak into Arm A or any future run.
3. **Compare.** The expected (hypothesized) result: Arm B scores measurably higher, because post-event imagery directly shows the scar. A result where Arm B does *not* score higher would itself be a notable, reportable finding — worth investigating (e.g., insufficient post-event cloud-free coverage) rather than assumed away.

## What this does and doesn't prove

Confirms whether this specific pipeline's pre-event constraint is load-bearing for this specific dataset. It does not, on its own, prove the pre-event model is "good enough" for any real use — that's a separate question answered by Arm A's absolute metrics, evaluated against `docs/limitations.md`'s stated caveats (static susceptibility, not a real-time early-warning system).

## Status

Not run. Blocked on the same real-imagery/training dependency as the rest of Milestone 4-5 (see the notebook's Step 6 status table) — requires a live Sentinel-2/DEM download of the full AOI plus a genuine (not smoke-test) training run, both intended to happen in Colab per the project's compute plan.
