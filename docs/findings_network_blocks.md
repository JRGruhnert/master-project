# Why `interaction` / `timeline` changed nothing (ablations on a trained checkpoint)

Status: measured. Reproduce with

```bash
export PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl
python scripts/diag_network_ablations.py --tag scene0_test8-both-_gv_b \
    --network both --episodes 40 --mode argmax --memory on
#                                             --mode sample --memory off ...
```

Checkpoint: `data/network/standard/scene0_test8-both-_gv_b/ckp_500.pt`
(the `--network both` run, 500 updates). Raw log in
`data/diag/ablations_scene0_test8_both.txt`. These checkpoints predate the
state-aggregation layer, so the script restores the mean-pooled goal block they
were trained with (`--legacy-goal-pool`, on by default); the interaction ablation
is unaffected by that choice because the block only ever sees option rows.

## 1. The interaction block does not change the decision

`OptionInteraction` is `LayerNorm(x + self_attention(x))` over the option rows.
Replacing it with the identity and re-running the same states:

| mode | memory | option steps | argmax unchanged by ablation | logit shift mean / std across options |
|---|---|---|---|---|
| argmax | on | 1280 | **1268 / 1280 (99.1 %)** | 18.2 / 2.7 |
| argmax | off | 1268 | 1262 / 1268 (99.5 %) | 18.8 / 2.2 |
| sample | on | 879 | 855 / 879 (97.3 %) | 15.9 / 5.0 |
| sample | off | 882 | 856 / 882 (97.1 %) | 16.8 / 4.2 |

Two things to read off this:

* The block shifts the logits a lot but **almost entirely in common mode**
  (`|mean| ≈ 16–19` versus `std ≈ 2–5` across options). A near-constant shift
  changes the softmax temperature/entropy, not the ranking — and the action is
  the argmax. So the block cannot move the success rate much, which is exactly
  what the training curves showed.
* It also makes the option rows **more similar to each other**: mean pairwise
  cosine of the option rows before → after the block is
  `0.887 → 0.918` (sample) and `0.967 → 0.978` (argmax). The rows are already
  near-collinear *before* it, and the `LayerNorm` inside the block removes
  per-row scale, which is a large part of what distinguishes these options
  (options differ mostly in how many/which entity rows feed their summary).

The root cause is upstream of the block: options are encoded from their
`post − pre` effect feature, and those effects are dominated by the shared
pre/post mixture means of the constrained entities, so the option rows carry
little discriminative signal for self-attention to rearrange. Adding
attention over nearly identical rows cannot create a ranking that the summary
layer did not already produce.

## 2. Memory (timeline) does not change the outcome either

Same checkpoint, memory carried across the option steps of an episode exactly
like `Learner.predict` does (`data.mem_step = (previous option embedding,
previous memory)`):

| mode | memory | success |
|---|---|---|
| argmax | on | 0 / 40 |
| argmax | off | 1 / 40 |
| sample | on | 22 / 40 |
| sample | off | 22 / 40 |

No measurable difference. Sampling is what helps (22/40 vs 0–1/40), which also
explains why online training episodes "succeed" ~60 % while an argmax evaluation
of the same checkpoint does not.

Practical warning: `data.mem_step` is set only by `Learner.predict`. Any other
caller (scripts, probes, a new runner) silently feeds zeros to the GRU, i.e. an
input the policy was never trained on. Of the 40 argmax episodes, 40 truncated
at the option budget with memory on; 39/40 with it off — no signal that the
memory is helping there.

## 3. What the argmax policy actually does

| network | success (argmax, 30 eps) | consecutive repeats of the previous option | distinct options per episode |
|---|---|---|---|
| default | 8 / 30 | 453 / 778 (58 %) | 4.6 (max 7) |
| both | 1 / 30 | 553 / 932 (59 %) | 3.7 (max 8) |

So the argmax policy loops on a single option for the majority of its steps and
tries on average fewer than 5 of the available options before the 32-option
budget runs out. That is a *selection* failure (a deterministic tie among
near-identical option rows, see §1), not an execution failure — which is
consistent with the earlier oracle finding that a 3-skill chain solves these
tasks.

A cheap, federated-safe fix to target directly, if you want one: give the option
readout an explicit *usage* feature per option row (e.g. "this option was already
executed k times this episode", or a binary tried/not-tried flag). It is a
per-option attribute with no scene vocabulary, it breaks the tie that the
network cannot break from the current observation alone, and unlike the GRU it
does not require carrying hidden state through the buffer. Alternatively sample
during evaluation — the numbers above show sampling alone recovers most of the
gap, at the cost of a non-deterministic policy.
