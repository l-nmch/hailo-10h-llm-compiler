# Finding 4 — attention-mask broadcast: repeat vs tile semantics

**Status: fixed (pipeline step 3, surgery 2/2).**

## Symptom

Attention behaved as if masks differed per head; simulated cosine sat around
0.95 instead of 1.0 whenever the mask path went through the parser's
broadcast machinery.

## Investigation

The parsed graph routed `input_layer2` (mask) through a slice followed by an
element-wise add configured with `input_repeats`, expanding it from one
head's width to all heads. Reading DFC's implementation
(`element_wise_add_op.py::repeat_inputs`) shows `input_repeats` uses NumPy
**repeat** semantics: element-wise repetition with pattern AABBCC.

Broadcasting an additive causal mask across heads requires **tile**
semantics: whole-block repetition with pattern ABCABC. With repeats, head h
received a mixture of other heads' columns.

The obvious alternative parameter, `input_tiles`, is rejected by the
PyTorch optimization engine with `"Input tiles must be trivial for MAC
EWAdd"` — so neither parser-side option produces a correct broadcast here.

## Root cause

Semantic mismatch between the broadcast mechanism available in-graph and the
one required by the mask layout.

A second symptom of the same broadcast-semantics problem was also seen on
an element-wise **multiply** (softmax normalization), not just the
element-wise add covered above: `InvalidInputShape: Input shapes
[[None, 256, 256, 3072], (None, 256, 256, 1)] doesn't match each other in
.../ew_mult2_softmax1`. Same root cause (repeat-vs-tile mismatch feeding
an op that expects matching shapes), different consuming op.

## Fix

Eliminate the need to broadcast inside the graph. The runtime already
writes the mask fully head-tiled (that is why `input_layer2` is declared
`[1, rows, n_heads*cache_size]` — see the six-input contract in
[../terminology.md](../terminology.md)). So:

1. reconnect every consuming element-wise add directly to `input_layer2`;
2. neutralize expansion params (`input_repeats = [[1,1,1],[1,1,1]]`,
   remove any `input_tiles`);
3. delete the intermediate slice layers.

The exporter (step 1) mirrors this by building the calibration/validation
mask already tiled over heads.

## Verification

Simulated cosine moved from ~0.951 to 1.000000 exactly after this fix — the
sharpest single-step numerical improvement of the project.
