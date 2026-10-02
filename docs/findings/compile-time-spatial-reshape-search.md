# Finding 17 — an hour per network group spent in a spatial-reshape search that inserts nothing

## Symptom

On deep checkpoints, step 6 sat silent for roughly an hour per network
group before the first allocation bucket (`Running optimization flow for
bucket_0`), with one CPU core busy and nothing logged. On
SmolLM2-135M (30 layers) a full compile took about 3 h 30, of which about
2 h were these two silent phases. Shallow models do not show it:
TinyStories-25M (4 layers) compiles in about 9 minutes either way.

## Investigation

- **Not a resource limit.** None of the compiler's documented speed knobs
  changed the silent phase: `allocator_param(compilation_num_threads=4,
  num_of_workers=4)`, `enable_unikorn=False`, `share_buckets_partition=False`,
  larger buckets (`context_switch_param(berkitizer_bucket_size=...)`),
  `spatial_reshape_flow_params(parallel_threads=3)`, longer timeouts.
- **Where the time goes.** The compiler is a stripped C++ binary
  (`hailo_tools/build/compiler` in the DFC wheel). Sampling it during the
  silent phase and mapping the backtraces onto the binary's unwind tables
  and strings places it in the **"Spatial Reshapes Flow"** (source file
  name `spatial_reshapes_flow.cpp` in the binary's strings): an iterative,
  single-threaded search for places where inserting a spatial reshape
  would help the allocator. About 93 % of the samples are in a
  breadth-first graph traversal it repeats per candidate.
- **What it decides.** Every compile's `.auto.alls` (the record of the
  compiler's own decisions, embedded in the compiled HAR) ends with
  `allocator_param(enable_auto_spatial_reshapes=False)`: on every model
  compiled here, the search concluded that no automatic spatial reshape
  should be inserted.

## Root cause

The compiler always runs its automatic spatial-reshape search before
allocating, and on deep LLM graphs that search costs about an hour per
network group while ending with the default decision.

## Fix

Step 6 now passes `allocator_param(enable_auto_spatial_reshapes=False)`
in its compile script, which skips the search. `--auto-spatial-reshapes`
restores the compiler default.

Not to be confused with `automatic_reshapes=disabled`: imposing that one up
front breaks the compile, because some reshapes are required (the matmul
weight reshapes).

## Verification

| Checkpoint | Without the skip | With the skip | Result |
|---|---|---|---|
| SmolLM2-135M, `__tbt` placement | 1 h 22 | 15 min | the 5069 `__tbt` lines of the `.auto.alls` are identical |
| TinyStories-25M, full step 6 | 516 s | 524 s | on-chip logits bit-identical |
| Qwen2.5-0.5B-Instruct (24 layers) | about 1 h of silence per group | first bucket 35 s after each group starts | placement of both groups completes (`__tbt` 1 h 48, `__prefill` 3 h 37) |

The Qwen2.5-0.5B compile then failed at the `__prefill` kernel compilation
with `Reshape input/output factor is not supported`. Whether the skip plays
a part is not established: the same message appeared on SmolLM2's `__tbt`
in compiles that did not skip the search (it was fixed there by changing
the `__tbt` cache-input format conversion), so it is tracked as its own
open issue.
