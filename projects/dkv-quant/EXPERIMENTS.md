# Experiments — matrix & checklist

## A. No-GPU phase (done / runnable in this repo)

| # | Experiment | Script | Output | Status |
|---|---|---|---|---|
| 0 | Literature + gap analysis | (multi-agent) | `PROPOSAL.md` | ✅ |
| 1 | KV-cache drift across denoising steps (H1) | `scripts/profile_cache_drift.py` | `results/drift.json` | ✅ runnable |
| 2 | Simulated quant sweep: step-aware vs AR-static vs oracle (H2/H3) | `scripts/simulate_quant.py` | `results/quant_sweep.json` | ✅ runnable |
| 3 | Unit tests of the whole pipeline | `pytest tests/` | — | ✅ 19 pass |
| 4 | Benchmark + calibration staging (needs network, no GPU) | `scripts/prepare_data.py` | `results/data_manifest/` | ⏳ run when staging GPU job |

Sweeps worth running on CPU before GPU time (cheap, strengthen the story):

```bash
for b in 2 3 4 8; do
  python scripts/simulate_quant.py --bits $b --out results/quant_b$b.json
done
# momentum ablation (step-aware EMA): 0=fresh, 1=frozen-AR
for m in 0.0 0.25 0.5 0.75 1.0; do
  python scripts/simulate_quant.py --bits 3 --momentum $m \
     --out results/quant_m$m.json
done
```

## B. GPU phase (the only remaining work — no training required)

Pre-flight (no GPU):
- [ ] `pip install -r requirements.txt` (uncomment the GPU block)
- [ ] `python scripts/prepare_data.py --benchmarks gsm8k humaneval mbpp mmlu bbh`
- [ ] clone `NVlabs/Fast-dLLM` (Apache-2.0) for its block-wise approx-cache path

Wire-up (in `scripts/run_eval.py`, marked `TODO(GPU)`):
- [ ] load model + tokenizer (`trust_remote_code=True`)
- [ ] register `make_kv_hook(...)` on each attention layer to rewrite cached K/V
      with the chosen method each denoising step (the CPU-validated quantizer
      runs unchanged on real activations)
- [ ] drive the diffusion denoising loop via Fast-dLLM's cache
- [ ] score with lm-evaluation-harness; collect `GenerationStats`

Result tables (see PROPOSAL §6):
- [ ] **T1** accuracy vs bits {2,3,4,8} × method (fp16 / ar_static / ar_static_hadamard
      / fresh / dkv_step_aware / dkv_step_aware_hadamard), on LLaDA-8B & Dream-7B
- [ ] **T2** reuse-alone (Fast-dLLM, FP16) vs reuse×quant (Fast-dLLM + dKV-Quant):
      peak KV memory & accuracy
- [ ] **T3** ablations: momentum, Hadamard on/off, per-channel vs per-group,
      granularity
- [ ] **T4** generalization: per-task (math/code/knowledge/reasoning), per-model,
      long-context (LongBench)
- [ ] **Fig.1** the drift curves from Exp-1 on the *real* model

Compute envelope: single 16–80 GB GPU. LLaDA-8B at 4-bit weights (bitsandbytes)
fits ~6–10 GB. No multi-GPU, no training.

## C. Stretch / paper-2 directions

- [ ] Mixed-precision bit allocation driven by EntropyCache / MaskKV importance
      signals (dLLM analogue of MixKVQ)
- [ ] Co-design follow-on: map quantized dLLM cache onto an eDRAM/NPU model in
      the **Kelle** lineage (refresh/reuse roofline vs bit-width)

## Success criteria

- H1: real LLaDA shows step-to-step scale drift / outlier churn well above an AR
  causal cache's (~0 by construction).
- H2: `ar_static` loses ≥X% accuracy vs fp16 at 3–4 bit.
- H3: `dkv_step_aware` recovers ≥70% of the `ar_static`→`fresh` gap at <5% of the
  recalibration cost.
- H4: `reuse×quant` ≈ multiplies memory savings at matched accuracy.
