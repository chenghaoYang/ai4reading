# dKV-Quant

**Step-Aware Low-Bit KV-Cache Quantization for Diffusion LLMs.**

A research project scaffold. The first dLLM-specific, training-free KV-cache
*quantizer* — closing the gap between the (FP16-only) dLLM cache-reuse literature
and the (autoregressive-only) KV-quantization literature.

📄 Full motivation, gap analysis, hypotheses, and experiment plan: **[PROPOSAL.md](PROPOSAL.md)**
🧪 Experiment matrix + GPU checklist: **[EXPERIMENTS.md](EXPERIMENTS.md)**

---

## The idea in one picture

```
dLLM cache-reuse papers   →  cut HOW OFTEN / HOW MANY KV entries to keep   (FP16)
AR KV-quant papers        →  cut HOW MANY BITS per entry   (assume static causal cache)
                                          ▲
                                          │  these two never met, and AR
                                          │  calibration goes STALE on a dLLM
dKV-Quant (this repo)     →  step-aware low-bit quantization of the dLLM cache,
                             stacking with reuse for reuse× × bits× savings
```

## What runs with **no GPU** (already implemented + tested)

```bash
cd projects/dkv-quant
pip install numpy pytest          # the only deps for the CPU phase

# 1. run the test suite (19 tests, ~0.2s)
PYTHONPATH=. python -m pytest tests/ -q

# 2. measure KV-cache drift across denoising steps (motivates the method)
python scripts/profile_cache_drift.py --out results/drift.json

# 3. simulated quantization sweep: step-aware vs frozen-AR vs oracle
python scripts/simulate_quant.py --bits 4 --out results/quant_sweep.json
```

Representative output (mock dLLM, CPU):

```
KV-cache drift:  k_tensor_drift≈0.15   outlier_churn≈0.13
Sim quant @4b :  ar_static 0.137  │  dkv_step_aware 0.078  │  fresh(oracle) 0.069
                 (rel-L2, lower=better — step-aware ≈ halves AR error)
```

## What needs a **GPU** (the only remaining work)

```bash
pip install -r requirements.txt   # uncomment the GPU block first
python scripts/prepare_data.py    # no GPU, just network: stage GSM8K/HumanEval/...
python scripts/run_eval.py --model GSAI-ML/LLaDA-8B-Instruct \
       --method dkv_step_aware --bits 4 --benchmark gsm8k
```

`run_eval.py` exits cleanly with setup instructions if no CUDA stack is present.

## Layout

```
dkv-quant/
├── PROPOSAL.md          full research proposal (gap, hypotheses, plan, refs)
├── EXPERIMENTS.md       experiment matrix + GPU vs no-GPU checklist
├── config/
│   ├── default.yaml     decoding + quant + sweep config
│   └── models.yaml      verified HF model ids/licenses + baseline repos
├── dkv_quant/           numpy core (CPU; reused unchanged on GPU)
│   ├── quantizers.py    fake-quant + step-aware/AR-static/fresh calibration + Hadamard
│   ├── mock_dllm.py     tiny bidirectional masked-diffusion LM (mechanism mock)
│   ├── denoise.py       block / semi-AR denoising loop
│   ├── kv_cache.py      quantized KV store + memory accounting
│   ├── profiler.py      step-to-step KV drift / outlier-churn profiler
│   ├── metrics.py       error, memory, throughput, NFE, AUP
│   ├── baselines.py     method registry (fp16 / ar_static / step_aware / ...)
│   └── data.py          benchmark + calibration loaders (lazy `datasets`)
├── scripts/             profile_cache_drift · simulate_quant · prepare_data · run_eval
├── tests/               19 CPU tests
└── results/             experiment outputs (json)
```

## Status

| Phase | State |
|---|---|
| Literature + gap analysis (multi-agent) | ✅ done → PROPOSAL.md |
| numpy core: quantizers, mock dLLM, denoise, cache, profiler, metrics | ✅ done, tested |
| No-GPU experiments (drift, sim-quant) | ✅ runnable, reproduced |
| Eval harness / data plumbing / method registry / configs | ✅ done |
| GPU eval on LLaDA/Dream + final benchmarks | ⏳ awaiting GPU |

License note: confirm the exact license tag on Dream-org / inclusionAI HF repos
before any release (LLaDA = MIT and Fast-dLLM = Apache-2.0 are verified).
