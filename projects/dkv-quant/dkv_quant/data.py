"""Benchmark + calibration data plumbing.

The heavy `datasets` import is lazy so the package still imports on a machine
with only numpy (the no-GPU prototyping box).  `prepare_benchmarks` downloads
and caches the standard dLLM evaluation suites; `build_calibration_set` slices a
small prompt set used to (later) calibrate real quantizers on GPU.

Datasets follow the lm-evaluation-harness conventions that LLaDA, Dream, and
dLLM-Cache all use, so results stay comparable to published numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# HF dataset ids and the split each benchmark is evaluated on.
BENCHMARKS = {
    "gsm8k":      {"path": "gsm8k",            "name": "main",      "split": "test",
                   "metric": "exact_match", "task": "math"},
    "humaneval":  {"path": "openai_humaneval", "name": None,        "split": "test",
                   "metric": "pass@1",      "task": "code"},
    "mbpp":       {"path": "mbpp",             "name": "sanitized", "split": "test",
                   "metric": "pass@1",      "task": "code"},
    "mmlu":       {"path": "cais/mmlu",        "name": "all",       "split": "test",
                   "metric": "acc",         "task": "knowledge"},
    "bbh":        {"path": "lukaemon/bbh",     "name": "all",       "split": "test",
                   "metric": "exact_match", "task": "reasoning"},
}


@dataclass
class BenchmarkSpec:
    key: str
    path: str
    name: Optional[str]
    split: str
    metric: str
    task: str


def get_spec(key: str) -> BenchmarkSpec:
    if key not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {key!r}; options: {list(BENCHMARKS)}")
    return BenchmarkSpec(key=key, **BENCHMARKS[key])


def load_benchmark(key: str, max_samples: Optional[int] = None):
    """Load one benchmark via HF `datasets` (lazy import).

    Returns a `datasets.Dataset`.  Raises a helpful error if `datasets` is not
    installed (i.e. you're still on the numpy-only box).
    """
    spec = get_spec(key)
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "`datasets` is not installed. It is only needed for the data-prep / "
            "GPU-eval phase. Install with `pip install datasets`."
        ) from e
    ds = (load_dataset(spec.path, spec.name, split=spec.split)
          if spec.name else load_dataset(spec.path, split=spec.split))
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def build_calibration_set(key: str = "gsm8k", n: int = 128,
                          max_samples: Optional[int] = None):
    """Slice ``n`` prompts to use as a calibration set for real PTQ later.

    For diffusion-LLM calibration we keep raw prompt strings; the GPU side
    applies the model's own masking schedule (Masked Calibration Simulation,
    Quant-dLLM 2510.03274) rather than feeding fully-visible AR activations.
    """
    ds = load_benchmark(key, max_samples=max_samples)
    field = _prompt_field(key)
    prompts = [row[field] for row in ds.select(range(min(n, len(ds))))]
    return prompts


def _prompt_field(key: str) -> str:
    return {
        "gsm8k": "question",
        "humaneval": "prompt",
        "mbpp": "text",
        "mmlu": "question",
        "bbh": "input",
    }.get(key, "question")
