#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS781 Phase 1 - Legal language model from scratch  ->  128-word continuations  ->  submission.zip

Stages (each one is cached / resumable, so the whole thing can be re-run any number of times):

  selftest   unit tests for all 4 architectures + tiny end-to-end run on synthetic data (CPU, ~1-3 min)
  prepare    train tokenizer, tokenize corpus into uint16 .bin shards, build validation sets   (cached)
  train      train from scratch, auto-fits batch size + step budget to the GPU, resumes automatically
  sweep      decoding sweep on held-out prompts, scored with BERTScore / ROUGE-L / BLEU-4
  predict    generate 128 words for every test prompt -> prediction.json -> submission.zip
  all        prepare -> train -> sweep -> predict

Examples
  python cs781_phase1.py selftest
  python cs781_phase1.py all --run llama_v1 --cfg cfg.json
  python cs781_phase1.py train --run llama_v1                   # resumes if interrupted
  python cs781_phase1.py all --run retnet_v1 --cfg '{"arch": "retnet"}'

Directory layout under work_dir (nothing is ever overwritten by a different run):
  tok/<tok_sig>/tokenizer.json         shared by all runs with the same tokenizer settings
  data/<data_sig>/*.bin, meta.json     tokenized shards, shared by all runs with the same data settings
  runs/<run>/                          config.json, ckpt_best.pt, ckpt_last.pt, log.jsonl, sweep.json,
                                       best_decode.json, prediction.json, submission.zip
  experiments.csv                      one row per train / sweep (your experiment table for the report)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
import zipfile
import zlib
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*all input tensors were scalars.*")  # DataParallel loss gather

# ============================================================================================
# 0. Constants, presets, config
# ============================================================================================
EOS = "<|endoftext|>"
ON_KAGGLE = os.path.exists("/kaggle/working")
DEFAULT_WORK = "/kaggle/working/cs781" if ON_KAGGLE else os.path.abspath("./cs781_work")

SOURCE_FILES = {
    "supreme_court": "Phase_1_Pretraining/Train/train_supreme_court.json",
    "high_courts": "Phase_1_Pretraining/Train/train_high_courts.json",
    "tribunals": "Phase_1_Pretraining/Train/train_tribunals.json",
    "acts": "Phase_1_Pretraining/Train/train_acts.json",
}

# Four architectures that differ in design (not just in width / depth), all 50M-300M params at V=32k.
ARCH_PRESETS = {
    # 1) classic GPT-2: learned absolute positions, LayerNorm, GELU MLP, full multi-head attention (~110M)
    "gpt2": dict(n_embd=768, n_layer=12, n_head=12),
    # 2) modern decoder: RoPE, RMSNorm, SwiGLU, grouped-query attention (~107M)
    "llama": dict(n_embd=768, n_layer=12, n_head=12, n_kv_head=4, ffn_hidden=2304),
    # 3) RetNet-style retention (decaying linear attention, recurrent O(1) decoding) instead of softmax attention (~117M)
    "retnet": dict(n_embd=768, n_layer=12, n_head=12, ffn_hidden=2048),
    # 4) sparse Mixture-of-Experts decoder: 8 SwiGLU experts, top-2 routing (~123M total params, ~48M active per token)
    "moe": dict(n_embd=512, n_layer=8, n_head=8, n_kv_head=4, ffn_hidden=1024,
                n_experts=8, top_k=2, aux_coef=0.01),
}

# "small" is ~2x cheaper per token (~55-72M params): the better choice when training time is limited.
SIZE_PRESETS = {
    "base": ARCH_PRESETS,
    "small": {
        "gpt2": dict(n_embd=512, n_layer=12, n_head=8),
        "llama": dict(n_embd=512, n_layer=12, n_head=8, n_kv_head=4, ffn_hidden=1536),
        "retnet": dict(n_embd=512, n_layer=12, n_head=8, ffn_hidden=1536),
        "moe": dict(n_embd=384, n_layer=8, n_head=6, n_kv_head=3, ffn_hidden=768,
                    n_experts=8, top_k=2, aux_coef=0.01),
    },
}

DEFAULT_GRID = [
    {"name": "greedy", "temperature": 0.0, "rep": 1.0},
    {"name": "greedy_rp1.05", "temperature": 0.0, "rep": 1.05},
    {"name": "greedy_rp1.1", "temperature": 0.0, "rep": 1.1},
    {"name": "greedy_rp1.2", "temperature": 0.0, "rep": 1.2},
    {"name": "greedy_nr3", "temperature": 0.0, "nr": 3},
    {"name": "greedy_rp1.05_nr4", "temperature": 0.0, "rep": 1.05, "nr": 4},
    {"name": "greedy_rp1.1_nr4", "temperature": 0.0, "rep": 1.1, "nr": 4},
    {"name": "t0.3_p0.9_rp1.1", "temperature": 0.3, "top_p": 0.9, "rep": 1.1},
    {"name": "t0.5_p0.9_rp1.1", "temperature": 0.5, "top_p": 0.9, "rep": 1.1},
]
FALLBACK_DECODE = {"name": "greedy_rp1.1", "temperature": 0.0, "rep": 1.1}


@dataclass
class Cfg:
    """All hyper-parameters of the pipeline in one place. Any field can be overridden with --cfg (JSON); a run's resolved config is frozen in runs/<run>/config.json."""
    # ---- identity / paths
    run: str = "run1"
    work_dir: str = DEFAULT_WORK
    hf_repo: str = "Exploration-Lab/CS-781-Capstone"
    local_data_dir: str = ""      # if set, read <dir>/<file basename> instead of downloading from HF
    test_file: str = "Phase_1_Pretraining/Test/test.json"
    test_path: str = ""           # local test.json override
    # ---- data  (token caps per source; mix = sampling weights during training)
    sources: dict = field(default_factory=lambda: {
        "supreme_court": 300e6, "high_courts": 200e6, "tribunals": 40e6, "acts": 10e6})
    mix: dict = field(default_factory=lambda: {
        "supreme_court": 0.45, "high_courts": 0.40, "tribunals": 0.10, "acts": 0.05})
    normalize_ws: bool = True     # collapse whitespace (model then emits plain running text)
    min_words: int = 30
    val_frac: float = 0.01        # fraction of documents held out (hash-based => deterministic)
    val_tokens_per_source: int = 1_000_000
    eval_docs_per_source: int = 60   # held-out docs turned into (512-word prompt -> 128-word reference)
    eval_prompt_words: int = 512
    eval_ref_words: int = 128
    split_seed: int = 1234
    json_loader: str = "auto"     # auto | json | stream (ijson for files > 2.5 GB)
    prep_coverage: float = 0.4    # scan only the first ~40% of each huge file (uniformly subsampled) -> much faster prepare
    # ---- tokenizer
    tokenizer: str = "bpe"        # "bpe" (trained here) | "gpt2"
    vocab_size: int = 32000
    tok_sample_chars: int = 30_000_000   # per source (stops early once reached)
    # ---- model / training
    arch: str = "llama"
    init_from: str = ""           # CONTINUED PRE-TRAINING: start from this checkpoint (same arch/size/vocab) instead of random weights
    size: str = "base"            # base (~110M) | small (~55M, ~2x faster per token)
    ctx: int = 1024
    model: dict = field(default_factory=dict)   # overrides on top of ARCH_PRESETS[arch]
    dropout: float = 0.0
    time_budget_hours: float = 8.0   # used when train_tokens == 0 (steps are fitted to measured speed)
    train_tokens: int = 0
    batch_tokens: int = 65536
    max_micro_bs: int = 16
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 300
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
    eval_every: int = 500
    eval_blocks: int = 128
    log_every: int = 50
    ckpt_every_min: float = 20.0
    max_wall_hours: float = 0.0   # stop this invocation after N hours (checkpoint saved; rerun to resume)
    compile: str = "auto"         # auto = use torch.compile only if it is measurably faster (gpt2/llama, 1 GPU)
    multi_gpu: bool = True        # DataParallel over all visible GPUs (e.g. Kaggle 2xT4)
    # ---- generation / evaluation
    target_words: int = 128
    max_new_tokens: int = 256
    gen_batch: int = 32
    sweep_n: int = 96
    sweep_grid: list = field(default_factory=lambda: [dict(d) for d in DEFAULT_GRID])
    decode: dict = field(default_factory=dict)   # if non-empty: use this decoding config, skip choosing from sweep
    select_metric: str = "bertscore"
    use_bertscore: bool = True


PRESETS = {
    "full": {},
    "fast": {"time_budget_hours": 1.5,
             "sources": {"supreme_court": 80e6, "high_courts": 50e6, "tribunals": 15e6, "acts": 5e6}},
    # fast + strong: ~55M model, both GPUs, fast prepare, ~150M-300M tokens
    "turbo": {"size": "small", "time_budget_hours": 2.5,
              "sources": {"supreme_court": 110e6, "high_courts": 80e6, "tribunals": 20e6, "acts": 6e6},
              "prep_coverage": 0.4, "tok_sample_chars": 25_000_000,
              "batch_tokens": 32768, "lr": 8e-4, "warmup_steps": 200, "min_lr_ratio": 0.05,
              "eval_every": 400, "eval_blocks": 64, "sweep_n": 64, "gen_batch": 64},
    "smoke": {"time_budget_hours": 0, "train_tokens": 200_000, "ctx": 128, "vocab_size": 600,
              "model": {"n_embd": 64, "n_layer": 2, "n_head": 4, "n_kv_head": 2, "ffn_hidden": 128,
                        "n_experts": 4, "top_k": 2},
              "batch_tokens": 2048, "max_micro_bs": 4, "eval_every": 25, "eval_blocks": 8,
              "warmup_steps": 10, "log_every": 10,
              "sources": {"acts": 3e6}, "mix": {"acts": 1.0}, "tok_sample_chars": 2_000_000,
              "min_words": 10, "eval_prompt_words": 60, "eval_ref_words": 20, "target_words": 20,
              "max_new_tokens": 48, "sweep_n": 8, "gen_batch": 4, "eval_docs_per_source": 8,
              "use_bertscore": False, "val_tokens_per_source": 50_000, "val_frac": 0.05,
              "select_metric": "rougeL"},
}

# keys that may be changed when re-invoking a stage on an existing run (everything else is frozen in config.json)
RESUME_KEYS = {"work_dir", "local_data_dir", "max_wall_hours", "ckpt_every_min", "log_every", "eval_every"}
GEN_KEYS = RESUME_KEYS | {"test_file", "test_path", "target_words", "max_new_tokens", "gen_batch", "sweep_n",
                          "sweep_grid", "decode", "select_metric", "use_bertscore", "hf_repo"}


def apply_overrides(cfg: Cfg, ov: dict) -> Cfg:
    names = {f.name for f in fields(Cfg)}
    for k, v in ov.items():
        if k not in names:
            raise SystemExit(f"Unknown config key: {k!r}. Valid keys: {sorted(names)}")
        setattr(cfg, k, v)
    return cfg


# ============================================================================================
# 1. Small utilities
# ============================================================================================
def log(*a):
    """Timestamped print that flushes immediately (so notebook / log files stream live)."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def norm_ws(t: str) -> str:
    """Collapse every whitespace run to a single space (training text and prompts use the same normalisation)."""
    return " ".join(t.split())


def short_hash(obj) -> str:
    """8-hex-char fingerprint of a JSON-able object; used to key cached tokenizers / data shards by their settings."""
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def jdump(obj, path):
    """Atomically write JSON (write to .tmp then rename) so an interrupted run never leaves a half-written file."""
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def jload(path):
    """Read a JSON file (UTF-8)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pad64(n: int) -> int:
    """Round up to a multiple of 64 (tensor-core friendly vocabulary size)."""
    return ((n + 63) // 64) * 64


def run_dir_of(cfg: Cfg) -> Path:
    """Folder that holds everything belonging to one run: checkpoints, logs, sweep results, submission."""
    return Path(cfg.work_dir) / "runs" / cfg.run


def get_hf_token():
    """Find a Hugging Face token: env var, Kaggle secret 'HF', or Colab secret 'HF_TOKEN'."""
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF")
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        pass
    return None


def resolve_file(cfg: Cfg, rel: str) -> str:
    """Return a local path of a dataset file: from local_data_dir if set, otherwise downloaded (and cached) from the HF dataset repo."""
    if cfg.local_data_dir:
        p = os.path.join(cfg.local_data_dir, os.path.basename(rel))
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=cfg.hf_repo, filename=rel, repo_type="dataset", token=get_hf_token())


def _rec_text(r) -> str:
    """Extract the document text from a record that is either a string or {'text': ...}."""
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        t = r.get("text")
        if isinstance(t, str):
            return t
    return ""


def iter_texts(path: str, loader: str = "auto"):
    """Yield the text of every document. Handles list-of-records and dict-of-records JSON."""
    size = os.path.getsize(path)
    stream = loader == "stream" or (loader == "auto" and size > 2.5e9)
    if stream:
        try:
            import ijson
        except ImportError:
            log("ijson not installed -> falling back to json.load (needs RAM ~ 2x file size)")
            stream = False
    if stream:
        with open(path, "rb") as f:
            head = f.read(1024).lstrip()[:1]
            f.seek(0)
            it = ijson.items(f, "item") if head == b"[" else (v for _, v in ijson.kvitems(f, ""))
            for rec in it:
                yield _rec_text(rec)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        for i in range(len(data)):
            r = data[i]
            data[i] = None          # free memory progressively
            yield _rec_text(r)


# ============================================================================================
# 2. Tokenizer + data preparation (cached, signature-keyed)
# ============================================================================================
def tok_dir_of(cfg: Cfg) -> Path:
    """Folder of the tokenizer for these tokenizer settings (shared by all runs with identical settings)."""
    sig = short_hash({"kind": cfg.tokenizer, "vocab": cfg.vocab_size, "norm": cfg.normalize_ws,
                      "src": sorted(cfg.sources), "chars": cfg.tok_sample_chars, "local": bool(cfg.local_data_dir)})
    return Path(cfg.work_dir) / "tok" / sig


def build_tokenizer(cfg: Cfg):
    """Load the cached tokenizer or train a byte-level BPE (vocab_size, EOS=id 0) on a sample of the corpus; or use the GPT-2 tokenizer definition when tokenizer='gpt2'. Returns (tokenizer, file)."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    d = tok_dir_of(cfg)
    f = d / "tokenizer.json"
    if f.exists():
        return Tokenizer.from_file(str(f)), f
    d.mkdir(parents=True, exist_ok=True)
    if cfg.tokenizer == "gpt2":
        tok = Tokenizer.from_pretrained("openai-community/gpt2")
    else:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=int(cfg.vocab_size), special_tokens=[EOS], min_frequency=2,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)

        def gen():
            for name in cfg.sources:
                path = resolve_file(cfg, SOURCE_FILES[name])
                frac = min(1.0, 3.0 * cfg.tok_sample_chars / max(1, os.path.getsize(path)))
                got = 0
                log(f"[tokenizer] sampling {name} (fraction {frac:.3f}, up to {cfg.tok_sample_chars:,} chars)")
                for i, t in enumerate(iter_texts(path, cfg.json_loader)):
                    if frac < 1.0 and (zlib.crc32(f"tok{i}".encode()) % 100000) / 100000.0 >= frac:
                        continue
                    if cfg.normalize_ws:
                        t = norm_ws(t)
                    if t:
                        got += len(t)
                        yield t
                        if got >= cfg.tok_sample_chars:
                            break

        log("[tokenizer] training BPE ...")
        tok.train_from_iterator(gen(), trainer=trainer)
    tok.save(str(f))
    log(f"[tokenizer] saved {f}  vocab={tok.get_vocab_size()}")
    return tok, f


class _SrcWriter:
    """Buffers documents, tokenizes them in batches (multi-threaded Rust tokenizer) and appends uint16 token ids (+EOS) to the train / validation shard files, honouring the token caps."""
    def __init__(self, tok, eos, ftr, fva, cap, val_cap):
        self.tok, self.eos, self.ftr, self.fva = tok, eos, ftr, fva
        self.cap, self.val_cap = cap, val_cap
        self.n_tr = self.n_va = self.val_words = 0
        self.buf = []

    def add(self, text, is_val):
        self.buf.append((text, is_val))
        if len(self.buf) >= 512:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        encs = self.tok.encode_batch([t for t, _ in self.buf])
        for (t, is_val), e in zip(self.buf, encs):
            arr = np.asarray(e.ids + [self.eos], dtype=np.uint16)
            if is_val:
                if self.n_va < self.val_cap:
                    arr.tofile(self.fva)
                    self.n_va += len(arr)
                    self.val_words += len(t.split())
            elif self.n_tr < self.cap:
                arr.tofile(self.ftr)
                self.n_tr += len(arr)
        self.buf = []

    def full(self):
        return self.n_tr >= self.cap and self.n_va >= self.val_cap


def prepare_source(cfg: Cfg, name: str, cap: int, tok, eos: int, ddir: Path):
    """Stream one source file, hash-split documents into train/validation, subsample uniformly, tokenize and write shards; also builds (512-word prompt -> 128-word reference) evaluation examples from held-out documents."""
    path = resolve_file(cfg, SOURCE_FILES[name])
    est_tokens = os.path.getsize(path) / 4.2
    keep = min(1.0, 1.1 * cap / max(1.0, cfg.prep_coverage * est_tokens))   # uniform subsample => no bias towards file start
    log(f"[prepare] {name}: cap={cap:,} tokens, keep-prob={keep:.3f}")
    tr_tmp, va_tmp = ddir / f"{name}.train.bin.tmp", ddir / f"{name}.val.bin.tmp"
    P, R = cfg.eval_prompt_words, cfg.eval_ref_words
    eval_items, docs = [], 0
    with open(tr_tmp, "wb") as ftr, open(va_tmp, "wb") as fva:
        w = _SrcWriter(tok, eos, ftr, fva, cap, cfg.val_tokens_per_source)
        t0 = time.time()
        for i, text in enumerate(iter_texts(path, cfg.json_loader)):
            if cfg.normalize_ws:
                text = norm_ws(text)
            words = text.split()
            if len(words) < cfg.min_words:
                continue
            h = zlib.crc32(f"{cfg.split_seed}:{name}:{i}".encode()) % 100000
            is_val = h < cfg.val_frac * 100000
            if not is_val:
                if (zlib.crc32(f"keep:{name}:{i}".encode()) % 100000) >= keep * 100000:
                    continue
            else:
                if len(words) >= P + R and len(eval_items) < cfg.eval_docs_per_source:
                    k = len(eval_items)
                    span = len(words) - (P + R)
                    start = 0 if (k % 2 == 0 or span == 0) else zlib.crc32(f"s:{name}:{i}".encode()) % (span + 1)
                    eval_items.append({"prompt": " ".join(words[start:start + P]),
                                       "reference": " ".join(words[start + P:start + P + R]),
                                       "source": name, "start": start})
            docs += 1
            w.add(text, is_val)
            if docs % 5000 == 0:
                log(f"[prepare] {name}: docs={docs:,} train_tok={w.n_tr:,} val_tok={w.n_va:,} ({time.time()-t0:.0f}s)")
            if not w.buf and w.full() and len(eval_items) >= cfg.eval_docs_per_source:
                break
        w.flush()
    os.replace(tr_tmp, ddir / f"{name}.train.bin")
    os.replace(va_tmp, ddir / f"{name}.val.bin")
    jdump(eval_items, ddir / f"{name}.eval.json")
    info = {"train_tokens": w.n_tr, "val_tokens": w.n_va, "val_words": w.val_words,
            "docs": docs, "eval_items": len(eval_items)}
    jdump(info, ddir / f"{name}.done.json")
    log(f"[prepare] {name}: {info}")
    return info


def prepare(cfg: Cfg) -> Path:
    """Prepare all sources (cached by settings signature): tokenizer, uint16 shards, validation shard, eval.json / eval_truth.json, meta.json. Returns the data directory."""
    tok, tok_file = build_tokenizer(cfg)
    eos = tok.token_to_id(EOS)
    sig = short_hash({"tok": str(tok_file), "caps": {k: int(v) for k, v in cfg.sources.items()},
                      "norm": cfg.normalize_ws, "minw": cfg.min_words, "vf": cfg.val_frac,
                      "vt": cfg.val_tokens_per_source, "ed": cfg.eval_docs_per_source,
                      "pw": cfg.eval_prompt_words, "rw": cfg.eval_ref_words, "ss": cfg.split_seed, "cov": cfg.prep_coverage})
    ddir = Path(cfg.work_dir) / "data" / sig
    ddir.mkdir(parents=True, exist_ok=True)
    if (ddir / "meta.json").exists():
        log(f"[prepare] cached data found: {ddir}")
        return ddir
    infos, evals = {}, []
    for name, cap in cfg.sources.items():
        if name not in SOURCE_FILES:
            raise SystemExit(f"Unknown source {name!r}; valid: {list(SOURCE_FILES)}")
        done = ddir / f"{name}.done.json"
        if done.exists():
            infos[name] = jload(done)
            log(f"[prepare] {name}: cached {infos[name]}")
        else:
            infos[name] = prepare_source(cfg, name, int(cap), tok, eos, ddir)
        evals += jload(ddir / f"{name}.eval.json")
    with open(ddir / "val.bin", "wb") as out:
        for name in infos:
            with open(ddir / f"{name}.val.bin", "rb") as f:
                shutil.copyfileobj(f, out)
    random.Random(cfg.split_seed).shuffle(evals)
    ev = {f"v{i}": e for i, e in enumerate(evals)}
    jdump(ev, ddir / "eval.json")
    jdump({k: {"text": v["reference"]} for k, v in ev.items()}, ddir / "eval_truth.json")   # for evaluate_phase1.py
    meta = {"sources": infos, "vocab": tok.get_vocab_size(), "eos": eos, "tok_file": str(tok_file),
            "val_tokens": sum(i["val_tokens"] for i in infos.values()),
            "val_words": sum(i["val_words"] for i in infos.values()),
            "n_eval_prompts": len(ev)}
    jdump(meta, ddir / "meta.json")
    tpw = meta["val_tokens"] / max(1, meta["val_words"])
    log(f"[prepare] done. tokens/word on validation = {tpw:.3f}; eval prompts = {len(ev)}")
    return ddir


# ============================================================================================
# 3. Models: gpt2 | llama | retnet | moe  (all with KV / recurrent cache and left-padding support)
# ============================================================================================
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias); computed in fp32 for stability under mixed precision."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.w.to(x.dtype)


def rope_tables(dim, maxlen, base=10000.0):
    """Pre-compute rotary-position-embedding cos/sin tables for positions [0, maxlen)."""
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    f = torch.outer(torch.arange(maxlen).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):           # x: (B,H,T,hd)  cos/sin: (B,1,T,hd/2)
    """Rotate query/key halves by position-dependent angles (RoPE). x: (B,H,T,hd); cos/sin: (B,1,T,hd/2)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class Attn(nn.Module):
    """Causal multi-head / grouped-query self-attention using PyTorch SDPA (flash / memory-efficient kernels), with an optional KV cache. gpt2: MHA with bias and no RoPE; llama/moe: RoPE + GQA, no bias."""
    def __init__(self, mc):
        super().__init__()
        d, gpt2 = mc["n_embd"], mc["arch"] == "gpt2"
        self.h = mc["n_head"]
        self.kvh = self.h if gpt2 else mc["n_kv_head"]
        self.hd = d // self.h
        self.rope = not gpt2
        self.p = mc.get("dropout", 0.0)
        self.q = nn.Linear(d, self.h * self.hd, bias=gpt2)
        self.k = nn.Linear(d, self.kvh * self.hd, bias=gpt2)
        self.v = nn.Linear(d, self.kvh * self.hd, bias=gpt2)
        self.o = nn.Linear(d, d, bias=gpt2)

    def forward(self, x, rope, amask, valid, cache):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        if self.rope:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if self.kvh != self.h:
            r = self.h // self.kvh
            k, v = k.repeat_interleave(r, dim=1), v.repeat_interleave(r, dim=1)
        if amask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class Retention(nn.Module):
    """RetNet-style multi-scale retention. Parallel form for training / prefill, recurrent form for decoding."""

    def __init__(self, mc):
        super().__init__()
        d = mc["n_embd"]
        self.h = mc["n_head"]
        self.hd = d // self.h
        self.q, self.k, self.v, self.g, self.o = (nn.Linear(d, d, bias=False) for _ in range(5))
        self.qn, self.kn, self.on = RMSNorm(self.hd), RMSNorm(self.hd), RMSNorm(self.hd)
        self.register_buffer("gamma", 1.0 - 2.0 ** (-5.0 - torch.arange(self.h).float()), persistent=False)
        self._D = {}

    def decay_matrix(self, T, device, dtype):
        key = (T, str(device), dtype)
        if key not in self._D:
            i = torch.arange(T, device=device)
            diff = i[:, None] - i[None, :]
            D = (self.gamma.view(-1, 1, 1) ** diff.clamp(min=0).float()) * (diff >= 0)
            self._D = {key: D.to(dtype)}          # keep only the most recent size
        return self._D[key]

    def forward(self, x, rope, amask, valid, cache):
        B, T, d = x.shape
        H, hd = self.h, self.hd
        q = self.qn(self.q(x).view(B, T, H, hd).transpose(1, 2)) * (hd ** -0.5)
        k = self.kn(self.k(x).view(B, T, H, hd).transpose(1, 2))
        v = self.v(x).view(B, T, H, hd).transpose(1, 2)
        g = self.g(x)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if valid is not None:                      # left-padding: padded tokens contribute nothing
            vm = valid[:, None, :, None].to(k.dtype)
            k, v = k * vm, v * vm
        dev_type = "cuda" if x.is_cuda else "cpu"
        if cache is not None and "S" in cache:     # recurrent decoding step
            assert T == 1, "recurrent decoding expects one token at a time"
            with torch.autocast(device_type=dev_type, enabled=False):
                S = self.gamma.view(1, -1, 1, 1) * cache["S"] + k.float().transpose(-1, -2) @ v.float()
                o = q.float() @ S
            cache["S"] = S
        else:                                      # parallel form
            D = self.decay_matrix(T, x.device, q.dtype)
            o = ((q @ k.transpose(-1, -2)) * D) @ v
            if cache is not None:                  # state after the prefill, for later recurrent steps
                with torch.autocast(device_type=dev_type, enabled=False):
                    dec = self.gamma.view(-1, 1, 1) ** (T - 1 - torch.arange(T, device=x.device).float()).view(1, -1, 1)
                    cache["S"] = (k.float() * dec[None]).transpose(-1, -2) @ v.float()
        o = self.on(o).transpose(1, 2).reshape(B, T, d)
        return self.o(o * F.silu(g))


class MLP(nn.Module):               # GPT-2 feed-forward
    """GPT-2 feed-forward block: Linear(d,4d) -> GELU -> Linear(4d,d)."""

    def __init__(self, mc):
        super().__init__()
        d = mc["n_embd"]
        self.fc = nn.Linear(d, 4 * d)
        self.w2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.w2(F.gelu(self.fc(x)))


class SwiGLU(nn.Module):
    """Gated feed-forward block: W2( SiLU(W1 x) * W3 x ) as used in LLaMA-style models."""
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Top-k routed mixture of SwiGLU experts with a Switch-style load-balancing loss."""

    def __init__(self, mc):
        super().__init__()
        d, E = mc["n_embd"], mc["n_experts"]
        self.E, self.k = E, mc["top_k"]
        self.router = nn.Linear(d, E, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d, mc["ffn_hidden"]) for _ in range(E)])
        self.aux = torch.zeros(())

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        probs = self.router(xf).float().softmax(-1)
        w, idx = probs.topk(self.k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for e, ex in enumerate(self.experts):
            tok_idx, slot = (idx == e).nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            y = ex(xf[tok_idx]) * w[tok_idx, slot].unsqueeze(-1).to(out.dtype)
            out.index_add_(0, tok_idx, y.to(out.dtype))
        frac = F.one_hot(idx, self.E).float().sum(1).mean(0) / self.k
        self.aux = self.E * (frac * probs.mean(0)).sum()
        return out.view(B, T, D)


class Block(nn.Module):
    """Pre-norm residual transformer block: x + mixer(norm(x)), then x + ffn(norm(x)). The mixer is attention or retention; the ffn is MLP / SwiGLU / MoE depending on the architecture."""
    def __init__(self, mc):
        super().__init__()
        arch, d = mc["arch"], mc["n_embd"]
        Norm = nn.LayerNorm if arch == "gpt2" else RMSNorm
        self.n1, self.n2 = Norm(d), Norm(d)
        self.mix = Retention(mc) if arch == "retnet" else Attn(mc)
        if arch == "gpt2":
            self.ffn = MLP(mc)
        elif arch == "moe":
            self.ffn = MoE(mc)
        else:
            self.ffn = SwiGLU(d, mc["ffn_hidden"])
        self.drop = nn.Dropout(mc.get("dropout", 0.0))

    def forward(self, x, rope, amask, valid, cache):
        x = x + self.drop(self.mix(self.n1(x), rope, amask, valid, cache))
        return x + self.drop(self.ffn(self.n2(x)))


class LM(nn.Module):
    """Decoder-only language model shared by all four architectures (gpt2 | llama | retnet | moe): tied input/output embeddings, optional learned positions, stack of Blocks, final norm. Supports teacher-forced training, and incremental decoding with left-padded batches via new_cache() / forward(cache=..., kmask=...)."""
    def __init__(self, mc):
        super().__init__()
        self.mc = mc
        d, V, arch, L = mc["n_embd"], mc["vocab"], mc["arch"], mc["n_layer"]
        self.wte = nn.Embedding(V, d)
        self.wpe = nn.Embedding(mc["ctx"], d) if arch == "gpt2" else None
        self.drop = nn.Dropout(mc.get("dropout", 0.0))
        self.blocks = nn.ModuleList([Block(mc) for _ in range(L)])
        self.norm_f = nn.LayerNorm(d) if arch == "gpt2" else RMSNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.wte.weight           # tied embeddings
        if arch == "gpt2":
            self.rope_cos = self.rope_sin = None
        else:
            cos, sin = rope_tables(d // mc["n_head"], mc["ctx"])
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("o.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * L))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def new_cache(self):
        return {"past": 0, "layers": [{} for _ in self.blocks]}

    def forward(self, idx, targets=None, pos=None, kmask=None, cache=None, last_only=False):
        B, T = idx.shape
        dev = idx.device
        past = cache["past"] if cache is not None else 0
        if pos is None:
            pos = torch.arange(past, past + T, device=dev).unsqueeze(0).expand(B, -1)
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(pos)
        x = self.drop(x)
        rope = None
        if self.rope_cos is not None:
            rope = (self.rope_cos[pos].unsqueeze(1), self.rope_sin[pos].unsqueeze(1))
        amask = valid = None
        if kmask is not None:
            S = past + T
            qi = torch.arange(past, past + T, device=dev).unsqueeze(1)
            kj = torch.arange(S, device=dev).unsqueeze(0)
            amask = ((kj <= qi)[None, None] & kmask[:, None, None, :]) | (kj == qi)[None, None]
            valid = kmask[:, -T:]
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope, amask, valid, cache["layers"][i] if cache is not None else None)
        if cache is not None:
            cache["past"] = past + T
        x = self.norm_f(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        if self.training and self.mc["arch"] == "moe":
            aux = sum(b.ffn.aux for b in self.blocks) / len(self.blocks)
            loss = loss + self.mc.get("aux_coef", 0.01) * aux
        return None, loss


def count_params(m) -> int:
    """Number of unique parameters (tied embeddings are counted once)."""
    return sum(p.numel() for p in m.parameters())


def model_config(cfg: Cfg, vocab: int) -> dict:
    """Resolve the final model configuration: SIZE_PRESETS[size][arch] updated by cfg.model overrides, plus ctx / vocab / dropout. Validates head dimensions."""
    if cfg.arch not in ARCH_PRESETS:
        raise SystemExit(f"Unknown arch {cfg.arch!r}; valid: {list(ARCH_PRESETS)}")
    if cfg.size not in SIZE_PRESETS:
        raise SystemExit(f"Unknown size {cfg.size!r}; valid: {list(SIZE_PRESETS)}")
    mc = dict(SIZE_PRESETS[cfg.size][cfg.arch])
    mc.update(cfg.model)
    mc.update(arch=cfg.arch, ctx=int(cfg.ctx), vocab=int(vocab), dropout=float(cfg.dropout))
    mc.setdefault("n_kv_head", mc["n_head"])
    assert mc["n_embd"] % mc["n_head"] == 0 and mc["n_head"] % mc["n_kv_head"] == 0
    assert (mc["n_embd"] // mc["n_head"]) % 2 == 0, "head dim must be even (RoPE)"
    return mc


# ============================================================================================
# 4. Training data loader (deterministic + resumable) and training loop
# ============================================================================================
class MixLoader:
    """Weighted mixture of uint16 token shards. Per-sample source assignment and per-source block order are
    pure functions of (seed, sample index) => identical batches after a resume. Blocks are drawn without
    replacement within an epoch of each source."""

    def __init__(self, bins, weights, T, seed, total_samples):
        self.names = list(bins)
        self.T, self.seed = T, seed
        self.mm = [np.memmap(bins[n], dtype=np.uint16, mode="r") for n in self.names]
        self.nblk = [(len(m) - 1) // T for m in self.mm]
        w = np.array([weights[n] for n in self.names], dtype=np.float64)
        w = w / w.sum()
        self.src = np.random.default_rng(seed).choice(len(self.names), size=total_samples, p=w).astype(np.int16)
        self.cnt = np.zeros(total_samples, dtype=np.int64)
        for k in range(len(self.names)):
            m = np.nonzero(self.src == k)[0]
            self.cnt[m] = np.arange(len(m))
        self._perm = {}
        for k, n in enumerate(self.names):
            used = int((self.src == k).sum())
            if used > 2 * self.nblk[k]:
                log(f"[data] WARNING: source {n} will be seen ~{used / self.nblk[k]:.1f} times; prepare more tokens or lower its weight")

    def _block(self, k, c):
        n = self.nblk[k]
        ep = c // n
        if (k, ep) not in self._perm:
            self._perm = {kk: v for kk, v in self._perm.items() if kk[0] != k}
            self._perm[(k, ep)] = np.random.default_rng([self.seed, k, ep]).permutation(n)
        return int(self._perm[(k, ep)][c % n])

    def batch(self, g0, n):
        arr = np.empty((n, self.T + 1), dtype=np.int64)
        for j in range(n):
            g = g0 + j
            k = int(self.src[g])
            b = self._block(k, int(self.cnt[g]))
            arr[j] = self.mm[k][b * self.T: b * self.T + self.T + 1]
        t = torch.from_numpy(arr)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


def lr_at(step, total, cfg: Cfg):
    """Learning rate at a given step: linear warm-up followed by cosine decay to min_lr_ratio * lr."""
    w = min(cfg.warmup_steps, max(1, total // 10))
    if step < w:
        return cfg.lr * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def make_opt(model, cfg: Cfg, dev):
    """AdamW (betas 0.9/0.95) with weight decay on matrices only; fused kernel when available."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    other = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}]
    kw = dict(lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
    if dev == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True, **kw)
        except Exception:
            pass
    return torch.optim.AdamW(groups, **kw)


def make_scaler(enabled):
    """GradScaler for fp16 training (disabled for bf16 / CPU); handles old and new torch APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(model, val_mm, T, n_blocks, mb, dev, amp):
    """Mean cross-entropy over evenly spaced validation blocks (fixed set => comparable across checkpoints)."""
    model.eval()
    nb = (len(val_mm) - 1) // T
    idxs = np.linspace(0, nb - 1, min(n_blocks, nb)).astype(int)
    tot, cnt = 0.0, 0
    for s in range(0, len(idxs), mb):
        ch = idxs[s:s + mb]
        arr = np.stack([val_mm[i * T: i * T + T + 1] for i in ch]).astype(np.int64)
        x = torch.from_numpy(arr[:, :-1]).to(dev)
        y = torch.from_numpy(arr[:, 1:]).to(dev)
        with amp():
            _, loss = model(x, y)
        tot += float(loss) * len(ch)
        cnt += len(ch)
    model.train()
    return tot / cnt


def save_ckpt(path, model, mc, state, opt=None, scaler=None):
    """Atomically save model (+ optimizer / scaler for resumable checkpoints), model config and training state."""
    obj = {"model": model.state_dict(), "mc": mc, "state": dict(state)}
    if opt is not None:
        obj["opt"] = opt.state_dict()
        obj["scaler"] = scaler.state_dict()
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, str(path))


def amp_setup(dev):
    """Pick mixed precision: bf16 on Ampere or newer, fp16 on older GPUs (T4/P100), none on CPU. Returns (dtype, autocast factory)."""
    if dev != "cuda":
        return None, contextlib.nullcontext
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return dtype, (lambda: torch.autocast("cuda", dtype=dtype))


def train(cfg: Cfg) -> bool:
    """Returns True if training finished, False if it stopped early (wall limit / Ctrl-C) and can be resumed."""
    ddir = prepare(cfg)
    meta = jload(ddir / "meta.json")
    rd = run_dir_of(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype, amp = amp_setup(dev)
    V = pad64(meta["vocab"])
    mc = model_config(cfg, V)
    T = mc["ctx"]
    model = LM(mc).to(dev)
    if cfg.init_from and not (rd / "ckpt_last.pt").exists():      # fresh run that continues an earlier Phase-1 model
        ck0 = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        diff = [k for k in ("arch", "n_embd", "n_layer", "n_head", "vocab", "ctx") if ck0["mc"].get(k) != mc.get(k)]
        if diff:
            raise SystemExit(f"init_from checkpoint differs from this run's model in {diff}: use the same arch / size / tokenizer settings")
        model.load_state_dict(ck0["model"])
        log(f"[train] initialised from {cfg.init_from} (step {ck0.get('state', {}).get('step', '?')}) - continued pre-training; "
            f"use a lower lr / short warm-up and a new seed so that new text is seen")
        del ck0
    n_params = count_params(model)
    log(f"[train] run={cfg.run} arch={cfg.arch} params={n_params/1e6:.1f}M ctx={T} device={dev} amp={amp_dtype}")
    if not (50e6 <= n_params <= 300e6) and "smoke" not in cfg.run and not cfg.run.startswith("selftest"):
        log("[train] WARNING: parameter count is outside the 50M-300M competition range!")

    jdump(asdict(cfg), rd / "config.json")
    shutil.copy(meta["tok_file"], rd / "tokenizer.json")
    shutil.copy(ddir / "eval.json", rd / "eval.json")
    shutil.copy(ddir / "eval_truth.json", rd / "eval_truth.json")
    jdump(meta, rd / "meta.json")

    weights = {n: float(cfg.mix.get(n, 0.0)) for n, i in meta["sources"].items() if i["train_tokens"] > T + 1}
    weights = {n: w for n, w in weights.items() if w > 0} or \
              {n: 1.0 for n, i in meta["sources"].items() if i["train_tokens"] > T + 1}
    if not weights:
        raise SystemExit("No training tokens available. Check sources / token caps.")
    bins = {n: ddir / f"{n}.train.bin" for n in weights}
    val_mm = np.memmap(ddir / "val.bin", dtype=np.uint16, mode="r")
    if len(val_mm) < T + 2:
        raise SystemExit("Validation shard too small for this context length.")
    tpw = meta["val_tokens"] / max(1, meta["val_words"])

    opt = make_opt(model, cfg, dev)
    scaler = make_scaler(dev == "cuda" and amp_dtype == torch.float16)
    state = {"step": 0, "best_val": float("inf"), "elapsed": 0.0, "total_steps": 0, "micro_bs": 0, "accum": 0}
    last_path, best_path = rd / "ckpt_last.pt", rd / "ckpt_best.pt"
    if last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        log(f"[train] RESUMED from step {state['step']}/{state['total_steps']}")

    # ---- fit parallelism / micro-batch / step budget to this machine (fresh runs only)
    n_gpu = torch.cuda.device_count() if dev == "cuda" else 0
    if state["total_steps"] == 0:
        def bench(m, mb):
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            x = torch.randint(0, V, (mb, T + 1), device=dev)
            m.train()
            t0 = time.time()
            for i in range(5):
                if i == 2:
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                with amp():
                    _, loss = m(x[:, :-1], x[:, 1:])
                if loss.dim() > 0:                      # DataParallel returns one loss per GPU
                    loss = loss.mean()
                scaler.scale(loss).backward()
                model.zero_grad(set_to_none=True)
            if dev == "cuda":
                torch.cuda.synchronize()
            return 3 * mb * T / (time.time() - t0)

        def make_cands(dp):
            if dev == "cpu":
                return [min(cfg.max_micro_bs, 4)]
            unit, cap_ = (n_gpu, cfg.max_micro_bs * n_gpu) if dp else (1, cfg.max_micro_bs)
            return [m for m in (64, 32, 16, 8, 4, 2, 1) if m % unit == 0 and m <= cap_] or [unit]

        def cleanup():
            model.zero_grad(set_to_none=True)
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()

        def tune(bm, cands):
            for c in cands:
                oom = False
                try:
                    return c, bench(bm, c)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    oom = True
                if oom:
                    cleanup()
            return None, None

        use_dp = bool(cfg.multi_gpu) and n_gpu > 1
        bm = nn.DataParallel(model) if use_dp else model
        try:
            mb, tps = tune(bm, make_cands(use_dp))
        except Exception as e:
            if not use_dp:
                raise
            log(f"[train] DataParallel failed ({type(e).__name__}: {str(e)[:120]}); falling back to a single GPU")
            use_dp, bm = False, model
            cleanup()
            mb, tps = tune(bm, make_cands(False))
        if mb is None:
            raise SystemExit("Even the smallest micro-batch does not fit in GPU memory; reduce ctx or model size.")
        unit = n_gpu if use_dp else 1
        if dev == "cuda" and mb > unit:
            if torch.cuda.max_memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                half = mb // 2
                if half >= unit and half % unit == 0:
                    mb = half
        compiled = False
        if (not use_dp and dev == "cuda" and str(cfg.compile).lower() in ("auto", "true", "on", "1")
                and cfg.arch in ("gpt2", "llama")):
            try:
                tps_c = bench(torch.compile(model), mb)
                log(f"[train] torch.compile {tps_c:,.0f} tok/s vs eager {tps:,.0f} tok/s")
                if tps_c > 1.08 * tps:
                    compiled, tps = True, tps_c
            except Exception as e:
                log(f"[train] torch.compile unavailable ({type(e).__name__}); using eager mode")
            cleanup()
        accum = max(1, round(cfg.batch_tokens / (mb * T)))
        tps_step = accum * mb * T
        if cfg.train_tokens > 0:
            total = max(1, math.ceil(cfg.train_tokens / tps_step))
        elif cfg.time_budget_hours > 0:
            total = max(50, int(cfg.time_budget_hours * 3600 * tps * 0.92 / tps_step))
        else:
            raise SystemExit("Set time_budget_hours > 0 or train_tokens > 0")
        state.update(total_steps=total, micro_bs=mb, accum=accum, use_dp=use_dp, compile=compiled)
        log(f"[train] measured {tps:,.0f} tok/s | gpus={n_gpu} dp={use_dp} compile={compiled} | micro_bs={mb} "
            f"accum={accum} ({tps_step:,} tok/step) | total_steps={total:,} "
            f"(~{total * tps_step / 1e6:,.0f}M tokens, ~{total * tps_step / tps / 3600:.2f} h)")
        cleanup()
        jdump(state, rd / "plan.json")

    mb, accum, total = state["micro_bs"], state["accum"], state["total_steps"]
    loader = MixLoader(bins, weights, T, cfg.seed, total * accum * mb)
    log(f"[train] mixture (normalised): { {n: round(w / sum(weights.values()), 3) for n, w in weights.items()} }")
    use_dp = bool(state.get("use_dp", False)) and n_gpu > 1
    if use_dp:
        cmodel = nn.DataParallel(model)
    elif state.get("compile") and dev == "cuda":
        cmodel = torch.compile(model)
    else:
        cmodel = model
    log(f"[train] parallel={'DataParallel x%d' % n_gpu if use_dp else 'single'} compile={bool(state.get('compile'))}")

    model.train()
    t_session, last_ck, t_log = time.time(), time.time(), time.time()
    loss_acc, n_acc, step = torch.zeros((), device=dev), 0, state["step"]
    elapsed0 = state["elapsed"]
    logf = open(rd / "log.jsonl", "a")
    finished = True
    try:
        while step < total:
            lr = lr_at(step, total, cfg)
            for gp in opt.param_groups:
                gp["lr"] = lr
            for micro in range(accum):
                x, y = loader.batch((step * accum + micro) * mb, mb)
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with amp():
                    _, loss = cmodel(x, y)
                if loss.dim() > 0:
                    loss = loss.mean()
                scaler.scale(loss / accum).backward()
                loss_acc += loss.detach()
                n_acc += 1
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            state["step"] = step
            state["elapsed"] = elapsed0 + time.time() - t_session

            if step % cfg.log_every == 0 or step == total:
                l = float(loss_acc) / n_acc
                if not math.isfinite(l):
                    raise RuntimeError("Loss is NaN/inf. Lower lr, or use a GPU with bf16 support.")
                dt = time.time() - t_log
                tok_s = cfg.log_every * accum * mb * T / max(dt, 1e-9) if step % cfg.log_every == 0 else 0
                eta = (total - step) * (dt / cfg.log_every) / 3600 if tok_s else 0
                log(f"step {step}/{total} loss {l:.4f} lr {lr:.2e} gnorm {float(gn):.2f} {tok_s:,.0f} tok/s eta {eta:.2f}h")
                logf.write(json.dumps({"type": "train", "step": step, "loss": l, "lr": lr, "tok_s": tok_s}) + "\n")
                logf.flush()
                loss_acc.zero_()
                n_acc, t_log = 0, time.time()

            if step % cfg.eval_every == 0 or step == total:
                vl = evaluate(model, val_mm, T, cfg.eval_blocks, mb, dev, amp)
                rec = {"type": "eval", "step": step, "val_loss": vl, "val_ppl": math.exp(vl),
                       "val_ppl_word": math.exp(vl * tpw)}
                log(f"  EVAL step {step}: val_loss {vl:.4f} ppl {rec['val_ppl']:.2f} (per-word ppl {rec['val_ppl_word']:.1f})")
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if vl < state["best_val"]:
                    state["best_val"] = vl
                    save_ckpt(best_path, model, mc, state)
                t_log = time.time()

            if time.time() - last_ck > cfg.ckpt_every_min * 60:
                save_ckpt(last_path, model, mc, state, opt, scaler)
                last_ck = time.time()
            if cfg.max_wall_hours > 0 and time.time() - t_session > cfg.max_wall_hours * 3600 and step < total:
                log(f"[train] wall limit reached at step {step}; checkpointing. Re-run the same command to resume.")
                finished = False
                break
    except KeyboardInterrupt:
        log("[train] interrupted; checkpointing. Re-run the same command to resume.")
        finished = False
    finally:
        logf.close()
    # reached only on a clean finish / wall limit / Ctrl-C (a crash such as NaN loss propagates and does NOT
    # overwrite the last good checkpoint)
    save_ckpt(last_path, model, mc, state, opt, scaler)
    if not finished:
        return False

    ev = [json.loads(l) for l in open(rd / "log.jsonl") if '"eval"' in l]
    last_ev = ev[-1] if ev else {}
    jdump({"params": n_params, "steps": step, "best_val_loss": state["best_val"],
           "last": last_ev, "train_hours": state["elapsed"] / 3600}, rd / "train_summary.json")
    log_experiment(cfg, {"stage": "train", "params_M": round(n_params / 1e6, 1), "steps": step,
                         "val_loss": round(state["best_val"], 4), "val_ppl": round(math.exp(state["best_val"]), 2),
                         "val_ppl_word": round(math.exp(state["best_val"] * tpw), 1)})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = [json.loads(l) for l in open(rd / "log.jsonl") if '"train"' in l]
        plt.figure(figsize=(9, 4.5))
        plt.plot([r["step"] for r in tr], [r["loss"] for r in tr], label="train loss")
        plt.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], label="val loss", marker="o")
        plt.xlabel("step"); plt.ylabel("loss"); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(rd / "loss.png", dpi=150)
        plt.close()
    except Exception as e:
        log(f"[train] (plot skipped: {e})")
    log(f"[train] finished. best val loss {state['best_val']:.4f}")
    return True


# ============================================================================================
# 5. Generation (batched, left-padded, KV/recurrent cache, exact word-count output)
# ============================================================================================
def load_for_inference(cfg: Cfg, which="best"):
    """Load best (or last) checkpoint + tokenizer of a run onto the GPU in eval mode."""
    from tokenizers import Tokenizer
    rd = run_dir_of(cfg)
    p = rd / f"ckpt_{which}.pt"
    if not p.exists():
        p = rd / "ckpt_last.pt"
    if not p.exists():
        raise SystemExit(f"No checkpoint in {rd}. Run the train stage first.")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = LM(ck["mc"])
    model.load_state_dict(ck["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    tok = Tokenizer.from_file(str(rd / "tokenizer.json"))
    eos = tok.token_to_id(EOS)
    amp_dtype, _ = amp_setup(dev)
    log(f"[infer] loaded {p.name} (step {ck['state']['step']}) on {dev}")
    return model, tok, eos, dev, amp_dtype


@torch.no_grad()
def _gen_batch(model, tok, ids_list, dec, cfg: Cfg, dev, amp_dtype, eos):
    """Generate continuations for one batch of prompts: left-padded prefill, then token-by-token decoding with KV / recurrent cache, EOS suppression, repetition penalty, no-repeat n-gram, temperature / top-k / top-p; stops once every row has > target_words words; returns exactly target_words words each."""
    B = len(ids_list)
    L = max(len(x) for x in ids_list)
    inp = torch.zeros((B, L), dtype=torch.long)
    km = torch.zeros((B, L), dtype=torch.bool)
    for b, x in enumerate(ids_list):
        inp[b, L - len(x):] = torch.tensor(x, dtype=torch.long)
        km[b, L - len(x):] = True
    inp, km = inp.to(dev), km.to(dev)
    pos = (km.long().cumsum(1) - 1).clamp(min=0)
    last_pos = pos[:, -1:]
    cache = model.new_cache()
    temp, top_p = float(dec.get("temperature", 0.0)), float(dec.get("top_p", 1.0))
    top_k, rep, nr = int(dec.get("top_k", 0)), float(dec.get("rep", 1.0)), int(dec.get("nr", 0))
    gcpu = torch.Generator(device=dev).manual_seed(cfg.seed)
    gen = torch.zeros((B, 0), dtype=torch.long, device=dev)
    seqs = [[] for _ in range(B)]
    maps = [dict() for _ in range(B)]
    ctxm = torch.autocast("cuda", dtype=amp_dtype) if dev == "cuda" else contextlib.nullcontext()
    target = cfg.target_words
    with ctxm:
        logits, _ = model(inp, pos=pos, kmask=km, cache=cache, last_only=True)
        for t in range(cfg.max_new_tokens):
            lg = logits[:, -1].float().clone()
            lg[:, eos] = -float("inf")                 # never stop early: we need a full 128 words
            if rep != 1.0 and gen.shape[1] > 0:
                sc = lg.gather(1, gen)
                lg.scatter_(1, gen, torch.where(sc < 0, sc * rep, sc / rep))
            if nr >= 2:
                rows, cols = [], []
                for b in range(B):
                    s = seqs[b]
                    if len(s) >= nr - 1:
                        for tk in maps[b].get(tuple(s[-(nr - 1):]), ()):
                            rows.append(b)
                            cols.append(tk)
                if rows:
                    lg[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)] = -float("inf")
            if temp <= 0:
                nxt = lg.argmax(-1)
            else:
                lg = lg / temp
                if top_k > 0:
                    kth = lg.topk(min(top_k, lg.shape[-1]), dim=-1).values[:, -1:]
                    lg = lg.masked_fill(lg < kth, -float("inf"))
                if top_p < 1.0:
                    sl, si = lg.sort(descending=True, dim=-1)
                    pr = sl.softmax(-1)
                    rm = (pr.cumsum(-1) - pr) > top_p
                    sl = sl.masked_fill(rm, -float("inf"))
                    lg = torch.full_like(lg, -float("inf")).scatter(1, si, sl)
                nxt = torch.multinomial(lg.softmax(-1), 1, generator=gcpu)[:, 0]
            nl = nxt.tolist()
            if nr >= 2:
                for b in range(B):
                    s = seqs[b]
                    if len(s) >= nr - 1:
                        maps[b].setdefault(tuple(s[-(nr - 1):]), []).append(nl[b])
            for b in range(B):
                seqs[b].append(nl[b])
            gen = torch.cat([gen, nxt[:, None]], dim=1)
            if (t + 1) % 16 == 0:
                texts = tok.decode_batch(gen.tolist(), skip_special_tokens=True)
                if all(len(x.split()) >= target + 1 for x in texts):
                    break
            km = torch.cat([km, torch.ones((B, 1), dtype=torch.bool, device=dev)], dim=1)
            last_pos = last_pos + 1
            logits, _ = model(nxt[:, None], pos=last_pos, kmask=km, cache=cache, last_only=True)
    texts = tok.decode_batch(gen.tolist(), skip_special_tokens=True)
    return [" ".join(x.split()[:target]) for x in texts]


def generate_texts(model, tok, prompts, dec, cfg: Cfg, dev, amp_dtype, eos, progress=True):
    """Generate for any number of prompts: truncates prompts to ctx - max_new_tokens tokens (keeping the end), sorts by length to minimise padding, batches, and halves the batch size on CUDA OOM. Returns texts in the original order."""
    budget = model.mc["ctx"] - cfg.max_new_tokens
    enc = []
    for p in prompts:
        ids = tok.encode(norm_ws(p) if cfg.normalize_ws else p).ids
        ids = ids[-budget:] if len(ids) > budget else ids
        enc.append(ids if ids else [eos])
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = [None] * len(enc)
    bs, i, t0 = cfg.gen_batch, 0, time.time()
    while i < len(order):
        idxs = order[i:i + bs]
        oom = False
        try:
            texts = _gen_batch(model, tok, [enc[j] for j in idxs], dec, cfg, dev, amp_dtype, eos)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                oom = True
            else:
                raise
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            log(f"[gen] OOM -> batch size {bs}")
            continue
        for j, t in zip(idxs, texts):
            out[j] = t
        i += len(idxs)
        if progress:
            log(f"[gen] {i}/{len(order)} ({time.time() - t0:.0f}s)")
    return out


# ============================================================================================
# 6. Metrics (same formulas as evaluate_phase1.py) + sweep + predict
# ============================================================================================
def lcs_length(a, b):
    """Length of the longest common subsequence of two token lists (bit-parallel algorithm, identical to the official evaluator)."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks = {}
    for j, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << j)
    state = 0
    for t in b:
        m = masks.get(t, 0)
        x = m | state
        y = (state << 1) | 1
        state = x & ~(x - y)
    return bin(state).count("1")


def rouge_l_f1(p, r):
    """ROUGE-L F1 on whitespace tokens (identical to the official evaluator)."""
    pt, rt = p.split(), r.split()
    if not pt or not rt:
        return 0.0
    l = lcs_length(pt, rt)
    if not l:
        return 0.0
    pp, rr = l / len(pt), l / len(rt)
    return 2 * pp * rr / (pp + rr)


def _ngrams(t, n):
    """Counter of n-grams of a token list."""
    from collections import Counter
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1)) if len(t) >= n else Counter()


def bleu4(preds, refs):
    """Corpus-level BLEU-4 without smoothing (identical to the official evaluator)."""
    order, clipped, totals, pl, rl = 4, [0] * 4, [0] * 4, 0, 0
    for p, r in zip(preds, refs):
        pl += len(p)
        rl += len(r)
        for n in range(1, order + 1):
            pn, rn = _ngrams(p, n), _ngrams(r, n)
            totals[n - 1] += sum(pn.values())
            for g, c in pn.items():
                clipped[n - 1] += min(c, rn.get(g, 0))
    prec = [clipped[i] / totals[i] if totals[i] else 0.0 for i in range(order)]
    if any(x <= 0 for x in prec) or pl == 0:
        return 0.0
    bp = 1.0 if pl > rl else math.exp(1.0 - rl / pl)
    return bp * math.exp(sum(math.log(x) for x in prec) / order)


_BERT = None


def bert_f1(preds, refs, dev):
    """Mean BERTScore F1 with roberta-large, layer 17, no baseline rescaling (the official configuration). The scorer is created once and reused."""
    global _BERT
    if _BERT is None:
        from bert_score import BERTScorer
        _BERT = BERTScorer(lang="en", device=dev, batch_size=16)    # roberta-large, layer 17, no rescaling = official
    _, _, f = _BERT.score(preds, refs)
    return float(f.mean().item())


def score_preds(preds, refs, cfg: Cfg, dev):
    """ROUGE-L, BLEU-4 and (if available) BERTScore for predictions vs references; falls back gracefully if BERTScore cannot be loaded."""
    out = {"rougeL": sum(rouge_l_f1(p, r) for p, r in zip(preds, refs)) / len(refs),
           "bleu4": bleu4([p.split() for p in preds], [r.split() for r in refs])}
    if cfg.use_bertscore:
        try:
            out["bertscore"] = bert_f1(preds, refs, dev)
        except Exception as e:
            log(f"[score] BERTScore unavailable ({type(e).__name__}: {e}); selecting by ROUGE-L instead")
            cfg.use_bertscore = False
    return out


def log_experiment(cfg: Cfg, row: dict):
    """Append one row to <work_dir>/experiments.csv (the experiment table for the report)."""
    fields_ = ["time", "run", "stage", "arch", "params_M", "steps", "val_loss", "val_ppl", "val_ppl_word",
               "decode", "bertscore", "rougeL", "bleu4"]
    p = Path(cfg.work_dir) / "experiments.csv"
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": cfg.run, "arch": cfg.arch, **row})


def sweep(cfg: Cfg, which="best") -> dict:
    """Decoding sweep: generate for held-out prompts with every config in sweep_grid, score each, save sweep.json, best_decode.json and a few qualitative samples; returns the best config by select_metric."""
    rd = run_dir_of(cfg)
    model, tok, eos, dev, amp_dtype = load_for_inference(cfg, which)
    ev = jload(rd / "eval.json")
    keys = list(ev)[:cfg.sweep_n]
    prompts, refs = [ev[k]["prompt"] for k in keys], [ev[k]["reference"] for k in keys]
    log(f"[sweep] {len(keys)} held-out prompts, {len(cfg.sweep_grid)} decoding configs")
    results = []
    for dec in cfg.sweep_grid:
        t0 = time.time()
        preds = generate_texts(model, tok, prompts, dec, cfg, dev, amp_dtype, eos, progress=False)
        m = score_preds(preds, refs, cfg, dev)
        r = {"decode": dec, **m, "avg_words": float(np.mean([len(p.split()) for p in preds])),
             "sec": round(time.time() - t0, 1)}
        results.append(r)
        log(f"[sweep] {dec['name']:<22} " + "  ".join(f"{k}={m[k]:.4f}" for k in m) + f"  ({r['sec']}s)")
        log_experiment(cfg, {"stage": "sweep", "decode": dec["name"], **{k: round(v, 5) for k, v in m.items()}})
    key = cfg.select_metric if cfg.select_metric in results[0] else "rougeL"
    best = max(results, key=lambda r: r[key])
    jdump(results, rd / "sweep.json")
    jdump(best["decode"], rd / "best_decode.json")
    # a few qualitative samples for the report
    jdump([{"id": keys[i], "prompt_tail": " ".join(prompts[i].split()[-40:]), "reference": refs[i],
            "generated": generate_texts(model, tok, [prompts[i]], best["decode"], cfg, dev, amp_dtype, eos, False)[0]}
           for i in range(min(5, len(keys)))], rd / "samples.json")
    log(f"[sweep] BEST by {key}: {best['decode']['name']}  ({key}={best[key]:.4f})")
    return best


def predict(cfg: Cfg, which="best") -> str:
    """Generate 128-word continuations for the test prompts with the chosen decoding config, validate the output, write prediction.json and submission.zip (prediction.json at the zip root)."""
    rd = run_dir_of(cfg)
    model, tok, eos, dev, amp_dtype = load_for_inference(cfg, which)
    if cfg.decode:
        dec = cfg.decode
    elif (rd / "best_decode.json").exists():
        dec = jload(rd / "best_decode.json")
    else:
        dec = FALLBACK_DECODE
        log("[predict] no sweep results found; using fallback decoding config")
    log(f"[predict] decoding config: {dec}")
    tp = cfg.test_path or resolve_file(cfg, cfg.test_file)
    data = jload(tp)
    items = list(data.items()) if isinstance(data, dict) else [(str(i + 1), r) for i, r in enumerate(data)]
    ids = [str(k) for k, _ in items]
    prompts = [_rec_text(r) for _, r in items]
    log(f"[predict] {len(prompts)} test prompts from {tp}")
    preds = generate_texts(model, tok, prompts, dec, cfg, dev, amp_dtype, eos)
    out = {}
    for i, p in zip(ids, preds):
        out[i] = {"text": p if p.strip() else "the"}
    bad = [i for i in out if len(out[i]["text"].split()) != cfg.target_words]
    if bad:
        log(f"[predict] WARNING: {len(bad)} outputs do not have exactly {cfg.target_words} words (e.g. ids {bad[:5]})")
    pj = rd / "prediction.json"
    jdump(out, pj)
    zp = rd / "submission.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")        # must sit at the root of the zip
    hist = Path(cfg.work_dir) / "submissions"
    hist.mkdir(parents=True, exist_ok=True)
    shutil.copy(zp, hist / f"{cfg.run}_{time.strftime('%m%d_%H%M%S')}.zip")
    log(f"[predict] wrote {pj}  and  {zp}   ({len(out)} predictions)")
    return str(zp)


# ============================================================================================
# 7. Self-test
# ============================================================================================
def selftest(e2e=True) -> bool:
    """Unit tests (cached/left-padded decoding == full forward pass; each architecture can overfit a batch; parameter counts in range) plus a tiny end-to-end run on synthetic data for every architecture."""
    import tempfile
    torch.manual_seed(0)
    ok = True
    log("=== unit tests: cached / left-padded decoding must equal full forward pass ===")
    for arch in ARCH_PRESETS:
        mc = {**ARCH_PRESETS[arch], "n_embd": 64, "n_layer": 2, "n_head": 4, "n_kv_head": 2, "ffn_hidden": 96,
              "n_experts": 4, "top_k": 2, "arch": arch, "ctx": 64, "vocab": 128, "dropout": 0.0}
        m = LM(mc).eval()
        lens, ext_n = (11, 7), 4
        seqs = [torch.randint(1, 128, (n + ext_n,)) for n in lens]
        with torch.no_grad():
            ref = [m(s[None])[0][0] for s in seqs]                   # (n+ext, V) full forward, no padding
            L = max(lens)
            inp = torch.zeros((2, L), dtype=torch.long)
            km = torch.zeros((2, L), dtype=torch.bool)
            for b, n in enumerate(lens):
                inp[b, L - n:] = seqs[b][:n]
                km[b, L - n:] = True
            pos = (km.long().cumsum(1) - 1).clamp(min=0)
            lp = pos[:, -1:]
            cache = m.new_cache()
            lg, _ = m(inp, pos=pos, kmask=km, cache=cache, last_only=True)
            diffs = [(lg[b, -1] - ref[b][lens[b] - 1]).abs().max().item() for b in range(2)]
            for j in range(ext_n):
                tk = torch.stack([seqs[b][lens[b] + j] for b in range(2)])[:, None]
                km = torch.cat([km, torch.ones((2, 1), dtype=torch.bool)], 1)
                lp = lp + 1
                lg, _ = m(tk, pos=lp, kmask=km, cache=cache, last_only=True)
                diffs += [(lg[b, -1] - ref[b][lens[b] + j]).abs().max().item() for b in range(2)]
        good = max(diffs) < 2e-3
        ok &= good
        log(f"  {arch:<7} max |cached - full| = {max(diffs):.2e}  {'PASS' if good else 'FAIL'}")
        # can it learn? (overfit one random batch)
        m.train()
        o = torch.optim.AdamW(m.parameters(), lr=3e-3)
        x = torch.randint(1, 128, (4, 33))
        l0 = None
        for _ in range(100):
            _, loss = m(x[:, :-1], x[:, 1:])
            l0 = l0 or loss.item()
            o.zero_grad()
            loss.backward()
            o.step()
        good = loss.item() < 0.8 * l0
        ok &= good
        log(f"  {arch:<7} overfit test: loss {l0:.2f} -> {loss.item():.2f}  {'PASS' if good else 'FAIL'}")

    log("=== parameter counts at real size (V=32064) ===")
    for size, presets in SIZE_PRESETS.items():
        for arch in presets:
            try:
                mc = {**presets[arch], "arch": arch, "ctx": 1024, "vocab": 32064, "dropout": 0.0}
                mc.setdefault("n_kv_head", mc["n_head"])
                with torch.device("meta"):
                    n = count_params(LM(mc))
                good = 50e6 <= n <= 300e6
                ok &= good
                log(f"  {size:<5} {arch:<7} {n / 1e6:6.1f}M  {'PASS' if good else 'FAIL (outside 50M-300M)'}")
            except Exception as e:
                log(f"  {size:<5} {arch:<7} (skipped: {type(e).__name__}: {e})")

    if e2e:
        log("=== end-to-end smoke run on synthetic data (prepare -> train -> sweep -> predict) ===")
        tmp = tempfile.mkdtemp(prefix="cs781_selftest_")
        rng = random.Random(0)
        words = ["".join(rng.choice("abcdefghilmnoprstu") for _ in range(rng.randint(2, 8))) for _ in range(300)]

        def doc(n):
            return " ".join(rng.choice(words) for _ in range(n))

        dd = os.path.join(tmp, "data")
        os.makedirs(dd)
        for rel in SOURCE_FILES.values():
            json.dump([{"text": doc(rng.randint(150, 400))} for _ in range(300)],
                      open(os.path.join(dd, os.path.basename(rel)), "w"))
        json.dump({str(i): {"text": doc(60)} for i in range(1, 7)}, open(os.path.join(dd, "test.json"), "w"))
        for arch in ARCH_PRESETS:
            cfg = apply_overrides(Cfg(), PRESETS["smoke"])
            apply_overrides(cfg, {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd,
                                  "run": f"selftest_{arch}", "arch": arch, "train_tokens": 60_000,
                                  "ctx": 192, "max_new_tokens": 96,
                                  "compile": "off",
                                  "sources": {"acts": 3e6, "tribunals": 3e6}, "mix": {"acts": 0.5, "tribunals": 0.5},
                                  "test_path": os.path.join(dd, "test.json"),
                                  "sweep_grid": [DEFAULT_GRID[1], DEFAULT_GRID[4], DEFAULT_GRID[6]]})
            try:
                prepare(cfg)
                assert train(cfg), "train did not finish"
                sweep(cfg)
                zp = predict(cfg)
                with zipfile.ZipFile(zp) as z:
                    assert z.namelist() == ["prediction.json"], z.namelist()
                    pr = json.loads(z.read("prediction.json"))
                assert set(pr) == {str(i) for i in range(1, 7)}
                # an untrained 27-step model can emit few whitespace-separated words, so only check the contract:
                # non-empty, never longer than the target (exact length is expected from a trained model)
                assert all(1 <= len(v["text"].split()) <= cfg.target_words for v in pr.values()), "bad output length"
                log(f"  e2e {arch:<7} PASS")
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  e2e {arch:<7} FAIL: {type(e).__name__}: {e}")
                ok = False
        shutil.rmtree(tmp, ignore_errors=True)
    log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


# ============================================================================================
# 8. CLI
# ============================================================================================
def build_cfg(args) -> Cfg:
    """Merge preset + --cfg overrides; for an existing run, reuse its frozen config and only accept the keys that are safe to change."""
    cfg = Cfg()
    if args.preset:
        apply_overrides(cfg, PRESETS[args.preset])
    ov = {}
    if args.cfg:
        ov = json.loads(args.cfg) if args.cfg.strip().startswith("{") else jload(args.cfg)
    if args.run:
        ov["run"] = args.run
    apply_overrides(cfg, ov)
    saved = run_dir_of(cfg) / "config.json"
    if saved.exists() and not args.fresh and args.stage in ("train", "sweep", "predict", "all"):
        frozen = Cfg()
        apply_overrides(frozen, {k: v for k, v in jload(saved).items() if k in {f.name for f in fields(Cfg)}})
        allowed = RESUME_KEYS if args.stage == "train" else GEN_KEYS
        ignored = []
        for k, v in ov.items():
            if k in allowed:
                setattr(frozen, k, v)
            elif getattr(frozen, k, None) != v:
                ignored.append(k)
        if ignored:
            log(f"[config] run '{cfg.run}' already exists -> using its saved config; ignored: {ignored}. "
                f"(use a new --run name or --fresh to change them)")
        cfg = frozen
    return cfg


def main(argv=None) -> int:
    """Command-line entry point: selftest | prepare | train | sweep | predict | all."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["selftest", "prepare", "train", "sweep", "predict", "all"])
    ap.add_argument("--run", default=None, help="run name (a new name = a fresh experiment)")
    ap.add_argument("--cfg", default="", help="JSON string or path to a JSON file with config overrides")
    ap.add_argument("--preset", default="", choices=["", "full", "fast", "turbo", "smoke"])
    ap.add_argument("--fresh", action="store_true", help="discard checkpoints/logs of this run and start over")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"], help="checkpoint used by sweep/predict")
    ap.add_argument("--no-e2e", action="store_true", help="selftest: skip the synthetic end-to-end part")
    args = ap.parse_args(argv)

    if args.stage == "selftest":
        return 0 if selftest(e2e=not args.no_e2e) else 1
    cfg = build_cfg(args)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    if args.fresh:
        rd = run_dir_of(cfg)
        if rd.exists():
            shutil.rmtree(rd)
            log(f"[fresh] removed {rd}")
    log(f"work_dir={cfg.work_dir} run={cfg.run} arch={cfg.arch}")
    if args.stage == "prepare":
        prepare(cfg)
    elif args.stage == "train":
        return 0 if train(cfg) else 2
    elif args.stage == "sweep":
        sweep(cfg, args.ckpt)
    elif args.stage == "predict":
        predict(cfg, args.ckpt)
    elif args.stage == "all":
        if not train(cfg):
            log("Training not finished -> stop here. Re-run the same command to resume.")
            return 2
        sweep(cfg, args.ckpt)
        predict(cfg, args.ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
