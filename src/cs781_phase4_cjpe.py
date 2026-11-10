#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS781 Phase 4 - Court Judgment Prediction (binary outcome, long legal documents) on top of YOUR Phase-1 language models.

Needs cs781_phase1.py and cs781_phase2_ner.py in the same folder (model classes, LoRA / LoRMA adapters, distillation helper).

Model (hierarchical, long-document aware)
  case text -> Phase-1 BPE tokens -> chunks of max_len tokens (built from the END of the document backwards, so the last chunk is complete)
  -> chunk selection ("select": tail | head | head_tail | uniform) keeps at most max_chunks chunks
  -> Phase-1 backbone (bidirectional attention, or causal for retnet) -> chunk vector = [mean of token states ; state of the closing <eos>]
  -> + chunk-position embeddings (relative position in the case, "is last chunk")
  -> pooling over chunks ("pool": attn | gru | mean | last) -> [pooled ; last-chunk vector] -> MLP -> 2 logits
  Decision threshold tuned on dev for macro-F1.   Output: [{"id": ..., "pred": 0|1}, ...]

Adaptation methods (cfg "method")
  full      all weights                         partial   freeze embeddings + the first `freeze_layers` blocks
  bitfit    only biases + normalisation scales  adapter   Houlsby bottleneck adapters after every attention/FFN sub-layer
  lora      additive low-rank                   lorma     multiplicative low-rank (LoRMA+ / LoRMAπ)
  kd        knowledge distillation: pruned student <- fine-tuned teacher (logits)

Stages:  selftest | check | probe | train | predict | ensemble
  check  : data statistics + duplicate / train-test overlap report      probe : which part of a case carries the label? (diagnostic only)
  python cs781_phase4_cjpe.py check --run cj_check --cfg '{"p1_dir": ".../explore_llama"}'
  python cs781_phase4_cjpe.py train --run cj_full_llama --cfg cfg_cj.json
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
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cs781_phase1 as P1
import cs781_phase2_ner as N

log = P1.log


# ============================================================================================
# 0. Config
# ============================================================================================
@dataclass
class JCfg:
    run: str = "cj_full_v1"
    work_dir: str = P1.DEFAULT_WORK
    p1_dir: str = ""
    p1_ckpt: str = "ckpt_best.pt"
    hf_repo: str = "Exploration-Lab/CS-781-Capstone"
    local_data_dir: str = ""          # folder with train.json, dev.json, test.json
    train_prefix: str = "Phase_4_CJPE/Train"
    test_file: str = "Phase_4_CJPE/Test/test.json"
    max_train_docs: int = 0           # debug / speed: use at most this many training cases (0 = all)
    dedupe_train: bool = True         # drop exact duplicate cases (the single- and multi-instance sets overlap) from train AND dev
    probe_docs: int = 12000           # probe stage: training cases used
    # ---- long-document representation
    max_len: int = 512                # tokens per chunk (<= Phase-1 context)
    max_chunks: int = 8               # chunks per case that go through the backbone
    select: str = "tail"              # tail | head | head_tail | uniform   (which chunks are kept)
    overlap: int = 0                  # tokens shared by consecutive chunks
    head_chunks: int = 0              # select=head_tail: chunks taken from the START of the case (0 = max_chunks // 4)
    pool: str = "attn"                # attn | gru | mean | last   (how chunk vectors are combined)
    chunk_dim: int = 384
    head_dropout: float = 0.1
    bidir: str = "auto"               # auto = bidirectional attention for gpt2/llama/moe, causal for retnet
    # ---- method
    method: str = "full"              # full | partial | bitfit | adapter | lora | lorma | kd
    freeze_layers: int = 0
    adapter_dim: int = 64
    peft_r: int = 16
    peft_alpha: float = 32.0
    peft_dropout: float = 0.05
    peft_lr: float = 5e-4
    peft_targets: str = "all"
    peft_experts: bool = False
    lorma_variant: str = "plus"
    teacher_run: str = ""
    student_layers: int = 0
    kd_alpha: float = 0.5
    kd_temp: float = 2.0
    # ---- optimisation
    epochs: int = 6
    docs_per_epoch: int = 0           # 0 = all training cases per epoch; else a random subset (faster epochs for huge training sets)
    batch_docs: int = 2
    accum: int = 2
    eval_docs: int = 4
    lr: float = 2e-4
    llrd: float = 0.85
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    class_weight: str = "balanced"    # none | balanced
    label_smoothing: float = 0.0
    ema_decay: float = 0.995
    time_budget_min: float = 45.0     # hard wall-clock limit of the training loop (the LR schedule follows whichever of epochs / time ends first)
    dev_max_docs: int = 600           # model selection on a fixed random subset of dev (final scores use all of dev)
    eval_every_steps: int = 3000      # also evaluate every N optimiser steps (0 = only at epoch ends) - time-limited runs never finish an epoch
    patience: int = 4                 # evaluations without improvement before stopping
    seed: int = 1337
    predict_after: bool = True
    # ---- ensemble stage
    ens_runs: list = field(default_factory=list)
    ens_auto: bool = False
    ens_prefix: str = ""
    ens_max: int = 6


def apply_overrides(cfg: JCfg, ov: dict) -> JCfg:
    names = {f.name for f in fields(JCfg)}
    for k, v in ov.items():
        if k not in names:
            raise SystemExit(f"Unknown config key {k!r}. Valid: {sorted(names)}")
        setattr(cfg, k, v)
    return cfg


def run_dir4(cfg: JCfg) -> Path:
    return Path(cfg.work_dir) / "cj_runs" / cfg.run


CJ_FIELDS = ["time", "run", "p1_arch", "method", "pool", "select", "max_chunks", "params_total_M", "params_trainable_M",
             "dev_macro_f1", "dev_precision", "dev_recall", "dev_acc", "threshold", "train_s", "peak_gpu_mb", "infer_docs_s",
             "teacher_run"]


def log_csv(cfg: JCfg, row: dict):
    p = Path(cfg.work_dir) / "cj_experiments.csv"
    if p.exists() and open(p).readline().strip().split(",") != CJ_FIELDS:
        p.rename(p.with_suffix(".old.csv"))
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CJ_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row})


# ============================================================================================
# 1. Data
# ============================================================================================
class JDoc:
    pass


def _first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_split(cfg: JCfg, rel: str, tok, labelled: bool, max_docs: int = 0):
    raw = P1.jload(P1.resolve_file(cfg, rel))
    items = raw if isinstance(raw, list) else [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in raw.items()]
    if max_docs:
        items = items[:max_docs]
    assert tok.get_vocab_size() < 65536, "token ids are stored as uint16"
    ids, texts, labels = [], [], []
    for i, r in enumerate(items):
        did = _first(r, ("id", "case_id", "doc_id", "document_id"))
        ids.append(str(did) if did is not None else str(i))
        t = _first(r, ("text", "document", "case_text", "content")) or ""
        texts.append(" ".join(" ".join(t).split() if isinstance(t, list) else t.split()))
        if labelled:
            y = _first(r, ("label", "gold", "labels", "y", "outcome", "decision"))
            if y is None or isinstance(y, str) and not y.strip().lstrip("-").isdigit() and y not in ("0", "1"):
                raise SystemExit(f"{rel}: case {ids[-1]} has no usable binary label (got {y!r})")
            y = int(float(y))
            if y not in (0, 1):
                raise SystemExit(f"{rel}: case {ids[-1]} has label {y}, expected 0 or 1")
            labels.append(y)
    docs = []
    for b0 in range(0, len(texts), 128):
        for k, e in enumerate(tok.encode_batch(texts[b0:b0 + 128])):
            d = JDoc()
            d.id, d.tok = ids[b0 + k], np.asarray(e.ids, dtype=np.uint16)
            d.n, d.label = len(d.tok), (labels[b0 + k] if labelled else -1)
            docs.append(d)
    if len({d.id for d in docs}) != len(docs):
        log(f"[data] WARNING: duplicate case ids in {rel}")
    return docs


def chunk_bounds(n: int, L: int, overlap: int):
    """Chunks of <= L tokens built from the END of the document backwards (the last chunk is always complete)."""
    if n <= 0:
        return [(0, 0)]
    step = max(1, L - overlap)
    out, e = [], n
    while e > 0:
        s = max(0, e - L)
        out.append((s, e))
        if s == 0:
            break
        e -= step
    return out[::-1]


def select_chunks(m: int, k: int, strategy: str, head: int = 0):
    """Indices (into the m chunks of a case) of the chunks that are kept, in document order."""
    if m <= k:
        return list(range(m))
    if strategy == "head":
        return list(range(k))
    if strategy == "head_tail":
        h = min(k, head if head > 0 else max(1, k // 4))
        return list(range(h)) + list(range(m - (k - h), m))
    if strategy == "uniform":
        return sorted({int(x) for x in np.linspace(0, m - 1, k).round()})
    return list(range(m - k, m))                       # tail (default)


def collate_docs(docs, idxs, cfg: JCfg, dev, eos, n_chunks):
    chunks, owner, slot, rel, last, counts = [], [], [], [], [], []
    for b, di in enumerate(idxs):
        d = docs[di]
        bounds = chunk_bounds(d.n, cfg.max_len - 1, cfg.overlap)
        sel = select_chunks(len(bounds), n_chunks, cfg.select, cfg.head_chunks)
        for k_, j in enumerate(sel):
            s, e = bounds[j]
            chunks.append(d.tok[s:e])
            owner.append(b)
            slot.append(k_)
            rel.append(min(7, int(8 * j / len(bounds))))
            last.append(int(j == len(bounds) - 1))
        counts.append(len(sel))
    L = max(len(c) for c in chunks) + 1
    ids = np.zeros((len(chunks), L), dtype=np.int64)
    km = np.zeros((len(chunks), L), dtype=bool)
    for i, c in enumerate(chunks):
        n = len(c)
        ids[i, :n] = c
        ids[i, n] = eos
        km[i, :n + 1] = True
    y = np.array([docs[di].label for di in idxs], dtype=np.int64)
    t = lambda x: torch.from_numpy(x).to(dev)
    return t(ids), t(km), t(np.array(owner)), t(np.array(slot)), t(np.array(rel)), t(np.array(last)), t(np.array(counts)), t(y)


def dedupe_docs(docs):
    """Drop exact duplicate cases (same token sequence). Returns (kept, n_removed, n_removed_with_conflicting_label)."""
    seen, keep, conflict = {}, [], 0
    for d in docs:
        h = hashlib.md5(d.tok.tobytes()).hexdigest()
        if h in seen:
            conflict += int(seen[h] != d.label)
            continue
        seen[h] = d.label
        keep.append(d)
    return keep, len(docs) - len(keep), conflict


def overlap_report(tr, dv, te):
    """How many dev / test cases also occur in train: identical text, identical first 256 tokens, identical last 256 tokens.
    (A large overlap, especially with agreeing labels, means the benchmark can be 'solved' by memorising train cases.)"""
    def key(d, mode):
        t = d.tok[:256] if mode == "first 256 tokens" else d.tok[-256:] if mode == "last 256 tokens" else d.tok
        return hashlib.md5(t.tobytes()).hexdigest()
    rows = []
    for mode in ("full text", "first 256 tokens", "last 256 tokens"):
        tmap = {}
        for d in tr:
            tmap.setdefault(key(d, mode), []).append(d.label)
        for name, docs in (("dev", dv), ("test", te)):
            hit = [(d, tmap[key(d, mode)]) for d in docs if key(d, mode) in tmap]
            agree = float(np.mean([(np.mean(ls) >= 0.5) == bool(d.label) for d, ls in hit])) if (name == "dev" and hit) else None
            rows.append((name, mode, len(hit), len(docs), agree))
            log(f"[overlap] {name:<4} cases whose {mode:<16} also occurs in train: {len(hit):>5}/{len(docs)}"
                + (f" | train label agrees for {100 * agree:.1f}% of them" if agree is not None else ""))
    return rows


# ============================================================================================
# 2. Metrics (identical definitions to the official evaluator)
# ============================================================================================
def binary_scores(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = np.array([((y == c) & (yhat == c)).sum() for c in (0, 1)], dtype=np.float64)
    fp = np.array([((y != c) & (yhat == c)).sum() for c in (0, 1)], dtype=np.float64)
    fn = np.array([((y == c) & (yhat != c)).sum() for c in (0, 1)], dtype=np.float64)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(2), where=(2 * tp + fp + fn) > 0)
    prec = tp[1] / (tp[1] + fp[1]) if tp[1] + fp[1] else 0.0
    rec = tp[1] / (tp[1] + fn[1]) if tp[1] + fn[1] else 0.0
    return {"macro_f1": float(f1.mean()), "f1_per_class": f1.tolist(), "precision": float(prec), "recall": float(rec),
            "acc": float((y == yhat).mean()) if len(y) else 0.0, "tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist()}


def best_threshold(y, p1, min_gain=0.003):
    base = binary_scores(y, (p1 >= 0.5).astype(int))["macro_f1"]
    best_t, best_f = 0.5, base
    for t in np.arange(0.2, 0.801, 0.02):
        f = binary_scores(y, (p1 >= t).astype(int))["macro_f1"]
        if f > best_f + min_gain:
            best_t, best_f = float(round(t, 2)), f
    return best_t, best_f, base


# ============================================================================================
# 3. Adaptation methods beyond Phase 2: Houlsby adapters, BitFit, layer freezing
# ============================================================================================
class AdapterWrap(nn.Module):
    """Houlsby-style bottleneck adapter applied to the OUTPUT of a sub-layer (attention / retention / FFN) before the residual add.
    The up-projection is zero-initialised, so the network starts exactly as the Phase-1 model."""

    def __init__(self, sub, d, r):
        super().__init__()
        self.sub = sub
        self.down, self.up = nn.Linear(d, r), nn.Linear(r, d)
        nn.init.normal_(self.down.weight, 0.0, 0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, *a, **k):
        y = self.sub(*a, **k)
        return y + self.up(F.gelu(self.down(y)))


def apply_method4(net, cfg: JCfg):
    m, lm = cfg.method, net.lm
    if m in ("full", "kd"):
        return
    if m == "partial":
        for p in lm.wte.parameters():
            p.requires_grad = False
        if lm.wpe is not None:
            for p in lm.wpe.parameters():
                p.requires_grad = False
        for blk in lm.blocks[:cfg.freeze_layers]:
            for p in blk.parameters():
                p.requires_grad = False
        return
    if m == "bitfit":                                    # bias terms + normalisation scales (the models without biases train their norm scales)
        keys = (".n1.", ".n2.", "norm_f", ".qn.", ".kn.", ".on.")
        for n, p in lm.named_parameters():
            p.requires_grad = n.endswith("bias") or any(k in n for k in keys)
        log(f"[bitfit] trainable backbone parameters: {sum(p.numel() for p in lm.parameters() if p.requires_grad):,}")
        return
    if m == "adapter":
        for p in lm.parameters():
            p.requires_grad = False
        d = lm.mc["n_embd"]
        for blk in lm.blocks:
            blk.mix = AdapterWrap(blk.mix, d, cfg.adapter_dim)
            blk.ffn = AdapterWrap(blk.ffn, d, cfg.adapter_dim)
        log(f"[adapter] Houlsby adapters (bottleneck {cfg.adapter_dim}) after every attention and FFN sub-layer")
        return
    if m in ("lora", "lorma"):
        n = N.inject_peft(lm, cfg)
        log(f"[peft] {m}{'-' + cfg.lorma_variant if m == 'lorma' else ''}: wrapped {n} linear layers (r={cfg.peft_r})")
        return
    raise SystemExit(f"unknown method {m!r}: use full | partial | bitfit | adapter | lora | lorma | kd")


# ============================================================================================
# 4. Model
# ============================================================================================
def backbone_hidden(lm, ids, km, bidir):
    B, T = ids.shape
    dev = ids.device
    pos = torch.arange(T, device=dev).unsqueeze(0)
    x = lm.wte(ids)
    if lm.wpe is not None:
        x = x + lm.wpe(pos)
    rope = None
    if lm.rope_cos is not None:
        rope = (lm.rope_cos[pos].unsqueeze(1), lm.rope_sin[pos].unsqueeze(1))
    if bidir:
        amask = km[:, None, None, :]
    else:
        qi, kj = torch.arange(T, device=dev).unsqueeze(1), torch.arange(T, device=dev).unsqueeze(0)
        amask = ((kj <= qi)[None, None] & km[:, None, None, :]) | (kj == qi)[None, None]
    for blk in lm.blocks:
        x = blk(x, rope, amask, km, None)
    return lm.norm_f(x)


class CJNet(nn.Module):
    """Phase-1 backbone + chunk encoder + (BiGRU) + attention pooling over chunks + MLP classifier."""

    def __init__(self, lm, bidir, pool, chunk_dim, head_dropout):
        super().__init__()
        self.lm, self.bidir, self.pool = lm, bidir, pool
        d = lm.mc["n_embd"]
        self.proj = nn.Linear(2 * d, chunk_dim)
        self.norm = nn.LayerNorm(chunk_dim)
        self.pos = nn.Embedding(8, chunk_dim)
        self.lastemb = nn.Embedding(2, chunk_dim)
        for e in (self.pos, self.lastemb):
            nn.init.normal_(e.weight, 0.0, 0.02)
        if pool == "gru":
            self.rnn = nn.GRU(chunk_dim, chunk_dim // 2, batch_first=True, bidirectional=True)
        self.att = nn.Linear(chunk_dim, 1)
        self.drop = nn.Dropout(head_dropout)
        self.cls = nn.Sequential(nn.Linear(2 * chunk_dim, chunk_dim), nn.GELU(), nn.Dropout(head_dropout), nn.Linear(chunk_dim, 2))

    def forward(self, ids, km, owner, slot, rel, last, counts):
        h = backbone_hidden(self.lm, ids, km, self.bidir)                              # (N, L, d)
        kf = km.unsqueeze(-1).to(h.dtype)
        mean = (h * kf).sum(1) / kf.sum(1).clamp(min=1.0)                              # mean of the chunk's token states
        lastv = h[torch.arange(h.size(0), device=h.device), km.sum(1) - 1]             # state of the closing <eos>
        c = self.proj(torch.cat([mean, lastv], dim=-1))
        c = self.drop(F.gelu(self.norm(c + self.pos(rel) + self.lastemb(last))))      # (N, D)
        B, Cm = counts.numel(), int(counts.max())
        pad = c.new_zeros(B, Cm, c.size(-1)).index_put((owner, slot), c)               # (B, Cm, D) chunk sequences
        mask = torch.arange(Cm, device=c.device).unsqueeze(0) < counts.unsqueeze(1)
        if self.pool == "gru":
            pk = nn.utils.rnn.pack_padded_sequence(pad, counts.cpu(), batch_first=True, enforce_sorted=False)
            o, _ = self.rnn(pk)
            pad, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=Cm)
        last_c = pad[torch.arange(B, device=c.device), counts - 1]
        if self.pool == "mean":
            pooled = (pad * mask.unsqueeze(-1).to(pad.dtype)).sum(1) / mask.sum(1, keepdim=True).to(pad.dtype)
        elif self.pool == "last":
            pooled = last_c
        else:                                                                          # attention pooling over chunks (attn | gru)
            sc = self.att(pad).squeeze(-1).float().masked_fill(~mask, -1e9)
            pooled = (sc.softmax(-1).unsqueeze(-1).to(pad.dtype) * pad).sum(1)
        return self.cls(self.drop(torch.cat([pooled, last_c], dim=-1)))                # (B, 2)


def build_net4(cfg: JCfg, lm):
    arch = lm.mc["arch"]
    bidir = (arch != "retnet") if cfg.bidir == "auto" else (cfg.bidir in ("true", "yes", "1", True))
    if arch == "retnet":
        bidir = False
    log(f"[model] arch={arch} bidirectional_attention={bidir} pool={cfg.pool} select={cfg.select} max_chunks={cfg.max_chunks}")
    return CJNet(lm, bidir, cfg.pool, cfg.chunk_dim, cfg.head_dropout)


def load_run4(cfg: JCfg, run_name: str, dev):
    from tokenizers import Tokenizer
    rd = Path(cfg.work_dir) / "cj_runs" / run_name
    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    saved = JCfg()
    apply_overrides(saved, {k: v for k, v in ck["cfg"].items() if k in {f.name for f in fields(JCfg)}})
    saved.work_dir = cfg.work_dir
    if cfg.local_data_dir:
        saved.local_data_dir = cfg.local_data_dir
    net = build_net4(saved, P1.LM(ck["mc"]))
    apply_method4(net, saved)
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    return {"rd": rd, "ck": ck, "saved": saved, "net": net, "tok": Tokenizer.from_file(str(rd / "tokenizer.json"))}


def load_teacher4(cfg: JCfg, dev, tok_path):
    tr = Path(cfg.work_dir) / "cj_runs" / cfg.teacher_run
    if not (tr / "best.pt").exists():
        raise SystemExit(f"teacher run not found: {tr}/best.pt (train it first)")
    if (tr / "tokenizer.json").read_bytes() != Path(tok_path).read_bytes():
        raise SystemExit("teacher and student use different tokenizers (distillation needs identical token ids)")
    r = load_run4(cfg, cfg.teacher_run, dev)
    for p in r["net"].parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in r["net"].parameters())
    log(f"[kd] teacher {cfg.teacher_run}: arch={r['ck']['mc']['arch']} method={r['saved'].method} params={n / 1e6:.1f}M")
    return r["net"], n, r["ck"]["mc"], r["saved"]


def kd_loss_cls(s_logits, t_logits, T):
    """T^2 * KL(softmax(teacher/T) || softmax(student/T)), averaged over the batch."""
    sl = F.log_softmax(s_logits.float() / T, dim=-1)
    tp = F.softmax(t_logits.float() / T, dim=-1)
    return (tp * (torch.log(tp.clamp_min(1e-9)) - sl)).sum(-1).mean() * (T * T)


# ============================================================================================
# 5. Inference, EMA, training
# ============================================================================================
@torch.no_grad()
def doc_probs4(net, docs, cfg, dev, amp, eos, idxs=None, n_chunks=None):
    net.eval()
    idxs = list(range(len(docs))) if idxs is None else list(idxs)
    order = sorted(range(len(idxs)), key=lambda i: docs[idxs[i]].n)            # similar lengths together -> less padding
    out = np.zeros((len(idxs), 2), dtype=np.float32)
    for b0 in range(0, len(order), cfg.eval_docs):
        sel = order[b0:b0 + cfg.eval_docs]
        batch = collate_docs(docs, [idxs[i] for i in sel], cfg, dev, eos, n_chunks or cfg.max_chunks)
        with amp():
            lg = net(*batch[:7])
        out[sel] = lg.float().softmax(-1).cpu().numpy()
    return out


class EMA:
    def __init__(self, net, decay):
        self.net, self.decay = net, decay
        self.shadow = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for n, p in self.net.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextlib.contextmanager
    def swapped(self):
        backup = {}
        with torch.no_grad():
            for n, p in self.net.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.net.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])


def lr_prog(base, prog, warm):
    if prog < warm:
        return base * max(prog, 1e-3) / warm
    return base * max(0.0, (1.0 - prog) / (1.0 - warm))


def run_train4(cfg: JCfg):
    rd = run_dir4(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    jd = asdict(cfg)
    P1.jdump(jd, rd / "config.json")
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype, amp = P1.amp_setup(dev)
    tok, lm, mc = N.load_p1(cfg)
    shutil.copy(Path(cfg.p1_dir) / "tokenizer.json", rd / "tokenizer.json")
    if cfg.method == "kd":
        if not cfg.teacher_run:
            raise SystemExit('method "kd" needs "teacher_run": the name of a finished CJPE run')
        lm, mc = N.make_student(lm, mc, cfg)
    eos = tok.token_to_id(P1.EOS)
    assert cfg.max_len <= mc["ctx"], "max_len exceeds the Phase-1 context length"
    t0 = time.time()
    tr = load_split(cfg, f"{cfg.train_prefix}/train.json", tok, True, cfg.max_train_docs)
    dv = load_split(cfg, f"{cfg.train_prefix}/dev.json", tok, True)
    if cfg.dedupe_train:
        tr, n1, c1 = dedupe_docs(tr)
        dv, n2, c2 = dedupe_docs(dv)
        log(f"[data] removed exact duplicates: {n1} from train ({c1} with a conflicting label), {n2} from dev ({c2} conflicting)")
    ytr, ydv = np.array([d.label for d in tr]), np.array([d.label for d in dv])
    log(f"[data] train {len(tr)} cases (label-1 share {ytr.mean():.2f}) | dev {len(dv)} cases (label-1 share {ydv.mean():.2f}) | "
        f"tokens/case median {int(np.median([d.n for d in tr]))} | tokenised in {time.time() - t0:.0f}s")
    teacher, teacher_params = None, 0
    if cfg.method == "kd":
        teacher, teacher_params, t_mc, _ = load_teacher4(cfg, dev, rd / "tokenizer.json")
        assert cfg.max_len <= t_mc["ctx"]

    net = build_net4(cfg, lm)
    apply_method4(net, cfg)
    net.to(dev)
    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"[model] total params {n_total / 1e6:.1f}M | trainable {n_train / 1e6:.3f}M | method={cfg.method}")

    peft = cfg.method in ("lora", "lorma", "adapter", "bitfit")
    body_lr, body_wd = (cfg.peft_lr, 0.0) if peft else (cfg.lr, cfg.weight_decay)
    nb = mc["n_layer"]

    def lr_scale(name):
        if peft or cfg.llrd >= 1.0 or not name.startswith("lm."):
            return 1.0
        m_ = re.match(r"lm\.blocks\.(\d+)\.", name)
        if m_:
            return cfg.llrd ** (nb - 1 - int(m_.group(1)))
        return cfg.llrd ** nb if name.startswith(("lm.wte", "lm.wpe")) else 1.0

    buckets = {}
    for n_, p_ in net.named_parameters():
        if not p_.requires_grad:
            continue
        if n_.startswith("lm."):
            base, wd = body_lr * lr_scale(n_), (body_wd if p_.dim() >= 2 else 0.0)
        else:
            base, wd = cfg.head_lr, (cfg.weight_decay if p_.dim() >= 2 else 0.0)
        buckets.setdefault((round(base, 12), wd), []).append(p_)
    opt = torch.optim.AdamW([{"params": ps, "lr": b_, "base": b_, "weight_decay": wd} for (b_, wd), ps in buckets.items()],
                            betas=(0.9, 0.98), eps=1e-8)
    scaler = P1.make_scaler(dev == "cuda" and amp_dtype == torch.float16)
    cw = None
    if cfg.class_weight == "balanced":
        cnt = np.bincount(ytr, minlength=2).astype(np.float64)
        cw = torch.tensor(np.clip(len(ytr) / (2.0 * np.maximum(cnt, 1)), 0.25, 4.0), dtype=torch.float32, device=dev)
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay > 0 else None

    rng = np.random.default_rng(cfg.seed)
    dev_sub = np.sort(rng.permutation(len(dv))[:cfg.dev_max_docs]) if cfg.dev_max_docs and len(dv) > cfg.dev_max_docs else np.arange(len(dv))
    n_epoch = len(tr) if not cfg.docs_per_epoch else min(cfg.docs_per_epoch, len(tr))
    steps_per_epoch = max(1, math.ceil(n_epoch / (cfg.batch_docs * cfg.accum)))
    total_steps = cfg.epochs * steps_per_epoch
    budget = cfg.time_budget_min * 60.0
    log(f"[train] {n_epoch} cases/epoch, {steps_per_epoch} steps/epoch x {cfg.epochs} epochs; time budget {cfg.time_budget_min} min; EMA {cfg.ema_decay}")
    best, step, bad, t_start, stop = {"f1": -1.0}, 0, 0, time.time(), False
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logf = open(rd / "log.jsonl", "a")

    def do_eval(tag, loss_val):
        """Evaluate the (EMA-averaged) weights on the dev subset; keep the best by threshold-tuned macro-F1."""
        nonlocal best, bad
        with (ema.swapped() if ema is not None else contextlib.nullcontext()):
            pdv_ = doc_probs4(net, dv, cfg, dev, amp, eos, idxs=dev_sub)
            sc_ = binary_scores(ydv[dev_sub], pdv_.argmax(1))
            thr_, f_tuned, _ = best_threshold(ydv[dev_sub], pdv_[:, 1])
            log(f"[{tag}] loss {loss_val:.4f} | dev(subset {len(dev_sub)}) macro-F1 {sc_['macro_f1']:.4f} at 0.5, "
                f"{f_tuned:.4f} at tuned threshold {thr_} | acc {sc_['acc']:.3f} | {time.time() - t_start:.0f}s")
            logf.write(json.dumps({"tag": tag, "step": step, "loss": loss_val, "dev_f1": sc_["macro_f1"], "dev_f1_tuned": f_tuned}) + "\n")
            logf.flush()
            if f_tuned > best["f1"]:
                best, bad = {"f1": f_tuned, "epoch": tag}, 0
                torch.save({"net": net.state_dict(), "mc": mc, "cfg": jd, "threshold": 0.5}, rd / "best.pt")
            else:
                bad += 1
        net.train()

    for ep in range(cfg.epochs):
        net.train()
        order = rng.permutation(len(tr))[:n_epoch]
        run_loss, n_b, early = 0.0, 0, False
        micro = cfg.batch_docs
        for g0 in range(0, len(order), micro * cfg.accum):
            prog = max(step / total_steps, (time.time() - t_start) / budget if budget > 0 else 0.0)
            if prog >= 1.0:
                stop = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_prog(g["base"], prog, cfg.warmup_frac)
            group = order[g0:g0 + micro * cfg.accum]
            for m0 in range(0, len(group), micro):
                batch = collate_docs(tr, group[m0:m0 + micro], cfg, dev, eos, cfg.max_chunks)
                y = batch[7]
                with amp():
                    logits = net(*batch[:7])
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(*batch[:7])
                loss = F.cross_entropy(logits.float(), y, weight=cw, label_smoothing=cfg.label_smoothing)
                if teacher is not None:
                    loss = cfg.kd_alpha * loss + (1.0 - cfg.kd_alpha) * kd_loss_cls(logits, t_logits, cfg.kd_temp)
                scaler.scale(loss * (len(y) / len(group))).backward()
                run_loss += float(loss.detach())
                n_b += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(step)
            step += 1
            if step % 20 == 0:
                log(f"  ep {ep + 1} step {step}/{total_steps} loss {run_loss / max(1, n_b):.4f} ({time.time() - t_start:.0f}s)")
            if cfg.eval_every_steps and step % cfg.eval_every_steps == 0:
                do_eval(f"step {step}", run_loss / max(1, n_b))
                if bad >= cfg.patience:
                    early = stop = True
                    break
        if not early:
            do_eval(f"epoch {ep + 1}/{cfg.epochs}", run_loss / max(1, n_b))
        if early or bad >= cfg.patience:
            log(f"[train] early stopping: no dev improvement in {bad} evaluations (best: {best['epoch']})")
            break
        if stop:
            log("[train] time budget reached")
            break
    logf.close()
    train_s = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0

    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"])
    if dev == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time()
    pdv = doc_probs4(net, dv, cfg, dev, amp, eos)
    if dev == "cuda":
        torch.cuda.synchronize()
    infer_docs_s = len(dv) / max(1e-9, time.time() - t_inf)
    thr, f_thr, f_base = best_threshold(ydv, pdv[:, 1])
    sc = binary_scores(ydv, (pdv[:, 1] >= thr).astype(int))
    ck["threshold"] = thr
    torch.save(ck, rd / "best.pt")
    np.save(rd / "dev_probs.npy", pdv)
    P1.jdump([d.id for d in dv], rd / "dev_ids.json")
    log(f"[dev] macro-F1 {f_base:.4f} (threshold 0.5) -> {sc['macro_f1']:.4f} (threshold {thr}) | precision {sc['precision']:.4f} "
        f"recall {sc['recall']:.4f} acc {sc['acc']:.4f}")
    log(f"[dev] per-class F1 {[round(x, 4) for x in sc['f1_per_class']]} | confusion tp {sc['tp']} fp {sc['fp']} fn {sc['fn']}")
    # error analysis: confident mistakes + accuracy by case length
    wrong = [(abs(pdv[i, 1] - 0.5), dv[i].id, int(ydv[i]), float(pdv[i, 1]), dv[i].n) for i in range(len(dv)) if (pdv[i, 1] >= thr) != ydv[i]]
    wrong.sort(reverse=True)
    errs = [{"id": i, "gold": g, "p1": round(p, 3), "tokens": n, "type": "false_positive" if g == 0 else "false_negative"} for _, i, g, p, n in wrong[:30]]
    P1.jdump(errs, rd / "errors.json")
    lens = np.array([d.n for d in dv])
    qs = np.quantile(lens, [0.25, 0.5, 0.75])
    pred_dv = (pdv[:, 1] >= thr).astype(int)
    for name, mk in (("shortest 25%", lens <= qs[0]), ("25-50%", (lens > qs[0]) & (lens <= qs[1])),
                     ("50-75%", (lens > qs[1]) & (lens <= qs[2])), ("longest 25%", lens > qs[2])):
        if mk.any():
            log(f"[dev] accuracy for {name:<13} cases ({int(mk.sum()):>4}): {float((pred_dv[mk] == ydv[mk]).mean()):.3f}")
    variant = cfg.lorma_variant if cfg.method == "lorma" else ""
    P1.jdump({"best_epoch": best["epoch"], "scores": sc, "threshold": thr, "macro_f1_default_threshold": f_base,
              "params_total": n_total, "params_trainable": n_train, "train_seconds": train_s, "peak_gpu_mb": peak,
              "infer_docs_per_s": infer_docs_s, "p1_arch": mc["arch"], "method": cfg.method, "teacher_run": cfg.teacher_run,
              "teacher_params": teacher_params}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": mc["arch"], "method": cfg.method + (f"-{variant}" if variant else ""), "pool": cfg.pool,
                  "select": cfg.select, "max_chunks": cfg.max_chunks, "params_total_M": round(n_total / 1e6, 2),
                  "params_trainable_M": round(n_train / 1e6, 3), "dev_macro_f1": round(sc["macro_f1"], 4),
                  "dev_precision": round(sc["precision"], 4), "dev_recall": round(sc["recall"], 4), "dev_acc": round(sc["acc"], 4),
                  "threshold": thr, "train_s": round(train_s), "peak_gpu_mb": round(peak), "infer_docs_s": round(infer_docs_s, 2),
                  "teacher_run": cfg.teacher_run})
    if cfg.predict_after:
        run_predict4(cfg)


def write_submission4(cfg: JCfg, rd: Path, ids, preds):
    out = [{"id": i, "pred": int(p)} for i, p in zip(ids, preds)]
    assert len(out) == len(set(ids)) and all(o["pred"] in (0, 1) for o in out)
    rd.mkdir(parents=True, exist_ok=True)
    pj = rd / "prediction.json"
    P1.jdump(out, pj)
    zp = rd / "submission.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    hist = Path(cfg.work_dir) / "cj_submissions"
    hist.mkdir(parents=True, exist_ok=True)
    shutil.copy(zp, hist / f"{cfg.run}_{time.strftime('%m%d_%H%M%S')}.zip")
    log(f"[predict] wrote {pj} ({len(out)} cases, share of 1 = {np.mean([o['pred'] for o in out]):.2f}) and {zp}")
    return str(zp)


def run_predict4(cfg: JCfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, amp = P1.amp_setup(dev)
    r = load_run4(cfg, cfg.run, dev)
    saved = r["saved"]
    eos = r["tok"].token_to_id(P1.EOS)
    te = load_split(saved, saved.test_file, r["tok"], False)
    p = doc_probs4(r["net"], te, saved, dev, amp, eos)
    np.save(r["rd"] / "test_probs.npy", p)
    P1.jdump([d.id for d in te], r["rd"] / "test_ids.json")
    thr = float(r["ck"].get("threshold", 0.5))
    log(f"[predict] {len(te)} test cases, threshold {thr}")
    return write_submission4(cfg, r["rd"], [d.id for d in te], (p[:, 1] >= thr).astype(int))


# ============================================================================================
# 6. Ensemble: average the saved probabilities of finished runs (no network needed)
# ============================================================================================
def run_ensemble4(cfg: JCfg):
    root = Path(cfg.work_dir) / "cj_runs"
    if cfg.ens_runs:
        names = list(cfg.ens_runs)
    elif cfg.ens_auto:
        names = [f.parent.name for f in sorted(root.glob("*/test_probs.npy"))
                 if (f.parent / "dev_probs.npy").exists() and ((f.parent.name.startswith(cfg.ens_prefix)) if cfg.ens_prefix
                                                              else not f.parent.name.startswith(("selftest", "check")))]
    else:
        names = []
    if len(names) < 2:
        raise SystemExit('need >= 2 finished runs: set "ens_runs": [...] or "ens_auto": true')
    dev_ids = P1.jload(root / names[0] / "dev_ids.json")
    test_ids = P1.jload(root / names[0] / "test_ids.json")
    dv_p, te_p = {}, {}
    for n in names:
        if P1.jload(root / n / "dev_ids.json") != dev_ids or P1.jload(root / n / "test_ids.json") != test_ids:
            log(f"[ensemble] skipping {n}: different case ids / order")
            continue
        dv_p[n], te_p[n] = np.load(root / n / "dev_probs.npy"), np.load(root / n / "test_probs.npy")
    # gold labels of dev, read once from the dev file through the first run's config
    saved = load_saved_cfg(cfg, names[0])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(root / names[0] / "tokenizer.json"))
    dvd = load_split(saved, f"{saved.train_prefix}/dev.json", tok, True)
    if saved.dedupe_train:
        dvd, _, _ = dedupe_docs(dvd)
    ydv = np.array([d.label for d in dvd])
    assert len(ydv) == len(next(iter(dv_p.values()))), "dev size differs from the saved probabilities"
    single = {k: binary_scores(ydv, v.argmax(1))["macro_f1"] for k, v in dv_p.items()}
    for k, v in sorted(single.items(), key=lambda kv: -kv[1]):
        log(f"[ensemble] single model {k:<28} dev macro-F1 {v:.4f}")
    mean = lambda keys, d: sum(d[k] for k in keys) / len(keys)
    if cfg.ens_runs or not cfg.ens_auto:
        chosen = list(dv_p)
    else:
        order = sorted(single, key=lambda k: -single[k])
        chosen, cur = [order[0]], single[order[0]]
        for k in order[1:]:
            if len(chosen) >= cfg.ens_max:
                break
            f = binary_scores(ydv, mean(chosen + [k], dv_p).argmax(1))["macro_f1"]
            log(f"[ensemble] try adding {k:<28} -> {f:.4f} (current {cur:.4f})")
            if f > cur + 0.002:
                chosen.append(k)
                cur = f
    log(f"[ensemble] members: {chosen}")
    pd_, pt_ = mean(chosen, dv_p), mean(chosen, te_p)
    thr, f_thr, f_base = best_threshold(ydv, pd_[:, 1])
    sc = binary_scores(ydv, (pd_[:, 1] >= thr).astype(int))
    log(f"[ensemble] dev macro-F1 {f_base:.4f} -> {sc['macro_f1']:.4f} (threshold {thr}) | precision {sc['precision']:.4f} recall {sc['recall']:.4f}")
    rd = run_dir4(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    P1.jdump({"members": chosen, "single_model_f1": single, "scores": sc, "threshold": thr}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": "ensemble", "method": "ensemble", "dev_macro_f1": round(sc["macro_f1"], 4),
                  "dev_precision": round(sc["precision"], 4), "dev_recall": round(sc["recall"], 4), "dev_acc": round(sc["acc"], 4),
                  "threshold": thr, "teacher_run": "+".join(chosen)})
    return write_submission4(cfg, rd, test_ids, (pt_[:, 1] >= thr).astype(int))


def load_saved_cfg(cfg: JCfg, run_name: str) -> JCfg:
    saved = JCfg()
    d = P1.jload(Path(cfg.work_dir) / "cj_runs" / run_name / "config.json")
    apply_overrides(saved, {k: v for k, v in d.items() if k in {f.name for f in fields(JCfg)}})
    saved.work_dir = cfg.work_dir
    if cfg.local_data_dir:
        saved.local_data_dir = cfg.local_data_dir
    return saved


# ============================================================================================
# 7. Data diagnostics
# ============================================================================================
def diagnose4(cfg: JCfg):
    tok, _, mc = N.load_p1(cfg)
    splits = {}
    for name, rel, lab in (("train", f"{cfg.train_prefix}/train.json", True), ("dev", f"{cfg.train_prefix}/dev.json", True),
                           ("test", cfg.test_file, False)):
        docs = load_split(cfg, rel, tok, lab, cfg.max_train_docs if name == "train" else 0)
        splits[name] = docs
        n = np.array([d.n for d in docs])
        nch = np.array([len(chunk_bounds(d.n, cfg.max_len - 1, cfg.overlap)) for d in docs])
        line = (f"[check] {name:<5}: {len(docs):>6} cases | tokens/case mean {n.mean():.0f} median {np.median(n):.0f} p90 {np.percentile(n, 90):.0f} "
                f"max {n.max()} | chunks/case median {np.median(nch):.0f} p90 {np.percentile(nch, 90):.0f} | "
                f"covered by max_chunks={cfg.max_chunks}: {100 * np.mean(nch <= cfg.max_chunks):.0f}% of cases fully")
        if lab:
            y = np.array([d.label for d in docs])
            line += f" | label 1: {100 * y.mean():.1f}%"
        log(line)
        _, n_dup, n_conf = dedupe_docs(docs)
        log(f"[check] {name}: {n_dup} exact duplicate cases inside this split ({n_conf} of them with a conflicting label)")
    if mc["ctx"] < cfg.max_len:
        log(f"[check] WARNING: max_len {cfg.max_len} > Phase-1 context {mc['ctx']}")
    overlap_report(splits["train"], splits["dev"], splits["test"])


def run_probe(cfg: JCfg):
    """DIAGNOSTIC ONLY (not a submission): which part of a case carries the label? A bag-of-n-grams logistic regression is fitted on
    one region of the Phase-1 token sequence at a time (first / last / middle tokens) and scored on dev. The answer tells you which
    chunks the Phase-1 model must see (select / max_chunks), and is the evidence for 'which sections matter' in the report."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise SystemExit("probe needs scikit-learn:  pip install scikit-learn")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(cfg.p1_dir) / "tokenizer.json"))
    tr = load_split(cfg, f"{cfg.train_prefix}/train.json", tok, True, cfg.max_train_docs)
    dv = load_split(cfg, f"{cfg.train_prefix}/dev.json", tok, True)
    tr, n1, _ = dedupe_docs(tr)
    dv, n2, _ = dedupe_docs(dv)
    rng = np.random.default_rng(0)
    trs = [tr[i] for i in rng.permutation(len(tr))[:cfg.probe_docs]]
    ytr, ydv = np.array([d.label for d in trs]), np.array([d.label for d in dv])
    log(f"[probe] {len(trs)} unique training cases, {len(dv)} unique dev cases (removed {n1}/{n2} duplicates)")
    mid = lambda t: t[max(0, len(t) // 2 - 512): len(t) // 2 + 512]
    regions = {"first 128 tokens": lambda t: t[:128], "first 512 tokens": lambda t: t[:512], "first 2048 tokens": lambda t: t[:2048],
               "last 128 tokens": lambda t: t[-128:], "last 512 tokens": lambda t: t[-512:], "last 2048 tokens": lambda t: t[-2048:],
               "middle 1024 tokens": mid, "first 1024 + last 1024": lambda t: np.concatenate([t[:1024], t[-1024:]]),
               "first 4096 + last 4096": lambda t: np.concatenate([t[:4096], t[-4096:]]) if len(t) > 8192 else t}
    text = lambda t: " ".join(f"t{int(x)}" for x in t)
    sizes = {"first 128 tokens": 128, "first 512 tokens": 512, "first 2048 tokens": 2048, "last 128 tokens": 128, "last 512 tokens": 512,
             "last 2048 tokens": 2048, "middle 1024 tokens": 1024, "first 1024 + last 1024": 2048, "first 4096 + last 4096": 8192}
    results, best = {}, (-1.0, None, None, None)
    for name, fn in regions.items():
        vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True, lowercase=False)
        Xtr = vec.fit_transform([text(fn(d.tok)) for d in trs])
        Xdv = vec.transform([text(fn(d.tok)) for d in dv])
        clf = LogisticRegression(max_iter=300, C=4.0).fit(Xtr, ytr)
        p = clf.predict_proba(Xdv)[:, 1]
        sc = binary_scores(ydv, (p >= 0.5).astype(int))
        thr, f_t, _ = best_threshold(ydv, p)
        results[name] = {"macro_f1": sc["macro_f1"], "acc": sc["acc"], "macro_f1_tuned": f_t}
        log(f"[probe] {name:<26} dev macro-F1 {sc['macro_f1']:.4f} | acc {sc['acc']:.3f} | tuned-threshold {f_t:.4f}")
        if f_t > best[0] + 0.003 or (abs(f_t - best[0]) <= 0.003 and best[1] is not None and sizes[name] < sizes[best[1]]):
            best = (max(f_t, best[0]), name, vec, clf)         # ties (within 0.003): prefer the shorter region
    f_t, name, vec, clf = best
    names, coef = vec.get_feature_names_out(), clf.coef_[0]
    order = np.argsort(coef)
    show = lambda f: tok.decode([int(x[1:]) for x in f.split()]).replace("\n", " ").strip()
    log(f"[probe] BEST region: {name} (macro-F1 {f_t:.4f}). Strongest n-grams there:")
    log("[probe]   -> label 1: " + " | ".join(show(names[i]) for i in order[::-1][:15]))
    log("[probe]   -> label 0: " + " | ".join(show(names[i]) for i in order[:15]))
    P1.jdump(results, Path(cfg.work_dir) / "cj_probe.json")
    return results


# ============================================================================================
# 8. Self-test
# ============================================================================================
def _synthetic(tmp, rng):
    common = "the court appeal petitioner respondent section order judgment high learned counsel submitted held impugned".split()

    def doc(y):
        body = [rng.choice(common) for _ in range(rng.randint(500, 1800))]
        tail = [rng.choice(common) for _ in range(40)] + (["allowed", "set", "aside"] if y else ["dismissed", "no", "merit"]) * 3
        return " ".join(body + tail + [rng.choice(common) for _ in range(10)])

    os.makedirs(tmp, exist_ok=True)
    texts = []
    for split, n in (("train", 40), ("dev", 12), ("test", 10)):
        recs = []
        for i in range(n):
            y = i % 2
            t = doc(y)
            texts.append(t)
            r = {"id": f"{split}_{i}", "text": t}
            if split != "test":
                r["label"] = y
            recs.append(r)
        json.dump(recs, open(os.path.join(tmp, f"{split}.json"), "w"))
    return texts


def selftest() -> bool:
    import tempfile
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    ok = True
    rng = random.Random(0)
    tmp = tempfile.mkdtemp(prefix="cs781_cj_selftest_")
    dd = os.path.join(tmp, "data")
    texts = _synthetic(dd, rng)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([" ".join(t.split()) for t in texts], trainers.BpeTrainer(
        vocab_size=500, special_tokens=[P1.EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    V = P1.pad64(tok.get_vocab_size())

    log("=== 1. chunking, selection, metrics ===")
    b = chunk_bounds(1000, 63, 0)
    g1 = b[-1][1] == 1000 and b[0][0] == 0 and all(e - s <= 63 for s, e in b) and all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
    g2 = chunk_bounds(0, 63, 0) == [(0, 0)] and chunk_bounds(10, 63, 0) == [(0, 10)]
    g3 = select_chunks(10, 3, "tail") == [7, 8, 9] and select_chunks(10, 3, "head") == [0, 1, 2] and \
        select_chunks(10, 4, "head_tail") == [0, 7, 8, 9] and select_chunks(10, 4, "uniform")[0] == 0 and select_chunks(10, 4, "uniform")[-1] == 9
    g3b = select_chunks(10, 4, "head_tail", 2) == [0, 1, 8, 9] and select_chunks(10, 2, "head_tail", 1) == [0, 9]
    ok &= g3b
    log(f"  explicit head size for head_tail {'PASS' if g3b else 'FAIL'}")
    y = np.array([0, 0, 1, 1, 1, 0])
    yh = np.array([0, 1, 1, 1, 0, 0])
    sc = binary_scores(y, yh)
    f_man = np.mean([2 * 2 / (2 * 2 + 1 + 1), 2 * 2 / (2 * 2 + 1 + 1)])
    g4 = abs(sc["macro_f1"] - f_man) < 1e-9 and abs(sc["precision"] - 2 / 3) < 1e-9 and abs(sc["recall"] - 2 / 3) < 1e-9
    t_, f_, f0 = best_threshold(np.array([0, 0, 1, 1]), np.array([0.2, 0.45, 0.4, 0.9]))
    g5 = f_ >= f0
    ok &= all([g1, g2, g3, g4, g5])
    log(f"  chunks from the end {'PASS' if g1 else 'FAIL'} | edge cases {'PASS' if g2 else 'FAIL'} | selection strategies {'PASS' if g3 else 'FAIL'} | "
        f"metrics = evaluator definitions {'PASS' if g4 else 'FAIL'} | threshold tuning {'PASS' if g5 else 'FAIL'}")

    cfg0 = apply_overrides(JCfg(), {"local_data_dir": dd, "max_len": 64, "max_chunks": 3})
    tr = load_split(cfg0, "Phase_4_CJPE/Train/train.json", tok, True)
    eos = tok.token_to_id(P1.EOS)
    batch = collate_docs(tr, [0, 1], cfg0, "cpu", eos, 3)
    g6 = len(tr) == 40 and batch[0].shape[0] == int(batch[6].sum()) and batch[0].shape[1] <= 64 and int(batch[1].sum(1).max()) <= 64
    ok &= g6
    log(f"  collate (chunks of <= max_len incl. <eos>) {'PASS' if g6 else 'FAIL'}")

    d1, d2 = JDoc(), JDoc()
    d1.tok, d1.label = np.arange(10, dtype=np.uint16), 1
    d2.tok, d2.label = np.arange(10, dtype=np.uint16), 0
    kept, nrem, nconf = dedupe_docs([d1, d2, tr[0]])
    rows = overlap_report(tr[:5], tr[:3] + tr[10:12], tr[3:6])
    g7 = len(kept) == 2 and nrem == 1 and nconf == 1 and rows[0][2] == 3 and rows[1][2] == 2
    ok &= g7
    log(f"  exact-duplicate removal + train/dev/test overlap report {'PASS' if g7 else 'FAIL'}")

    log("=== 2. adapters / BitFit / freezing: identity at init, only the intended parameters train ===")
    for arch in P1.ARCH_PRESETS:
        for meth, var in (("adapter", "plus"), ("lora", "plus"), ("lorma", "plus"), ("lorma", "pi"), ("bitfit", "plus"), ("partial", "plus")):
            tag = f"{arch}/{meth}" + (f"-{var}" if meth == "lorma" else "")
            try:
                torch.manual_seed(0)
                c = apply_overrides(JCfg(), {"method": meth, "lorma_variant": var, "peft_r": 4, "peft_alpha": 8.0, "peft_dropout": 0.0,
                                             "adapter_dim": 8, "freeze_layers": 1, "chunk_dim": 32, "max_len": 64, "max_chunks": 3})
                net = build_net4(c, P1.LM(N._tiny_mc(arch, V))).eval()
                inp = collate_docs(tr, [0, 1], c, "cpu", eos, 3)[:7]
                with torch.no_grad():
                    ref = net(*inp)
                apply_method4(net, c)
                net.eval()
                with torch.no_grad():
                    out = net(*inp)
                diff = (ref - out).abs().max().item()
                n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
                n_all = sum(p.numel() for p in net.parameters())
                net.train()
                net(*inp).float().pow(2).mean().backward()
                grads = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.parameters() if p.requires_grad)
                good = diff < 1e-4 and n_tr < n_all and grads
                ok &= good
                log(f"  {tag:<18} |adapted - base| = {diff:.1e}  trainable {n_tr:,}/{n_all:,}  {'PASS' if good else 'FAIL'}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  {tag:<18} FAIL: {type(e).__name__}: {e}")
                ok = False

    log("=== 3. end-to-end: every architecture / method / pooling, then ensembles ===")
    p1dirs = {}
    for arch in P1.ARCH_PRESETS:
        p1 = os.path.join(tmp, f"p1_{arch}")
        os.makedirs(p1, exist_ok=True)
        torch.save({"model": P1.LM(N._tiny_mc(arch, V)).state_dict(), "mc": N._tiny_mc(arch, V), "state": {"step": 0}},
                   os.path.join(p1, "ckpt_best.pt"))
        tok.save(os.path.join(p1, "tokenizer.json"))
        p1dirs[arch] = p1
    base = {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "max_len": 64, "max_chunks": 3, "epochs": 2, "batch_docs": 2,
            "accum": 1, "eval_docs": 2, "lr": 1e-3, "peft_lr": 3e-3, "head_lr": 3e-3, "peft_r": 4, "peft_alpha": 8.0, "chunk_dim": 32,
            "adapter_dim": 8, "ema_decay": 0.9, "time_budget_min": 5, "patience": 5, "dev_max_docs": 8}
    cases = [(f"full_{a}", a, {"method": "full"}) for a in P1.ARCH_PRESETS] + [
        ("lorma_plus", "gpt2", {"method": "lorma", "lorma_variant": "plus"}),
        ("lorma_pi", "retnet", {"method": "lorma", "lorma_variant": "pi"}),
        ("lora", "llama", {"method": "lora"}), ("adapter", "moe", {"method": "adapter"}),
        ("bitfit", "gpt2", {"method": "bitfit"}), ("partial", "llama", {"method": "partial", "freeze_layers": 1}),
        ("pool_mean", "llama", {"method": "full", "pool": "mean"}), ("pool_gru", "llama", {"method": "full", "pool": "gru"}),
        ("pool_last", "llama", {"method": "full", "pool": "last"}), ("head_tail", "llama", {"method": "full", "select": "head_tail"}),
        ("kd", "gpt2", {"method": "kd", "teacher_run": "selftest_full_llama", "student_layers": 1})]
    for tag, arch, extra in cases:
        try:
            cfg = apply_overrides(JCfg(), {**base, **extra, "run": f"selftest_{tag}", "p1_dir": p1dirs[arch]})
            run_train4(cfg)
            with zipfile.ZipFile(run_dir4(cfg) / "submission.zip") as z:
                assert z.namelist() == ["prediction.json"]
                pr = json.loads(z.read("prediction.json"))
            assert isinstance(pr, list) and len(pr) == 10 and [x["id"] for x in pr] == [f"test_{i}" for i in range(10)]
            assert all(set(x) == {"id", "pred"} and type(x["pred"]) is int and x["pred"] in (0, 1) for x in pr)
            rep = P1.jload(run_dir4(cfg) / "dev_report.json")
            if extra["method"] in ("lora", "lorma", "adapter", "bitfit", "partial"):
                assert rep["params_trainable"] < rep["params_total"]
            if extra["method"] == "kd":
                assert rep["params_total"] < rep["teacher_params"]
            log(f"  e2e {tag:<12} PASS  (dev macro-F1 {rep['scores']['macro_f1']:.3f}, trainable {rep['params_trainable']:,} / {rep['params_total']:,})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  e2e {tag:<12} FAIL: {type(e).__name__}: {e}")
            ok = False
    try:
        pcfg = apply_overrides(JCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "p1_dir": p1dirs["llama"], "probe_docs": 40})
        res = run_probe(pcfg)
        assert res["last 512 tokens"]["macro_f1_tuned"] > res["first 512 tokens"]["macro_f1_tuned"] + 0.1, res      # synthetic labels sit at the END of every case
        best_region = max(res, key=lambda k: res[k]["macro_f1_tuned"])
        log(f"  probe stage PASS (best region on synthetic data: {best_region})")
    except ImportError:
        log("  probe stage skipped (scikit-learn not installed)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"  probe stage FAIL: {type(e).__name__}: {e}")
        ok = False
    try:
        for tag, ov in (("ens", {"ens_runs": ["selftest_full_llama", "selftest_full_gpt2"]}),
                        ("ens_auto", {"ens_auto": True, "ens_prefix": "selftest_"})):
            cfg = apply_overrides(JCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "run": f"selftest_{tag}", **ov})
            zp = run_ensemble4(cfg)
            with zipfile.ZipFile(zp) as z:
                pr = json.loads(z.read("prediction.json"))
            assert len(pr) == 10 and all(x["pred"] in (0, 1) for x in pr)
        log("  ensembles (explicit + automatic) PASS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"  ensembles FAIL: {type(e).__name__}: {e}")
        ok = False
    shutil.rmtree(tmp, ignore_errors=True)
    log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


# ============================================================================================
# 9. CLI
# ============================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["selftest", "check", "probe", "train", "predict", "ensemble"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--cfg", default="", help="JSON string or JSON file with config overrides")
    args = ap.parse_args(argv)
    if args.stage == "selftest":
        return 0 if selftest() else 1
    cfg = JCfg()
    ov = {}
    if args.cfg:
        ov = json.loads(args.cfg) if args.cfg.strip().startswith("{") else P1.jload(args.cfg)
    if args.run:
        ov["run"] = args.run
    apply_overrides(cfg, ov)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    {"check": diagnose4, "probe": run_probe, "train": run_train4, "predict": run_predict4, "ensemble": run_ensemble4}[args.stage](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
