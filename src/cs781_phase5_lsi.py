#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS781 Phase 5 - Legal Statute Identification (multi-label) on top of YOUR Phase-1 language models.

Needs cs781_phase1.py, cs781_phase2_ner.py and cs781_phase4_cjpe.py in the same folder (model classes, adapters, distillation helpers).

Model (multi-label, long documents)
  case text -> Phase-1 BPE tokens -> chunks of max_len tokens -> Phase-1 backbone (bidirectional attention; causal for retnet)
  -> chunk vector = [mean of token states ; closing <eos> state] + chunk-position embedding
  -> pooling over chunks ("pool"):
       attn       LABEL-WISE attention: every statute has its own query over the chunks (statutes are mentioned in different places)
       chunk_max  chunk-level statute scores, document score = max over chunks
       max | mean pooled vector -> linear
  -> one logit per statute (bias initialised from the label priors).  Training reads `train_chunks` random chunks per case, inference up to
  `eval_chunks` chunks spread over the whole case.
  Losses: bce | bce_pw (positive-class weights for rare statutes) | focal | asl.
  Decision: global vs per-label thresholds (+ "at least one statute per case"), chosen by cross-fitting on dev.
  Output: [{"id": ..., "pred": [unique statute ids]}, ...]

Adaptation methods (cfg "method")
  full | partial | bitfit | adapter (Houlsby) | lora | lorma (LoRMA+ / LoRMAπ) | prompt (soft prompt tuning) | kd (pruned student <- teacher)

Stages:  selftest | check | probe | train | predict | ensemble
  python cs781_phase5_lsi.py check --run ls_check --cfg '{"p1_dir": ".../explore_llama"}'
  python cs781_phase5_lsi.py train --run ls_full_llama --cfg cfg_ls.json
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
import pickle
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
import cs781_phase4_cjpe as J

log = P1.log


# ============================================================================================
# 0. Config
# ============================================================================================
@dataclass
class LCfg:
    run: str = "ls_full_v1"
    work_dir: str = P1.DEFAULT_WORK
    p1_dir: str = ""
    p1_ckpt: str = "ckpt_best.pt"
    hf_repo: str = "Exploration-Lab/CS-781-Capstone"
    local_data_dir: str = ""          # folder with train.json, dev.json, test.json
    train_prefix: str = "Phase_5_LSI/Train"
    test_file: str = "Phase_5_LSI/Test/test.json"
    max_train_docs: int = 0
    dedupe_train: bool = True         # drop exact duplicate cases from train and dev
    cache_tokens: bool = True         # tokenised documents are cached in <work_dir>/cache (every run reuses them)
    # ---- long-document representation
    max_len: int = 512
    train_chunks: int = 4             # random chunks per case per training step
    eval_chunks: int = 16             # chunks per case at inference (spread over the whole case)
    train_select: str = "random"      # random | uniform | head | tail
    eval_select: str = "uniform"
    overlap: int = 0
    pool: str = "attn"                # attn (label-wise attention) | chunk_max | max | mean
    chunk_dim: int = 384
    head_dropout: float = 0.1
    bidir: str = "auto"
    # ---- method
    method: str = "full"              # full | partial | bitfit | adapter | lora | lorma | prompt | kd
    freeze_layers: int = 0
    adapter_dim: int = 64
    prompt_tokens: int = 16
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
    # ---- loss
    loss: str = "bce_pw"              # bce | bce_pw | focal | asl
    pos_weight_max: float = 20.0
    focal_gamma: float = 2.0
    # ---- optimisation
    epochs: int = 6
    docs_per_epoch: int = 0
    batch_docs: int = 2
    accum: int = 2
    eval_docs: int = 4
    lr: float = 2e-4
    llrd: float = 0.85
    head_lr: float = 2e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    ema_decay: float = 0.995
    time_budget_min: float = 60.0
    dev_max_docs: int = 800
    eval_every_steps: int = 3000
    patience: int = 4
    seed: int = 1337
    predict_after: bool = True
    # ---- ensemble / probe
    ens_runs: list = field(default_factory=list)
    ens_auto: bool = False
    ens_prefix: str = ""
    ens_max: int = 6
    probe_docs: int = 6000
    probe_labels: int = 10


def apply_overrides(cfg: LCfg, ov: dict) -> LCfg:
    names = {f.name for f in fields(LCfg)}
    for k, v in ov.items():
        if k not in names:
            raise SystemExit(f"Unknown config key {k!r}. Valid: {sorted(names)}")
        setattr(cfg, k, v)
    return cfg


def run_dir5(cfg: LCfg) -> Path:
    return Path(cfg.work_dir) / "ls_runs" / cfg.run


LS_FIELDS = ["time", "run", "p1_arch", "method", "pool", "loss", "params_total_M", "params_trainable_M", "dev_macro_f1", "dev_micro_f1",
             "dev_subset_acc", "decision", "n_labels", "train_s", "peak_gpu_mb", "infer_docs_s", "teacher_run"]


def log_csv(cfg: LCfg, row: dict):
    p = Path(cfg.work_dir) / "ls_experiments.csv"
    if p.exists() and open(p).readline().strip().split(",") != LS_FIELDS:
        p.rename(p.with_suffix(".old.csv"))
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LS_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row})


# ============================================================================================
# 1. Data (tolerant parsing, cached tokenisation)
# ============================================================================================
class LDoc:
    pass


def _first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_split5(cfg: LCfg, rel: str, tok, labelled: bool, max_docs: int = 0, tok_path: str = ""):
    """-> list of LDoc(id, tok uint16, n, labels [int])."""
    path = P1.resolve_file(cfg, rel)
    cache = None
    if cfg.cache_tokens and tok_path:
        key = hashlib.md5(open(tok_path, "rb").read() + f"{os.path.basename(path)}|{os.path.getsize(path)}|{max_docs}".encode()).hexdigest()[:12]
        cache = Path(cfg.work_dir) / "cache" / f"lsi_{key}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                recs = pickle.load(f)
            log(f"[data] {rel}: {len(recs)} cases from the token cache")
            return [_mk(i, t, l) for i, t, l in recs]
    raw = P1.jload(path)
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
            y = _first(r, ("labels", "label", "gold", "statutes", "statute_ids")) or []
            if isinstance(y, (int, float, str)):
                y = [y]
            labels.append(sorted({int(float(x)) for x in y}))
        else:
            labels.append([])
    recs = []
    for b0 in range(0, len(texts), 128):
        for k, e in enumerate(tok.encode_batch(texts[b0:b0 + 128])):
            recs.append((ids[b0 + k], np.asarray(e.ids, dtype=np.uint16), labels[b0 + k]))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(recs, f, protocol=4)
    return [_mk(i, t, l) for i, t, l in recs]


def _mk(i, t, l):
    d = LDoc()
    d.id, d.tok, d.n, d.labels = i, t, len(t), list(l)
    return d


def dedupe_docs5(docs):
    seen, keep, conflict = {}, [], 0
    for d in docs:
        h = hashlib.md5(d.tok.tobytes()).hexdigest()
        if h in seen:
            conflict += int(seen[h] != d.labels)
            continue
        seen[h] = d.labels
        keep.append(d)
    return keep, len(docs) - len(keep), conflict


def label_matrix(docs, L):
    Y = np.zeros((len(docs), L), dtype=np.uint8)
    for i, d in enumerate(docs):
        for x in d.labels:
            if 0 <= x < L:
                Y[i, x] = 1
    return Y


def pick_chunks(m, k, mode, rng):
    if m <= k:
        return list(range(m))
    if mode == "random":
        return sorted(rng.sample(range(m), k))
    if mode == "head":
        return list(range(k))
    if mode == "tail":
        return list(range(m - k, m))
    return sorted({int(x) for x in np.linspace(0, m - 1, k).round()})              # uniform


def collate5(docs, idxs, cfg: LCfg, dev, eos, L, n_chunks, mode, rng=None):
    chunks, owner, slot, rel, counts = [], [], [], [], []
    for b, di in enumerate(idxs):
        d = docs[di]
        bounds = J.chunk_bounds(d.n, cfg.max_len - 1, cfg.overlap)
        sel = pick_chunks(len(bounds), n_chunks, mode, rng or random)
        for k_, j in enumerate(sel):
            s, e = bounds[j]
            chunks.append(d.tok[s:e])
            owner.append(b)
            slot.append(k_)
            rel.append(min(7, int(8 * j / len(bounds))))
        counts.append(len(sel))
    T = max(len(c) for c in chunks) + 1
    ids = np.zeros((len(chunks), T), dtype=np.int64)
    km = np.zeros((len(chunks), T), dtype=bool)
    for i, c in enumerate(chunks):
        n = len(c)
        ids[i, :n] = c
        ids[i, n] = eos
        km[i, :n + 1] = True
    Y = label_matrix([docs[di] for di in idxs], L).astype(np.float32)
    t = lambda x: torch.from_numpy(x).to(dev)
    return t(ids), t(km), t(np.array(owner)), t(np.array(slot)), t(np.array(rel)), t(np.array(counts)), t(Y)


# ============================================================================================
# 2. Metrics + decision strategies
# ============================================================================================
def per_label_f1(Y, Yhat):
    Y, Yhat = Y.astype(bool), Yhat.astype(bool)
    tp = (Y & Yhat).sum(0).astype(np.float64)
    fp = (~Y & Yhat).sum(0).astype(np.float64)
    fn = (Y & ~Yhat).sum(0).astype(np.float64)
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0), tp, fp, fn


def ml_scores(Y, Yhat):
    """Same definitions as the official evaluator (sklearn, zero_division=0): macro over ALL label columns, micro, subset accuracy."""
    f1, tp, fp, fn = per_label_f1(Y, Yhat)
    den = 2 * tp.sum() + fp.sum() + fn.sum()
    P = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    R = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return {"macro_f1": float(f1.mean()), "micro_f1": float(2 * tp.sum() / den) if den > 0 else 0.0,
            "subset_acc": float((Y.astype(bool) == Yhat.astype(bool)).all(1).mean()) if len(Y) else 0.0,
            "macro_p": float(P.mean()), "macro_r": float(R.mean()), "f1": f1, "tp": tp, "fp": fp, "fn": fn}


GRID = np.arange(0.05, 0.9501, 0.025)


def f1_grid(P, Y):
    return np.stack([per_label_f1(Y, P >= t)[0] for t in GRID])            # (len(GRID), L)


def best_global(P, Y):
    F1g = f1_grid(P, Y)
    gi = int(F1g.mean(1).argmax())
    return float(GRID[gi]), float(F1g[gi].mean())


def tune_decision(P, Y, min_support=3):
    """Global threshold maximising macro-F1, then a per-label threshold (shrunk 50% towards the global one) for labels with enough
    positives and a clear gain."""
    F1g = f1_grid(P, Y)
    gi = int(F1g.mean(1).argmax())
    t_g = float(GRID[gi])
    per = np.full(Y.shape[1], t_g)
    for l in range(Y.shape[1]):
        if Y[:, l].sum() < min_support:
            continue
        bi = int(F1g[:, l].argmax())
        if F1g[bi, l] > F1g[gi, l] + 0.02:
            per[l] = 0.5 * GRID[bi] + 0.5 * t_g
    return t_g, per


def apply_decision(P, thr, ensure_one):
    pred = P >= np.asarray(thr)[None, :]
    if ensure_one and len(P):
        empty = ~pred.any(1)
        top = (P - np.asarray(thr)[None, :]).argmax(1)
        pred[np.where(empty)[0], top[empty]] = True
    return pred


def choose_decision(P, Y, seed=0):
    """Pick global / per-label thresholds and 'at least one label' by 2-fold cross-fitting on dev (tune on one half, score on the other),
    then tune the chosen strategy on all of dev."""
    L = Y.shape[1]
    names = ["global", "global+one", "per_label", "per_label+one"]
    cross = {n: [] for n in names}
    perm = np.random.default_rng(seed).permutation(len(P))
    halves = [perm[::2], perm[1::2]]
    if min(len(h) for h in halves) >= 2:
        for a, b in ((0, 1), (1, 0)):
            t_g, per = tune_decision(P[halves[a]], Y[halves[a]])
            for n in names:
                thr = np.full(L, t_g) if n.startswith("global") else per
                cross[n].append(ml_scores(Y[halves[b]], apply_decision(P[halves[b]], thr, n.endswith("+one")))["macro_f1"])
        cross = {n: float(np.mean(v)) for n, v in cross.items()}
    else:
        cross = {n: 0.0 for n in names}
    best = max(names, key=lambda n: cross[n]) if any(cross.values()) else "global+one"
    t_g, per = tune_decision(P, Y)
    thr = np.full(L, t_g) if best.startswith("global") else per
    sc = ml_scores(Y, apply_decision(P, thr, best.endswith("+one")))
    return {"name": best, "thr": [float(x) for x in thr], "ensure_one": best.endswith("+one"), "t_global": t_g,
            "cross_fit": cross, "scores": sc}


# ============================================================================================
# 3. Methods: reuse Phase 4 (adapters / BitFit / freezing / LoRA / LoRMA) + soft prompt tuning
# ============================================================================================
def apply_method5(net, cfg: LCfg):
    if cfg.method == "prompt":
        for p in net.lm.parameters():
            p.requires_grad = False
        log(f"[prompt] {cfg.prompt_tokens} learned soft-prompt vectors prepended to every chunk; backbone frozen")
        return
    J.apply_method4(net, cfg)


def backbone5(lm, ids, km, bidir, prompt):
    B, T = ids.shape
    dev = ids.device
    x = lm.wte(ids)
    P = 0
    if prompt is not None:
        P = prompt.size(0)
        x = torch.cat([prompt.unsqueeze(0).expand(B, -1, -1).to(x.dtype), x], dim=1)
        km = torch.cat([torch.ones(B, P, dtype=torch.bool, device=dev), km], dim=1)
    T2 = T + P
    pos = torch.arange(T2, device=dev).unsqueeze(0)
    if lm.wpe is not None:
        x = x + lm.wpe(pos)
    rope = None
    if lm.rope_cos is not None:
        rope = (lm.rope_cos[pos].unsqueeze(1), lm.rope_sin[pos].unsqueeze(1))
    if bidir:
        amask = km[:, None, None, :]
    else:
        qi, kj = torch.arange(T2, device=dev).unsqueeze(1), torch.arange(T2, device=dev).unsqueeze(0)
        amask = ((kj <= qi)[None, None] & km[:, None, None, :]) | (kj == qi)[None, None]
    for blk in lm.blocks:
        x = blk(x, rope, amask, km, None)
    return lm.norm_f(x)[:, P:]


class LSINet(nn.Module):
    def __init__(self, lm, L, bidir, pool, chunk_dim, head_dropout, prompt_tokens, prior):
        super().__init__()
        self.lm, self.bidir, self.pool, self.P = lm, bidir, pool, prompt_tokens
        d = lm.mc["n_embd"]
        if prompt_tokens:
            self.prompt = nn.Parameter(torch.randn(prompt_tokens, d) * 0.02)
        self.proj = nn.Linear(2 * d, chunk_dim)
        self.norm = nn.LayerNorm(chunk_dim)
        self.pos = nn.Embedding(8, chunk_dim)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        self.drop = nn.Dropout(head_dropout)
        bias0 = torch.tensor(np.log(prior / (1.0 - prior)), dtype=torch.float32)       # start from the label priors (huge help for rare statutes)
        if pool == "attn":
            self.q = nn.Parameter(torch.randn(L, chunk_dim) * 0.02)
            self.w = nn.Parameter(torch.randn(L, chunk_dim) * 0.02)
            self.b = nn.Parameter(bias0.clone())
        else:
            self.out = nn.Linear(chunk_dim, L)
            nn.init.normal_(self.out.weight, 0.0, 0.02)
            with torch.no_grad():
                self.out.bias.copy_(bias0)

    def forward(self, ids, km, owner, slot, rel, counts):
        h = backbone5(self.lm, ids, km, self.bidir, self.prompt if self.P else None)          # (N, T, d)
        kf = km.unsqueeze(-1).to(h.dtype)
        mean = (h * kf).sum(1) / kf.sum(1).clamp(min=1.0)
        lastv = h[torch.arange(h.size(0), device=h.device), km.sum(1) - 1]
        c = self.drop(F.gelu(self.norm(self.proj(torch.cat([mean, lastv], dim=-1)) + self.pos(rel))))        # (N, D)
        B, Cm = counts.numel(), int(counts.max())
        pad = c.new_zeros(B, Cm, c.size(-1)).index_put((owner, slot), c)
        mask = torch.arange(Cm, device=c.device).unsqueeze(0) < counts.unsqueeze(1)                  # (B, Cm)
        if self.pool == "attn":                                                                      # label-wise attention over chunks
            sc = torch.einsum("bnd,ld->bln", pad, self.q.to(pad.dtype)).float().masked_fill(~mask.unsqueeze(1), -1e9)
            ctx = torch.einsum("bln,bnd->bld", sc.softmax(-1).to(pad.dtype), pad)
            return (ctx * self.w.to(ctx.dtype)).sum(-1) + self.b.to(ctx.dtype)
        if self.pool == "mean":
            z = (pad * mask.unsqueeze(-1).to(pad.dtype)).sum(1) / mask.sum(1, keepdim=True).to(pad.dtype)
            return self.out(z)
        if self.pool == "max":
            return self.out(pad.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values)
        lg = self.out(pad)                                                                           # chunk_max: chunk-level scores, then max
        return lg.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values


def build_net5(cfg: LCfg, lm, L, prior):
    arch = lm.mc["arch"]
    bidir = (arch != "retnet") if cfg.bidir == "auto" else (cfg.bidir in ("true", "yes", "1", True))
    if arch == "retnet":
        bidir = False
    P = cfg.prompt_tokens if cfg.method == "prompt" else 0
    log(f"[model] arch={arch} bidirectional_attention={bidir} pool={cfg.pool} labels={L} loss={cfg.loss} prompt_tokens={P}")
    return LSINet(lm, L, bidir, cfg.pool, cfg.chunk_dim, cfg.head_dropout, P, prior)


def ml_loss(logits, Y, cfg: LCfg, pos_weight):
    x = logits.float()
    if cfg.loss == "bce":
        return F.binary_cross_entropy_with_logits(x, Y)
    if cfg.loss == "bce_pw":
        return F.binary_cross_entropy_with_logits(x, Y, pos_weight=pos_weight)
    if cfg.loss == "focal":
        bce = F.binary_cross_entropy_with_logits(x, Y, reduction="none")
        p = torch.sigmoid(x)
        pt = Y * p + (1 - Y) * (1 - p)
        return ((1 - pt) ** cfg.focal_gamma * bce).mean()
    if cfg.loss == "asl":                                                   # asymmetric loss: down-weights easy negatives
        p = torch.sigmoid(x)
        pm = (p - 0.05).clamp(min=0)
        return -(Y * torch.log(p.clamp(min=1e-8)) + (1 - Y) * pm ** 4 * torch.log((1 - pm).clamp(min=1e-8))).mean()
    raise SystemExit(f"unknown loss {cfg.loss!r}: use bce | bce_pw | focal | asl")


def kd_loss_ml(s, t, T):
    """Soft-target BCE between temperature-scaled student and teacher probabilities (multi-label distillation), times T^2."""
    return F.binary_cross_entropy_with_logits(s.float() / T, torch.sigmoid(t.float() / T)) * (T * T)


def load_run5(cfg: LCfg, run_name: str, dev):
    from tokenizers import Tokenizer
    rd = Path(cfg.work_dir) / "ls_runs" / run_name
    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    saved = LCfg()
    apply_overrides(saved, {k: v for k, v in ck["cfg"].items() if k in {f.name for f in fields(LCfg)}})
    saved.work_dir = cfg.work_dir
    if cfg.local_data_dir:
        saved.local_data_dir = cfg.local_data_dir
    L = ck["n_labels"]
    net = build_net5(saved, P1.LM(ck["mc"]), L, np.full(L, 0.01))
    apply_method5(net, saved)
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    return {"rd": rd, "ck": ck, "saved": saved, "net": net, "L": L, "tok": Tokenizer.from_file(str(rd / "tokenizer.json"))}


def load_teacher5(cfg: LCfg, dev, tok_path, L):
    tr = Path(cfg.work_dir) / "ls_runs" / cfg.teacher_run
    if not (tr / "best.pt").exists():
        raise SystemExit(f"teacher run not found: {tr}/best.pt (train it first)")
    if (tr / "tokenizer.json").read_bytes() != Path(tok_path).read_bytes():
        raise SystemExit("teacher and student use different tokenizers (distillation needs identical token ids)")
    r = load_run5(cfg, cfg.teacher_run, dev)
    if r["L"] != L:
        raise SystemExit("teacher and student use different label spaces")
    for p in r["net"].parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in r["net"].parameters())
    log(f"[kd] teacher {cfg.teacher_run}: arch={r['ck']['mc']['arch']} method={r['saved'].method} params={n / 1e6:.1f}M")
    return r["net"], n, r["ck"]["mc"]


# ============================================================================================
# 4. Inference + training
# ============================================================================================
@torch.no_grad()
def doc_probs5(net, docs, cfg: LCfg, dev, amp, eos, L, idxs=None):
    net.eval()
    idxs = list(range(len(docs))) if idxs is None else list(idxs)
    order = sorted(range(len(idxs)), key=lambda i: docs[idxs[i]].n)
    out = np.zeros((len(idxs), L), dtype=np.float32)
    for b0 in range(0, len(order), cfg.eval_docs):
        sel = order[b0:b0 + cfg.eval_docs]
        batch = collate5(docs, [idxs[i] for i in sel], cfg, dev, eos, L, cfg.eval_chunks, cfg.eval_select)
        with amp():
            lg = net(*batch[:6])
        out[sel] = torch.sigmoid(lg.float()).cpu().numpy()
    return out


def run_train5(cfg: LCfg):
    rd = run_dir5(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    jd = asdict(cfg)
    P1.jdump(jd, rd / "config.json")
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype, amp = P1.amp_setup(dev)
    tok, lm, mc = N.load_p1(cfg)
    tok_path = str(Path(cfg.p1_dir) / "tokenizer.json")
    shutil.copy(tok_path, rd / "tokenizer.json")
    if cfg.method == "kd":
        if not cfg.teacher_run:
            raise SystemExit('method "kd" needs "teacher_run": the name of a finished LSI run')
        lm, mc = N.make_student(lm, mc, cfg)
    eos = tok.token_to_id(P1.EOS)
    P_tok = cfg.prompt_tokens if cfg.method == "prompt" else 0
    assert cfg.max_len + P_tok <= mc["ctx"], "max_len (+ prompt tokens) exceeds the Phase-1 context length"
    t0 = time.time()
    tr = load_split5(cfg, f"{cfg.train_prefix}/train.json", tok, True, cfg.max_train_docs, tok_path)
    dv = load_split5(cfg, f"{cfg.train_prefix}/dev.json", tok, True, 0, tok_path)
    if cfg.dedupe_train:
        tr, n1, c1 = dedupe_docs5(tr)
        dv, n2, c2 = dedupe_docs5(dv)
        log(f"[data] removed exact duplicates: {n1} from train ({c1} with conflicting labels), {n2} from dev ({c2} conflicting)")
    L = 1 + max(max((max(d.labels) for d in docs if d.labels), default=0) for docs in (tr, dv))
    Ytr, Ydv = label_matrix(tr, L), label_matrix(dv, L)
    freq = Ytr.sum(0)
    log(f"[data] train {len(tr)} cases | dev {len(dv)} cases | {L} statute ids ({int((freq > 0).sum())} occur in train, {int((freq < 5).sum())} have < 5 train cases) | "
        f"labels/case mean {Ytr.sum(1).mean():.2f} | loaded in {time.time() - t0:.0f}s")
    prior = np.clip((freq + 1.0) / (len(tr) + 2.0), 1e-4, 0.5)
    teacher, teacher_params = None, 0
    if cfg.method == "kd":
        teacher, teacher_params, t_mc = load_teacher5(cfg, dev, rd / "tokenizer.json", L)
        assert cfg.max_len <= t_mc["ctx"]

    net = build_net5(cfg, lm, L, prior)
    apply_method5(net, cfg)
    net.to(dev)
    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"[model] total params {n_total / 1e6:.1f}M | trainable {n_train / 1e6:.3f}M | method={cfg.method}")

    peft = cfg.method in ("lora", "lorma", "adapter", "bitfit", "prompt")
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
        else:                                            # label-wise head, prompt, projection
            base, wd = cfg.head_lr, (0.0 if n_ in ("b", "prompt") or p_.dim() < 2 else cfg.weight_decay)
        buckets.setdefault((round(base, 12), wd), []).append(p_)
    opt = torch.optim.AdamW([{"params": ps, "lr": b_, "base": b_, "weight_decay": wd} for (b_, wd), ps in buckets.items()],
                            betas=(0.9, 0.98), eps=1e-8)
    scaler = P1.make_scaler(dev == "cuda" and amp_dtype == torch.float16)
    pos_weight = torch.tensor(np.clip(np.sqrt((len(tr) - freq + 1.0) / (freq + 1.0)), 1.0, cfg.pos_weight_max), dtype=torch.float32, device=dev)
    ema = J.EMA(net, cfg.ema_decay) if cfg.ema_decay > 0 else None

    rng = np.random.default_rng(cfg.seed)
    prng = random.Random(cfg.seed)
    dev_sub = np.sort(rng.permutation(len(dv))[:cfg.dev_max_docs]) if cfg.dev_max_docs and len(dv) > cfg.dev_max_docs else np.arange(len(dv))
    n_epoch = len(tr) if not cfg.docs_per_epoch else min(cfg.docs_per_epoch, len(tr))
    steps_per_epoch = max(1, math.ceil(n_epoch / (cfg.batch_docs * cfg.accum)))
    total_steps = cfg.epochs * steps_per_epoch
    budget = cfg.time_budget_min * 60.0
    log(f"[train] {n_epoch} cases/epoch, {steps_per_epoch} steps/epoch x {cfg.epochs} epochs; time budget {cfg.time_budget_min} min; "
        f"{cfg.train_chunks} random chunks per case; EMA {cfg.ema_decay}")
    best, step, bad, t_start, stop = {"f1": -1.0}, 0, 0, time.time(), False
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logf = open(rd / "log.jsonl", "a")

    def do_eval(tag, loss_val):
        nonlocal best, bad
        with (ema.swapped() if ema is not None else contextlib.nullcontext()):
            P_ = doc_probs5(net, dv, cfg, dev, amp, eos, L, idxs=dev_sub)
            t_, f_ = best_global(P_, Ydv[dev_sub])
            sc_ = ml_scores(Ydv[dev_sub], P_ >= t_)
            log(f"[{tag}] loss {loss_val:.4f} | dev(subset {len(dev_sub)}) macro-F1 {f_:.4f} micro-F1 {sc_['micro_f1']:.4f} at global threshold {t_:.3f} | "
                f"{time.time() - t_start:.0f}s")
            logf.write(json.dumps({"tag": tag, "step": step, "loss": loss_val, "dev_macro_f1": f_, "dev_micro_f1": sc_["micro_f1"]}) + "\n")
            logf.flush()
            if f_ > best["f1"]:
                best, bad = {"f1": f_, "epoch": tag}, 0
                torch.save({"net": net.state_dict(), "mc": mc, "cfg": jd, "n_labels": L, "decision": None}, rd / "best.pt")
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
                g["lr"] = J.lr_prog(g["base"], prog, cfg.warmup_frac)
            group = order[g0:g0 + micro * cfg.accum]
            for m0 in range(0, len(group), micro):
                batch = collate5(tr, group[m0:m0 + micro], cfg, dev, eos, L, cfg.train_chunks, cfg.train_select, prng)
                Y = batch[6]
                with amp():
                    logits = net(*batch[:6])
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(*batch[:6])
                loss = ml_loss(logits, Y, cfg, pos_weight)
                if teacher is not None:
                    loss = cfg.kd_alpha * loss + (1.0 - cfg.kd_alpha) * kd_loss_ml(logits, t_logits, cfg.kd_temp)
                scaler.scale(loss * (len(Y) / len(group))).backward()
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
    Pdv = doc_probs5(net, dv, cfg, dev, amp, eos, L)
    if dev == "cuda":
        torch.cuda.synchronize()
    infer_docs_s = len(dv) / max(1e-9, time.time() - t_inf)
    dec = choose_decision(Pdv, Ydv)
    t_g, f_g = best_global(Pdv, Ydv)
    sc = dec["scores"]
    ck["decision"] = {k: dec[k] for k in ("name", "thr", "ensure_one", "t_global")}
    torch.save(ck, rd / "best.pt")
    np.save(rd / "dev_probs.npy", Pdv)
    P1.jdump([d.id for d in dv], rd / "dev_ids.json")
    P1.jdump(label_matrix(dv, L).tolist(), rd / "dev_labels.json")
    log(f"[dev] decision strategies (2-fold cross-fit macro-F1 on dev): { {k: round(v, 4) for k, v in dec['cross_fit'].items()} } -> chosen '{dec['name']}'")
    log(f"[dev] macro-F1 {sc['macro_f1']:.4f} (global threshold {t_g:.3f} alone: {f_g:.4f}) | micro-F1 {sc['micro_f1']:.4f} | subset accuracy {sc['subset_acc']:.4f} | "
        f"macro-P {sc['macro_p']:.4f} macro-R {sc['macro_r']:.4f}")
    supp = Ydv.sum(0)
    for lo, hi, name in ((0, 1, "absent in dev"), (1, 5, "1-4 dev cases"), (5, 20, "5-19 dev cases"), (20, 10 ** 9, ">= 20 dev cases")):
        mk = (supp >= lo) & (supp < hi)
        if mk.any():
            log(f"[dev] labels with {name:<15} ({int(mk.sum()):>3}): mean F1 {sc['f1'][mk].mean():.4f}")
    worst = np.argsort(-supp)[:10]
    log("[dev] 10 most frequent statutes (id: dev cases, F1): " + ", ".join(f"{int(l)}: {int(supp[l])}, {sc['f1'][l]:.2f}" for l in worst))
    variant = cfg.lorma_variant if cfg.method == "lorma" else ""
    P1.jdump({"best_epoch": best["epoch"], "decision": ck["decision"], "cross_fit": dec["cross_fit"], "macro_f1": sc["macro_f1"],
              "micro_f1": sc["micro_f1"], "subset_acc": sc["subset_acc"], "macro_p": sc["macro_p"], "macro_r": sc["macro_r"],
              "macro_f1_global_only": f_g, "n_labels": L, "per_label_f1": [float(x) for x in sc["f1"]], "label_support_dev": supp.tolist(),
              "label_support_train": freq.tolist(), "params_total": n_total, "params_trainable": n_train, "train_seconds": train_s,
              "peak_gpu_mb": peak, "infer_docs_per_s": infer_docs_s, "p1_arch": mc["arch"], "method": cfg.method,
              "teacher_run": cfg.teacher_run, "teacher_params": teacher_params}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": mc["arch"], "method": cfg.method + (f"-{variant}" if variant else ""), "pool": cfg.pool, "loss": cfg.loss,
                  "params_total_M": round(n_total / 1e6, 2), "params_trainable_M": round(n_train / 1e6, 3),
                  "dev_macro_f1": round(sc["macro_f1"], 4), "dev_micro_f1": round(sc["micro_f1"], 4), "dev_subset_acc": round(sc["subset_acc"], 4),
                  "decision": dec["name"], "n_labels": L, "train_s": round(train_s), "peak_gpu_mb": round(peak),
                  "infer_docs_s": round(infer_docs_s, 2), "teacher_run": cfg.teacher_run})
    if cfg.predict_after:
        run_predict5(cfg)


def write_submission5(cfg: LCfg, rd: Path, ids, pred_mat, L):
    out = []
    for i, row in zip(ids, pred_mat):
        labs = sorted({int(x) for x in np.where(row)[0]})
        assert all(0 <= x < L for x in labs)
        out.append({"id": i, "pred": labs})
    assert len(out) == len(set(ids))
    rd.mkdir(parents=True, exist_ok=True)
    pj = rd / "prediction.json"
    P1.jdump(out, pj)
    zp = rd / "submission.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    hist = Path(cfg.work_dir) / "ls_submissions"
    hist.mkdir(parents=True, exist_ok=True)
    shutil.copy(zp, hist / f"{cfg.run}_{time.strftime('%m%d_%H%M%S')}.zip")
    log(f"[predict] wrote {pj} ({len(out)} cases, {np.mean([len(o['pred']) for o in out]):.2f} statutes per case on average) and {zp}")
    return str(zp)


def run_predict5(cfg: LCfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, amp = P1.amp_setup(dev)
    r = load_run5(cfg, cfg.run, dev)
    saved = r["saved"]
    eos = r["tok"].token_to_id(P1.EOS)
    te = load_split5(saved, saved.test_file, r["tok"], False, 0, str(r["rd"] / "tokenizer.json"))
    P = doc_probs5(r["net"], te, saved, dev, amp, eos, r["L"])
    np.save(r["rd"] / "test_probs.npy", P)
    P1.jdump([d.id for d in te], r["rd"] / "test_ids.json")
    dec = r["ck"]["decision"]
    pred = apply_decision(P, np.array(dec["thr"]), dec["ensure_one"])
    log(f"[predict] {len(te)} test cases, decision '{dec['name']}'")
    return write_submission5(cfg, r["rd"], [d.id for d in te], pred, r["L"])


# ============================================================================================
# 5. Ensemble (average saved probabilities of finished runs)
# ============================================================================================
def run_ensemble5(cfg: LCfg):
    root = Path(cfg.work_dir) / "ls_runs"
    if cfg.ens_runs:
        names = list(cfg.ens_runs)
    elif cfg.ens_auto:
        names = [f.parent.name for f in sorted(root.glob("*/test_probs.npy")) if (f.parent / "dev_probs.npy").exists()
                 and (f.parent.name.startswith(cfg.ens_prefix) if cfg.ens_prefix else not f.parent.name.startswith(("selftest", "check")))]
    else:
        names = []
    if len(names) < 2:
        raise SystemExit('need >= 2 finished runs: set "ens_runs": [...] or "ens_auto": true')
    dev_ids, test_ids = P1.jload(root / names[0] / "dev_ids.json"), P1.jload(root / names[0] / "test_ids.json")
    Ydv = np.array(P1.jload(root / names[0] / "dev_labels.json"), dtype=np.uint8)
    L = Ydv.shape[1]
    dv_p, te_p = {}, {}
    for n in names:
        if P1.jload(root / n / "dev_ids.json") != dev_ids or P1.jload(root / n / "test_ids.json") != test_ids or np.load(root / n / "dev_probs.npy").shape[1] != L:
            log(f"[ensemble] skipping {n}: different cases / label space")
            continue
        dv_p[n], te_p[n] = np.load(root / n / "dev_probs.npy"), np.load(root / n / "test_probs.npy")
    quick = lambda P: best_global(P, Ydv)[1]
    single = {k: quick(v) for k, v in dv_p.items()}
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
            f = quick(mean(chosen + [k], dv_p))
            log(f"[ensemble] try adding {k:<28} -> {f:.4f} (current {cur:.4f})")
            if f > cur + 0.002:
                chosen.append(k)
                cur = f
    log(f"[ensemble] members: {chosen}")
    Pd, Pt = mean(chosen, dv_p), mean(chosen, te_p)
    dec = choose_decision(Pd, Ydv)
    sc = dec["scores"]
    log(f"[ensemble] cross-fit { {k: round(v, 4) for k, v in dec['cross_fit'].items()} } -> '{dec['name']}' | dev macro-F1 {sc['macro_f1']:.4f} "
        f"micro-F1 {sc['micro_f1']:.4f} subset acc {sc['subset_acc']:.4f}")
    rd = run_dir5(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    P1.jdump({"members": chosen, "single_model_f1": single, "decision": {k: dec[k] for k in ("name", "t_global", "ensure_one")},
              "macro_f1": sc["macro_f1"], "micro_f1": sc["micro_f1"], "subset_acc": sc["subset_acc"]}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": "ensemble", "method": "ensemble", "dev_macro_f1": round(sc["macro_f1"], 4),
                  "dev_micro_f1": round(sc["micro_f1"], 4), "dev_subset_acc": round(sc["subset_acc"], 4), "decision": dec["name"],
                  "n_labels": L, "teacher_run": "+".join(chosen)})
    return write_submission5(cfg, rd, test_ids, apply_decision(Pt, np.array(dec["thr"]), dec["ensure_one"]), L)


# ============================================================================================
# 6. Diagnostics: check + probe
# ============================================================================================
def diagnose5(cfg: LCfg):
    tok, _, mc = N.load_p1(cfg)
    tp = str(Path(cfg.p1_dir) / "tokenizer.json")
    sp = {}
    for name, rel, lab in (("train", f"{cfg.train_prefix}/train.json", True), ("dev", f"{cfg.train_prefix}/dev.json", True), ("test", cfg.test_file, False)):
        sp[name] = load_split5(cfg, rel, tok, lab, cfg.max_train_docs if name == "train" else 0, tp)
        docs, n = sp[name], np.array([d.n for d in sp[name]])
        nch = np.array([len(J.chunk_bounds(d.n, cfg.max_len - 1, cfg.overlap)) for d in docs])
        line = (f"[check] {name:<5}: {len(docs):>6} cases | tokens/case mean {n.mean():.0f} median {np.median(n):.0f} p90 {np.percentile(n, 90):.0f} max {n.max()} | "
                f"chunks/case median {np.median(nch):.0f} p90 {np.percentile(nch, 90):.0f} (eval_chunks={cfg.eval_chunks} covers {100 * np.mean(nch <= cfg.eval_chunks):.0f}% of cases fully)")
        if lab:
            nl = np.array([len(d.labels) for d in docs])
            line += f" | statutes/case mean {nl.mean():.2f} max {nl.max()} | cases without any statute: {int((nl == 0).sum())}"
        log(line)
        _, nd, nc = dedupe_docs5(docs)
        log(f"[check] {name}: {nd} exact duplicate cases ({nc} with conflicting label sets)")
    L = 1 + max(max((max(d.labels) for d in sp[s] if d.labels), default=0) for s in ("train", "dev"))
    Ytr, Ydv = label_matrix(sp["train"], L), label_matrix(sp["dev"], L)
    f = Ytr.sum(0)
    order = np.argsort(-f)
    log(f"[check] {L} statute ids in train+dev | cases per statute in train: max {int(f.max())}, median {int(np.median(f))}, min {int(f.min())} | "
        f"statutes with 0 / <5 / <20 train cases: {int((f == 0).sum())} / {int((f < 5).sum())} / {int((f < 20).sum())}")
    log("[check] most frequent statutes (id: train cases): " + ", ".join(f"{int(l)}: {int(f[l])}" for l in order[:12]))
    log(f"[check] statutes present in dev but absent from train: {int(((Ydv.sum(0) > 0) & (f == 0)).sum())} | "
        f"top-1 statute alone would give micro-F1 {ml_scores(Ydv, np.tile((np.arange(L) == order[0]), (len(Ydv), 1)))['micro_f1']:.3f} on dev")
    co = np.triu(Ytr.T.astype(np.float32) @ Ytr.astype(np.float32), k=1)
    top_pairs = np.dstack(np.unravel_index(np.argsort(-co.ravel())[:5], co.shape))[0]
    log("[check] most frequent statute pairs in the same case: " + ", ".join(f"({int(a)},{int(b)}): {int(co[a, b])}" for a, b in top_pairs if co[a, b] > 0))

    def key(d, mode):
        t = d.tok[:256] if mode == "first 256 tokens" else d.tok[-256:] if mode == "last 256 tokens" else d.tok
        return hashlib.md5(t.tobytes()).hexdigest()
    for mode in ("full text", "first 256 tokens", "last 256 tokens"):
        tmap = {}
        for d in sp["train"]:
            tmap.setdefault(key(d, mode), []).append(set(d.labels))
        for name in ("dev", "test"):
            hit = [(d, tmap[key(d, mode)]) for d in sp[name] if key(d, mode) in tmap]
            extra = ""
            if name == "dev" and hit:
                jac = [len(set(d.labels) & u[0]) / max(1, len(set(d.labels) | u[0])) for d, u in hit]
                extra = f" | label-set Jaccard with the train copy {np.mean(jac):.2f}"
            log(f"[overlap] {name:<4} cases whose {mode:<16} also occurs in train: {len(hit):>5}/{len(sp[name])}{extra}")


def run_probe(cfg: LCfg):
    """DIAGNOSTIC ONLY: which part of a case identifies the statutes? One-vs-rest logistic regression on bag-of-n-grams of one region at a time,
    for the most frequent statutes; reports the mean F1 on dev per region and the strongest n-grams of the best region for the top statute."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise SystemExit("probe needs scikit-learn:  pip install scikit-learn")
    from tokenizers import Tokenizer
    tp = str(Path(cfg.p1_dir) / "tokenizer.json")
    tok = Tokenizer.from_file(tp)
    tr = load_split5(cfg, f"{cfg.train_prefix}/train.json", tok, True, cfg.max_train_docs, tp)
    dv = load_split5(cfg, f"{cfg.train_prefix}/dev.json", tok, True, 0, tp)
    tr, _, _ = dedupe_docs5(tr)
    dv, _, _ = dedupe_docs5(dv)
    L = 1 + max(max((max(d.labels) for d in docs if d.labels), default=0) for docs in (tr, dv))
    rng = np.random.default_rng(0)
    trs = [tr[i] for i in rng.permutation(len(tr))[:cfg.probe_docs]]
    Ytr, Ydv = label_matrix(trs, L), label_matrix(dv, L)
    top = [int(l) for l in np.argsort(-Ytr.sum(0))[:cfg.probe_labels] if Ytr[:, l].sum() >= 10 and Ydv[:, l].sum() >= 1]
    log(f"[probe] {len(trs)} training cases, {len(dv)} dev cases, statutes probed: {top}")
    mid = lambda t: t[max(0, len(t) // 2 - 1024): len(t) // 2 + 1024]
    regions = {"first 512 tokens": lambda t: t[:512], "first 2048 tokens": lambda t: t[:2048], "last 512 tokens": lambda t: t[-512:],
               "last 2048 tokens": lambda t: t[-2048:], "middle 2048 tokens": mid,
               "first 4096 + last 4096": lambda t: np.concatenate([t[:4096], t[-4096:]]) if len(t) > 8192 else t}
    text = lambda t: " ".join(f"t{int(x)}" for x in t)
    results, best = {}, (-1.0, None, None, None)
    for name, fn in regions.items():
        vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=3, max_features=150000, sublinear_tf=True, lowercase=False)
        Xtr = vec.fit_transform([text(fn(d.tok)) for d in trs])
        Xdv = vec.transform([text(fn(d.tok)) for d in dv])
        f1s, clfs = [], {}
        for l in top:
            clf = LogisticRegression(max_iter=200, C=4.0, class_weight="balanced").fit(Xtr, Ytr[:, l])
            p = clf.predict_proba(Xdv)[:, 1]
            f1s.append(max(per_label_f1(Ydv[:, [l]], (p >= t)[:, None])[0][0] for t in (0.3, 0.5, 0.7)))
            clfs[l] = clf
        results[name] = float(np.mean(f1s))
        log(f"[probe] {name:<24} mean F1 over the probed statutes on dev: {results[name]:.4f}")
        if results[name] > best[0]:
            best = (results[name], name, vec, clfs)
    _, name, vec, clfs = best
    l0 = top[0]
    coef, names = clfs[l0].coef_[0], vec.get_feature_names_out()
    order = np.argsort(coef)[::-1][:15]
    show = lambda f: tok.decode([int(x[1:]) for x in f.split()]).replace("\n", " ").strip()
    log(f"[probe] BEST region: {name}. Strongest n-grams for statute {l0}: " + " | ".join(show(names[i]) for i in order))
    P1.jdump(results, Path(cfg.work_dir) / "ls_probe.json")
    return results


# ============================================================================================
# 7. Self-test
# ============================================================================================
def _synthetic(tmp, rng, L=8):
    filler = "the court held that appeal petitioner respondent order section act learned counsel submitted impugned judgment".split()

    def doc(labs):
        words = [rng.choice(filler) for _ in range(rng.randint(300, 1500))]
        for l in labs:
            for _ in range(rng.randint(2, 3)):
                words.insert(rng.randint(0, len(words)), f"statute{l}x")
        return " ".join(words)

    os.makedirs(tmp, exist_ok=True)
    texts = []
    for split, n in (("train", 60), ("dev", 16), ("test", 12)):
        recs = []
        for i in range(n):
            labs = sorted(rng.sample(range(L), rng.randint(1, 3))) if rng.random() < 0.9 else [0]
            t = doc(labs)
            texts.append(t)
            r = {"id": f"{split}_{i}", "text": t}
            if split != "test":
                r["labels"] = labs
            recs.append(r)
        json.dump(recs, open(os.path.join(tmp, f"{split}.json"), "w"))
    return texts


def selftest() -> bool:
    import tempfile
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    ok = True
    rng = random.Random(0)
    tmp = tempfile.mkdtemp(prefix="cs781_ls_selftest_")
    dd = os.path.join(tmp, "data")
    texts = _synthetic(dd, rng)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([" ".join(t.split()) for t in texts], trainers.BpeTrainer(
        vocab_size=500, special_tokens=[P1.EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    V = P1.pad64(tok.get_vocab_size())
    eos = tok.token_to_id(P1.EOS)

    log("=== 1. data, cache, metrics, thresholds, losses ===")
    p1x = os.path.join(tmp, "p1_cache")
    os.makedirs(p1x, exist_ok=True)
    tok.save(os.path.join(p1x, "tokenizer.json"))
    cfg0 = apply_overrides(LCfg(), {"local_data_dir": dd, "work_dir": os.path.join(tmp, "work0"), "max_len": 64})
    tr = load_split5(cfg0, "Phase_5_LSI/Train/train.json", tok, True, 0, os.path.join(p1x, "tokenizer.json"))
    tr2 = load_split5(cfg0, "Phase_5_LSI/Train/train.json", tok, True, 0, os.path.join(p1x, "tokenizer.json"))     # second call: from the cache
    g1 = len(tr) == 60 and all(np.array_equal(a.tok, b.tok) and a.labels == b.labels for a, b in zip(tr, tr2)) and any((Path(cfg0.work_dir) / "cache").glob("lsi_*.pkl"))
    L = 8
    Y = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    Yh = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    sc = ml_scores(Y, Yh)
    f_ref = [2 * 2 / (2 * 2 + 0 + 0), 2 * 2 / (2 * 2 + 0 + 0), 0.0, 0.0]            # per-label F1 by hand
    g2 = abs(sc["macro_f1"] - np.mean(f_ref)) < 1e-9 and abs(sc["subset_acc"] - 0.5) < 1e-9 and abs(sc["micro_f1"] - 2 * 4 / (2 * 4 + 1 + 1)) < 1e-9
    P = np.array([[0.9, 0.1, 0.2], [0.2, 0.3, 0.1], [0.6, 0.55, 0.1]])
    pr = apply_decision(P, np.full(3, 0.5), True)
    g3 = pr[1].sum() == 1 and pr[1, 1] and not apply_decision(P, np.full(3, 0.5), False)[1].any()
    Pm = np.random.default_rng(0).random((40, 6))
    Ym = (Pm > 0.7).astype(np.uint8)
    dec = choose_decision(Pm, Ym)
    g4 = dec["name"] in ("global", "global+one", "per_label", "per_label+one") and len(dec["thr"]) == 6 and dec["scores"]["macro_f1"] >= 0.5
    cfgl = apply_overrides(LCfg(), {})
    lg, Yt = torch.randn(4, 6), torch.randint(0, 2, (4, 6)).float()
    pw = torch.full((6,), 3.0)
    g5 = all(float(ml_loss(lg, Yt, apply_overrides(LCfg(), {"loss": k}), pw)) > 0 for k in ("bce", "bce_pw", "focal", "asl")) and \
        abs(float(kd_loss_ml(lg, lg, 2.0)) - float(F.binary_cross_entropy_with_logits(lg / 2.0, torch.sigmoid(lg / 2.0)) * 4)) < 1e-6
    g6 = pick_chunks(10, 3, "tail", random) == [7, 8, 9] and pick_chunks(10, 3, "uniform", random)[0] == 0 and pick_chunks(10, 3, "uniform", random)[-1] == 9 \
        and len(pick_chunks(10, 3, "random", random.Random(1))) == 3
    ok &= all([g1, g2, g3, g4, g5, g6])
    log(f"  token cache {'PASS' if g1 else 'FAIL'} | metrics = evaluator definitions {'PASS' if g2 else 'FAIL'} | 'at least one statute' rule {'PASS' if g3 else 'FAIL'} | "
        f"decision strategies {'PASS' if g4 else 'FAIL'} | losses {'PASS' if g5 else 'FAIL'} | chunk picking {'PASS' if g6 else 'FAIL'}")

    log("=== 2. methods: identity at init (adapters / LoRMA / BitFit / freezing), prompt tuning trains only prompt + head ===")
    prior = np.full(L, 0.2)
    for arch in P1.ARCH_PRESETS:
        for meth, var in (("adapter", "plus"), ("lora", "plus"), ("lorma", "plus"), ("lorma", "pi"), ("bitfit", "plus"), ("partial", "plus"), ("prompt", "plus")):
            tag = f"{arch}/{meth}" + (f"-{var}" if meth == "lorma" else "")
            try:
                torch.manual_seed(0)
                c = apply_overrides(LCfg(), {"method": meth, "lorma_variant": var, "peft_r": 4, "peft_alpha": 8.0, "peft_dropout": 0.0, "adapter_dim": 8,
                                             "freeze_layers": 1, "chunk_dim": 32, "max_len": 64, "prompt_tokens": 4})
                for pool in ("attn", "chunk_max", "max", "mean") if (arch == "llama" and meth == "adapter") else ("attn",):
                    c.pool = pool
                    net = build_net5(c, P1.LM(N._tiny_mc(arch, V)), L, prior).eval()
                    inp = collate5(tr, [0, 1], c, "cpu", eos, L, 3, "uniform")[:6]
                    with torch.no_grad():
                        ref = net(*inp)
                    assert ref.shape == (2, L)
                    if pool != "attn":
                        continue
                    apply_method5(net, c)
                    net.eval()
                    with torch.no_grad():
                        out = net(*inp)
                    diff = (ref - out).abs().max().item()
                    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    n_all = sum(p.numel() for p in net.parameters())
                    net.train()
                    net(*inp).float().pow(2).mean().backward()
                    grads = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.parameters() if p.requires_grad)
                    good = (diff < 1e-4 or meth == "prompt") and n_tr < n_all and grads
                    ok &= good
                    log(f"  {tag:<18} |adapted - base| = {diff:.1e}  trainable {n_tr:,}/{n_all:,}  {'PASS' if good else 'FAIL'}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  {tag:<18} FAIL: {type(e).__name__}: {e}")
                ok = False

    log("=== 3. end-to-end: architectures / methods / pooling / losses, then ensembles ===")
    p1dirs = {}
    for arch in P1.ARCH_PRESETS:
        p1 = os.path.join(tmp, f"p1_{arch}")
        os.makedirs(p1, exist_ok=True)
        torch.save({"model": P1.LM(N._tiny_mc(arch, V)).state_dict(), "mc": N._tiny_mc(arch, V), "state": {"step": 0}}, os.path.join(p1, "ckpt_best.pt"))
        tok.save(os.path.join(p1, "tokenizer.json"))
        p1dirs[arch] = p1
    base = {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "max_len": 64, "train_chunks": 3, "eval_chunks": 6, "epochs": 2, "batch_docs": 2,
            "accum": 1, "eval_docs": 2, "lr": 1e-3, "peft_lr": 3e-3, "head_lr": 5e-3, "peft_r": 4, "peft_alpha": 8.0, "chunk_dim": 32, "adapter_dim": 8,
            "ema_decay": 0.9, "time_budget_min": 5, "patience": 5, "dev_max_docs": 12, "prompt_tokens": 4}
    cases = [(f"full_{a}", a, {"method": "full"}) for a in P1.ARCH_PRESETS] + [
        ("lorma_plus", "gpt2", {"method": "lorma", "lorma_variant": "plus"}), ("lorma_pi", "retnet", {"method": "lorma", "lorma_variant": "pi"}),
        ("lora", "llama", {"method": "lora"}), ("adapter", "moe", {"method": "adapter"}), ("bitfit", "gpt2", {"method": "bitfit"}),
        ("partial", "llama", {"method": "partial", "freeze_layers": 1}), ("prompt", "llama", {"method": "prompt"}),
        ("pool_chunkmax", "llama", {"method": "full", "pool": "chunk_max"}), ("pool_mean", "llama", {"method": "full", "pool": "mean"}),
        ("loss_asl", "llama", {"method": "full", "loss": "asl"}), ("loss_focal", "llama", {"method": "full", "loss": "focal"}),
        ("kd", "gpt2", {"method": "kd", "teacher_run": "selftest_full_llama", "student_layers": 1})]
    for tag, arch, extra in cases:
        try:
            cfg = apply_overrides(LCfg(), {**base, **extra, "run": f"selftest_{tag}", "p1_dir": p1dirs[arch]})
            run_train5(cfg)
            with zipfile.ZipFile(run_dir5(cfg) / "submission.zip") as z:
                assert z.namelist() == ["prediction.json"]
                pr = json.loads(z.read("prediction.json"))
            assert isinstance(pr, list) and [x["id"] for x in pr] == [f"test_{i}" for i in range(12)]
            assert all(set(x) == {"id", "pred"} and isinstance(x["pred"], list) and len(set(x["pred"])) == len(x["pred"])
                       and all(type(v) is int and 0 <= v < 8 for v in x["pred"]) for x in pr)
            rep = P1.jload(run_dir5(cfg) / "dev_report.json")
            if extra["method"] in ("lora", "lorma", "adapter", "bitfit", "partial", "prompt"):
                assert rep["params_trainable"] < rep["params_total"]
            if extra["method"] == "kd":
                assert rep["params_total"] < rep["teacher_params"]
            log(f"  e2e {tag:<14} PASS  (dev macro-F1 {rep['macro_f1']:.3f}, trainable {rep['params_trainable']:,} / {rep['params_total']:,})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  e2e {tag:<14} FAIL: {type(e).__name__}: {e}")
            ok = False
    try:
        for tag, ov in (("ens", {"ens_runs": ["selftest_full_llama", "selftest_full_gpt2"]}), ("ens_auto", {"ens_auto": True, "ens_prefix": "selftest_"})):
            cfg = apply_overrides(LCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "run": f"selftest_{tag}", **ov})
            zp = run_ensemble5(cfg)
            with zipfile.ZipFile(zp) as z:
                pr = json.loads(z.read("prediction.json"))
            assert len(pr) == 12
        log("  ensembles (explicit + automatic) PASS")
        pcfg = apply_overrides(LCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "p1_dir": p1dirs["llama"], "probe_docs": 40, "probe_labels": 4})
        res = run_probe(pcfg)
        assert res["last 512 tokens"] >= 0.0
        log("  probe stage PASS")
    except ImportError:
        log("  probe stage skipped (scikit-learn not installed)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"  ensembles / probe FAIL: {type(e).__name__}: {e}")
        ok = False
    shutil.rmtree(tmp, ignore_errors=True)
    log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


# ============================================================================================
# 8. CLI
# ============================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["selftest", "check", "probe", "train", "predict", "ensemble"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--cfg", default="", help="JSON string or JSON file with config overrides")
    args = ap.parse_args(argv)
    if args.stage == "selftest":
        return 0 if selftest() else 1
    cfg = LCfg()
    ov = {}
    if args.cfg:
        ov = json.loads(args.cfg) if args.cfg.strip().startswith("{") else P1.jload(args.cfg)
    if args.run:
        ov["run"] = args.run
    apply_overrides(cfg, ov)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    {"check": diagnose5, "probe": run_probe, "train": run_train5, "predict": run_predict5, "ensemble": run_ensemble5}[args.stage](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
