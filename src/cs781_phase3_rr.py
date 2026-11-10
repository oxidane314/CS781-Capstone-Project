#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS781 Phase 3 - Rhetorical Role prediction (CL + IT) on top of YOUR Phase-1 language models.

Needs cs781_phase1.py and cs781_phase2_ner.py in the same folder (model classes, LoRA / LoRMA adapters, distillation helpers).

Model
  document -> sentences -> Phase-1 BPE tokens   [sent_1 <eos> sent_2 <eos> ...]   packed into windows of <= max_len tokens
  -> Phase-1 backbone (bidirectional attention, or causal for retnet / adapter runs)
  -> one vector per sentence = [mean of its token states ; state of its <eos> marker]
  -> + relative / absolute position-in-document + domain (CL / IT) embeddings
  -> BiGRU over the sentences of the window (document context)  -> linear -> 13 rhetorical roles
  ctx_mode "docseq" (recommended): the BiGRU runs over the sentences of the WHOLE document (training on crops of consecutive windows,
  inference on the full document), so every sentence sees the structure of the entire judgment. "doc" = overlapping windows,
  "sentence" = every sentence alone (ablation).
  Decoding is calibrated on the dev set: per-class thresholds (macro-F1) and a label-transition prior (Viterbi).
  Training uses weight averaging (EMA). Exactly one label per sentence, in the original order.

Adaptation methods (cfg "method")
  full    all weights                       lora    additive low-rank (W0 + s*BA)            rslora  LoRA with scale alpha/sqrt(r)
  lorma   multiplicative low-rank (LoRMA+ / LoRMAπ)   ia3   learned scaling vectors on keys, values and FFN activations
  kd      knowledge distillation: pruned student <- fine-tuned teacher (logits)               partial freeze embeddings + first blocks

Stages:  selftest | check | train | predict | ensemble
  python cs781_phase3_rr.py check --run rr_check --cfg '{"p1_dir": ".../explore_llama"}'
  python cs781_phase3_rr.py train --run rr_full_llama --cfg cfg_rr.json
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
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
OFFICIAL_LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent", "PrecedentReliedUpon",
                   "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
                   "RulingByPresentCourt", "None"]
DOMAINS = ["CL", "IT"]


# ============================================================================================
# 0. Config
# ============================================================================================
@dataclass
class RCfg:
    run: str = "rr_full_v1"
    work_dir: str = P1.DEFAULT_WORK
    p1_dir: str = ""                  # Phase-1 run folder (ckpt_best.pt + tokenizer.json)
    p1_ckpt: str = "ckpt_best.pt"
    hf_repo: str = "Exploration-Lab/CS-781-Capstone"
    local_data_dir: str = ""          # folder with CL_train.json, CL_dev.json, IT_train.json, IT_dev.json, label_vocab.json, CL_test.json, IT_test.json
    train_prefix: str = "Phase_3_RR/Train"
    test_prefix: str = "Phase_3_RR/Test"
    # ---- context
    ctx_mode: str = "docseq"          # docseq = BiGRU over the whole document | doc = overlapping windows | sentence = every sentence alone
    crop_windows: int = 8             # docseq training: a crop = this many consecutive windows (~ 8 x max_len tokens)
    batch_docs: int = 2               # docseq training: crops per step
    max_len: int = 512                # tokens per window (<= Phase-1 context)
    max_sent_tokens: int = 96         # longer sentences are truncated
    win_overlap: float = 0.5          # fraction of sentences shared by consecutive windows
    # ---- method
    method: str = "full"              # full | partial | lora | rslora | lorma | ia3 | kd
    freeze_layers: int = 0
    peft_r: int = 16
    peft_alpha: float = 32.0
    peft_dropout: float = 0.05
    peft_lr: float = 5e-4
    peft_targets: str = "all"         # attn | all
    peft_experts: bool = False
    lorma_variant: str = "plus"       # plus | pi
    teacher_run: str = ""
    student_layers: int = 0           # 0 = half of the Phase-1 blocks
    kd_alpha: float = 0.5
    kd_temp: float = 2.0
    # ---- architecture of the sentence head
    bidir: str = "auto"               # auto = bidirectional attention for gpt2/llama/moe, causal for retnet
    head: str = "bigru"               # bigru (sequence model over sentences) | linear
    sent_dim: int = 384
    head_hidden: int = 256
    head_layers: int = 1
    head_dropout: float = 0.1
    # ---- optimisation
    epochs: int = 12
    patience: int = 6
    batch_size: int = 8
    eval_batch: int = 16
    lr: float = 3e-4
    llrd: float = 0.85
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    class_weight: str = "sqrt"        # none | sqrt
    label_smoothing: float = 0.0
    class_bias: bool = True           # per-class thresholds tuned on dev for macro-F1
    trans_decode: bool = True         # Viterbi with a label-transition prior estimated on the training documents (weight tuned on dev)
    ema_decay: float = 0.998          # exponential moving average of the trainable weights (0 = off); the EMA model is evaluated and saved
    seed: int = 1337
    predict_after: bool = True
    # ---- ensemble stage
    ens_runs: list = field(default_factory=list)
    ens_auto: bool = False
    ens_prefix: str = ""
    ens_max: int = 6


def apply_overrides(cfg: RCfg, ov: dict) -> RCfg:
    names = {f.name for f in fields(RCfg)}
    for k, v in ov.items():
        if k not in names:
            raise SystemExit(f"Unknown config key {k!r}. Valid: {sorted(names)}")
        setattr(cfg, k, v)
    return cfg


def run_dir3(cfg: RCfg) -> Path:
    return Path(cfg.work_dir) / "rr_runs" / cfg.run


RR_FIELDS = ["time", "run", "p1_arch", "method", "variant", "ctx_mode", "head", "bidir", "params_total_M",
             "params_trainable_M", "dev_macro_f1", "dev_macro_p", "dev_macro_r", "dev_cl_f1", "dev_it_f1", "train_s",
             "peak_gpu_mb", "infer_sent_s", "teacher_run"]


def log_csv(cfg: RCfg, row: dict):
    p = Path(cfg.work_dir) / "rr_experiments.csv"
    if p.exists() and open(p).readline().strip().split(",") != RR_FIELDS:
        p.rename(p.with_suffix(".old.csv"))
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RR_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row})


# ============================================================================================
# 1. Data: tolerant JSON parsing, tokenisation, windows
# ============================================================================================
def parse_vocab(raw):
    """Same accepted formats as the official evaluator: list, label->id, or id->label. Returns (labels, id->label)."""
    if isinstance(raw, list):
        labels = [str(x) for x in raw]
        return labels, {i: l for i, l in enumerate(labels)}
    if isinstance(raw, dict) and raw:
        vals = list(raw.values())
        if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            return list(raw.keys()), {v: k for k, v in raw.items()}
        labels = [str(v) for v in vals]
        return labels, {(int(k) if str(k).isdigit() else k): str(v) for k, v in raw.items()}
    raise SystemExit("label_vocab.json must be a non-empty list or object")


def _first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_raw_docs(path, labels, id2):
    """-> list of {"id", "sents": [str], "labels": [str] | None}"""
    raw = P1.jload(path)
    if isinstance(raw, dict):
        items = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in raw.items()]
    else:
        items = raw
    lab_set, docs = set(labels), []
    for i, r in enumerate(items):
        did = _first(r, ("id", "doc_id", "document_id", "name"))
        did = str(did) if did is not None else str(i)
        text = _first(r, ("text", "sentences", "sents")) or []
        if isinstance(text, str):
            text = [text]
        sents = [s if isinstance(s, str) else str(_first(s, ("text", "sentence")) or "") for s in text]
        ls = _first(r, ("labels", "label", "roles"))
        if ls is not None:
            conv = []
            for x in ls:
                x = "None" if x is None else (id2.get(x, x) if not isinstance(x, str) else x)
                if x not in lab_set:
                    raise SystemExit(f"{path}: document {did} has label {x!r} which is not in label_vocab.json")
                conv.append(x)
            if len(conv) != len(sents):
                raise SystemExit(f"{path}: document {did} has {len(sents)} sentences but {len(conv)} labels")
            ls = conv
        docs.append({"id": did, "sents": sents, "labels": ls})
    return docs


class RDoc:
    pass


def build_docs(tok, raw_docs, label2i, domain, eos, max_sent_tokens):
    flat = [" ".join(s.split()) for d in raw_docs for s in d["sents"]]
    encs = []
    for i in range(0, len(flat), 20000):
        encs += tok.encode_batch(flat[i:i + 20000])
    k, out = 0, []
    for d in raw_docs:
        r = RDoc()
        r.id, r.domain, r.n = d["id"], domain, len(d["sents"])
        r.sents = []
        for _ in range(r.n):
            ids = encs[k].ids[:max_sent_tokens]
            k += 1
            r.sents.append(np.asarray(ids if ids else [eos], dtype=np.int64))
        r.lens = np.array([len(s) for s in r.sents], dtype=np.int64)
        r.lab = np.array([label2i[x] for x in d["labels"]], dtype=np.int64) if d["labels"] is not None else None
        out.append(r)
    return out


def make_windows3(d, cfg):
    """Pack consecutive sentences into windows of <= max_len tokens (each sentence costs len + 1 marker)."""
    if d.n == 0:
        return []
    if cfg.ctx_mode == "sentence":
        return [(i, i + 1) for i in range(d.n)]
    cost = d.lens + 1
    wins, a = [], 0
    while a < d.n:
        b, tot = a, 0
        while b < d.n and (tot + cost[b] <= cfg.max_len or b == a):
            tot += cost[b]
            b += 1
        wins.append((a, b))
        if b >= d.n:
            break
        a += max(1, int((b - a) * (1.0 - cfg.win_overlap)))
    return wins


def abs_bucket(i):
    return i if i < 3 else 3 if i < 5 else 4 if i < 10 else 5 if i < 20 else 6


def collate3(docs, items, dev, eos):
    B = len(items)
    L = max(sum(len(docs[di].sents[i]) + 1 for i in range(a, b)) for di, a, b in items)
    M = max(b - a for _, a, b in items)
    ids = np.zeros((B, L), dtype=np.int64)
    km = np.zeros((B, L), dtype=bool)
    pool = np.zeros((B, M, L), dtype=np.float32)
    mark = np.zeros((B, M), dtype=np.int64)
    sm = np.zeros((B, M), dtype=bool)
    lab = np.full((B, M), -100, dtype=np.int64)
    rel = np.zeros((B, M), dtype=np.int64)
    ab = np.zeros((B, M), dtype=np.int64)
    dom = np.zeros(B, dtype=np.int64)
    for j, (di, a, b) in enumerate(items):
        d, p = docs[di], 0
        dom[j] = d.domain
        for m, i in enumerate(range(a, b)):
            s = d.sents[i]
            n = len(s)
            ids[j, p:p + n] = s
            ids[j, p + n] = eos
            pool[j, m, p:p + n] = 1.0 / n
            mark[j, m] = p + n
            sm[j, m] = True
            rel[j, m] = min(9, int(10 * i / max(1, d.n)))
            ab[j, m] = abs_bucket(i)
            if d.lab is not None:
                lab[j, m] = d.lab[i]
            p += n + 1
        km[j, :p] = True
    t = lambda x: torch.from_numpy(x).to(dev)
    return t(ids), t(km), t(pool), t(mark), t(sm), t(rel), t(ab), t(dom), t(lab)


def load_all(cfg: RCfg, tok, with_test=False):
    res = lambda rel: P1.resolve_file(cfg, rel)
    labels, id2 = parse_vocab(P1.jload(res(f"{cfg.train_prefix}/label_vocab.json")))
    label2i = {l: i for i, l in enumerate(labels)}
    eos = tok.token_to_id(P1.EOS)
    out = {"labels": labels, "eos": eos}
    for split, prefix, labelled in (("train", cfg.train_prefix, True), ("dev", cfg.train_prefix, True), ("test", cfg.test_prefix, False)):
        if split == "test" and not with_test:
            continue
        docs = []
        for di, dom in enumerate(DOMAINS):
            raw = load_raw_docs(res(f"{prefix}/{dom}_{split}.json"), labels, id2)
            docs += build_docs(tok, raw, label2i, di, eos, cfg.max_sent_tokens)
        out[split] = docs
    return out


# ============================================================================================
# 2. Metrics (identical to the official evaluator: pooled macro over the 13 labels, zero_division=0)
# ============================================================================================
def macro_scores(y_true, y_pred, C):
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(np.float64)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    P = np.divide(tp, tp + fp, out=np.zeros(C), where=(tp + fp) > 0)
    R = np.divide(tp, tp + fn, out=np.zeros(C), where=(tp + fn) > 0)
    F1 = np.divide(2 * P * R, P + R, out=np.zeros(C), where=(P + R) > 0)
    return {"macro_f1": float(F1.mean()), "macro_p": float(P.mean()), "macro_r": float(R.mean()),
            "P": P.tolist(), "R": R.tolist(), "F1": F1.tolist(), "support": cm.sum(1).tolist(), "cm": cm}


def pooled(docs, probs, C, bias=None, domain=None):
    ys, ps = [], []
    for d, p in zip(docs, probs):
        if domain is not None and d.domain != domain:
            continue
        lp = np.log(np.clip(p, 1e-9, 1.0))
        if bias is not None:
            lp = lp + bias[None, :]
        ys.append(d.lab)
        ps.append(lp.argmax(1))
    if not ys:
        return macro_scores(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), C)
    return macro_scores(np.concatenate(ys), np.concatenate(ps), C)


def tune_bias(docs, probs, C, min_gain=0.002, grid=(-1.0, -0.5, -0.25, 0.25, 0.5, 1.0), passes=3):
    """Coordinate ascent on one additive log-probability bias per class to maximise pooled macro-F1 on the dev set."""
    y = np.concatenate([d.lab for d in docs])
    lp = np.log(np.clip(np.concatenate(probs), 1e-9, 1.0))
    f1 = lambda b: macro_scores(y, (lp + b[None, :]).argmax(1), C)["macro_f1"]
    b = np.zeros(C, dtype=np.float64)
    cur = f1(b)
    for _ in range(passes):
        improved = False
        for c in range(C):
            best_b, best_s = 0.0, cur
            for g in grid:
                t = b.copy()
                t[c] += g
                s = f1(t)
                if s > best_s + min_gain:
                    best_s, best_b = s, g
            if best_b != 0.0:
                b[c] += best_b
                cur, improved = best_s, True
        if not improved:
            break
    return b.astype(np.float32), cur


# ============================================================================================
# 3. Parameter-efficient adapters beyond Phase 2: rsLoRA and IA3
# ============================================================================================
class RsLoRALinear(N.LoRALinear):
    """Rank-stabilised LoRA (Kalajdzievski 2023): same as LoRA but scale = alpha / sqrt(r) instead of alpha / r."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__(base, r, alpha, dropout)
        self.scale = alpha / math.sqrt(r)


class IA3Out(nn.Module):
    """IA3 (Liu et al. 2022): rescale the OUTPUT of a projection (keys / values) with a learned vector, initialised to ones."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.l


class IA3In(nn.Module):
    """IA3: rescale the INPUT of the feed-forward down-projection (its hidden activations) with a learned vector."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


def inject_peft3(lm, cfg: RCfg) -> int:
    for p in lm.parameters():
        p.requires_grad = False
    names = N.PEFT_ATTN if cfg.peft_targets == "attn" else N.PEFT_ALL
    m = cfg.method

    def wrap(base):
        if m == "lora":
            return N.LoRALinear(base, cfg.peft_r, cfg.peft_alpha, cfg.peft_dropout)
        if m == "rslora":
            return RsLoRALinear(base, cfg.peft_r, cfg.peft_alpha, cfg.peft_dropout)
        return N.LoRMALinear(base, cfg.peft_r, cfg.peft_alpha, cfg.lorma_variant)

    n = 0
    for blk in lm.blocks:
        for part in (blk.mix, blk.ffn):
            for name, child in list(part.named_children()):
                if isinstance(child, nn.Linear):
                    if m == "ia3":
                        if name in ("k", "v"):
                            setattr(part, name, IA3Out(child))
                            n += 1
                        elif name == "w2":
                            setattr(part, name, IA3In(child))
                            n += 1
                    elif name in names:
                        setattr(part, name, wrap(child))
                        n += 1
                elif name == "experts" and cfg.peft_experts and m != "ia3":
                    for ex in child:
                        for en, ec in list(ex.named_children()):
                            if isinstance(ec, nn.Linear) and en in names:
                                setattr(ex, en, wrap(ec))
                                n += 1
    return n


def apply_method3(net, cfg: RCfg):
    m = cfg.method
    if m in ("full", "kd"):
        return
    if m == "partial":
        for p in net.lm.wte.parameters():
            p.requires_grad = False
        if net.lm.wpe is not None:
            for p in net.lm.wpe.parameters():
                p.requires_grad = False
        for blk in net.lm.blocks[:cfg.freeze_layers]:
            for p in blk.parameters():
                p.requires_grad = False
        return
    if m in ("lora", "rslora", "lorma", "ia3"):
        n = inject_peft3(net.lm, cfg)
        log(f"[peft] {m}{'-' + cfg.lorma_variant if m == 'lorma' else ''}: adapted {n} layers (r={cfg.peft_r}, targets={cfg.peft_targets})")
        return
    raise SystemExit(f"unknown method {m!r}: use full | partial | lora | rslora | lorma | ia3 | kd")


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


class RRNet(nn.Module):
    """Phase-1 backbone + sentence pooling + (BiGRU over sentences) + classifier."""

    def __init__(self, lm, n_classes, bidir, head, sent_dim, head_hidden, head_layers, head_dropout):
        super().__init__()
        self.lm, self.bidir, self.head_type = lm, bidir, head
        d = lm.mc["n_embd"]
        self.proj = nn.Linear(2 * d, sent_dim)
        self.pos_rel = nn.Embedding(10, sent_dim)
        self.pos_abs = nn.Embedding(7, sent_dim)
        self.dom = nn.Embedding(len(DOMAINS), sent_dim)
        self.norm = nn.LayerNorm(sent_dim)
        out_d = sent_dim
        if head == "bigru":
            self.rnn = nn.GRU(sent_dim, head_hidden, num_layers=head_layers, batch_first=True, bidirectional=True,
                              dropout=0.1 if head_layers > 1 else 0.0)
            out_d = 2 * head_hidden
        self.drop = nn.Dropout(head_dropout)
        self.cls = nn.Linear(out_d, n_classes)
        for e in (self.pos_rel, self.pos_abs, self.dom):
            nn.init.normal_(e.weight, 0.0, 0.02)
        nn.init.normal_(self.cls.weight, 0.0, 0.02)
        nn.init.zeros_(self.cls.bias)

    def sent_vectors(self, ids, km, pool, mark, sm, rel, ab, dom):
        """(B, M, sent_dim): one vector per sentence of every window."""
        h = backbone_hidden(self.lm, ids, km, self.bidir)                        # (B, L, d)
        mean = torch.bmm(pool.to(h.dtype), h)                                    # mean of the sentence's token states
        mk = torch.gather(h, 1, mark.unsqueeze(-1).expand(-1, -1, h.size(-1)))   # state at the sentence's <eos> marker
        s = self.proj(torch.cat([mean, mk], dim=-1))
        s = s + self.pos_rel(rel) + self.pos_abs(ab) + self.dom(dom).unsqueeze(1)
        return self.drop(F.gelu(self.norm(s)))

    def seq_logits(self, s, lengths):
        """(B, S, sent_dim) sentence sequences (padded; `lengths` = real lengths) -> (B, S, C) role logits."""
        if self.head_type == "bigru":
            M = s.size(1)
            pk = nn.utils.rnn.pack_padded_sequence(s, lengths.cpu(), batch_first=True, enforce_sorted=False)
            s, _ = self.rnn(pk)
            s, _ = nn.utils.rnn.pad_packed_sequence(s, batch_first=True, total_length=M)
        return self.cls(self.drop(s))

    def forward(self, ids, km, pool, mark, sm, rel, ab, dom):
        return self.seq_logits(self.sent_vectors(ids, km, pool, mark, sm, rel, ab, dom), sm.sum(1))


def build_net3(cfg: RCfg, lm, C):
    arch = lm.mc["arch"]
    bidir = (arch != "retnet") if cfg.bidir == "auto" else (cfg.bidir in ("true", "yes", "1", True))
    if arch == "retnet":
        bidir = False
    log(f"[model] arch={arch} bidirectional_attention={bidir} head={cfg.head} ctx_mode={cfg.ctx_mode}")
    return RRNet(lm, C, bidir, cfg.head, cfg.sent_dim, cfg.head_hidden, cfg.head_layers, cfg.head_dropout)


def load_teacher3(cfg: RCfg, dev, labels, tok_path):
    tr = Path(cfg.work_dir) / "rr_runs" / cfg.teacher_run
    if not (tr / "best.pt").exists():
        raise SystemExit(f"teacher run not found: {tr}/best.pt (train it first)")
    ck = torch.load(tr / "best.pt", map_location="cpu", weights_only=False)
    if ck["labels"] != labels:
        raise SystemExit("teacher and student use different label sets")
    if (tr / "tokenizer.json").read_bytes() != Path(tok_path).read_bytes():
        raise SystemExit("teacher and student use different tokenizers (distillation needs identical token ids)")
    saved = RCfg()
    apply_overrides(saved, {k: v for k, v in ck["cfg"].items() if k in {f.name for f in fields(RCfg)}})
    net = build_net3(saved, P1.LM(ck["mc"]), len(labels))
    apply_method3(net, saved)
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in net.parameters())
    log(f"[kd] teacher {cfg.teacher_run}: arch={ck['mc']['arch']} method={saved.method} params={n / 1e6:.1f}M")
    return net, n, ck["mc"]


def load_run3(cfg: RCfg, run_name: str, dev):
    from tokenizers import Tokenizer
    rd = Path(cfg.work_dir) / "rr_runs" / run_name
    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    saved = RCfg()
    apply_overrides(saved, {k: v for k, v in ck["cfg"].items() if k in {f.name for f in fields(RCfg)}})
    saved.work_dir = cfg.work_dir
    if cfg.local_data_dir:
        saved.local_data_dir = cfg.local_data_dir
    net = build_net3(saved, P1.LM(ck["mc"]), len(ck["labels"]))
    apply_method3(net, saved)
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    return {"rd": rd, "ck": ck, "saved": saved, "net": net, "labels": ck["labels"],
            "tok": Tokenizer.from_file(str(rd / "tokenizer.json"))}


# ============================================================================================
# 5. Inference + training
# ============================================================================================
@torch.no_grad()
def doc_probs3(net, docs, cfg, dev, amp, C, eos):
    net.eval()
    items = [(i, a, b) for i, d in enumerate(docs) for a, b in make_windows3(d, cfg)]
    order = sorted(range(len(items)), key=lambda k: int(docs[items[k][0]].lens[items[k][1]:items[k][2]].sum()))
    probs = [np.zeros((d.n, C), dtype=np.float32) for d in docs]
    wsum = [np.zeros(d.n, dtype=np.float32) for d in docs]
    for b0 in range(0, len(order), cfg.eval_batch):
        chunk = [items[k] for k in order[b0:b0 + cfg.eval_batch]]
        ids, km, pool, mark, sm, rel, ab, dom, _ = collate3(docs, chunk, dev, eos)
        with amp():
            logits = net(ids, km, pool, mark, sm, rel, ab, dom)
        pr = logits.float().softmax(-1).cpu().numpy()
        for j, (di, a, b) in enumerate(chunk):
            m = b - a
            w = np.minimum(np.arange(1, m + 1), np.arange(m, 0, -1)).astype(np.float32)   # centre sentences count more
            probs[di][a:b] += pr[j, :m] * w[:, None]
            wsum[di][a:b] += w
    return [p / np.maximum(w, 1e-6)[:, None] for p, w in zip(probs, wsum)]


def class_weights3(docs, C, mode):
    if mode == "none":
        return None
    cnt = np.bincount(np.concatenate([d.lab for d in docs]), minlength=C).astype(np.float64)
    return torch.tensor(np.clip(np.sqrt(cnt.max() / (cnt + 1.0)), 1.0, 6.0), dtype=torch.float32)


# ---- label-transition prior + calibrated decoding ---------------------------------------------------
def transition_logs(docs, C):
    """Log P(next role | role) and log P(first role) estimated (add-one smoothed) on the training documents."""
    T, S0 = np.ones((C, C)), np.ones(C)
    for d in docs:
        if d.n:
            S0[d.lab[0]] += 1
            np.add.at(T, (d.lab[:-1], d.lab[1:]), 1)
    return np.log(T / T.sum(1, keepdims=True)), np.log(S0 / S0.sum())


def viterbi_seq(lp, logT, logS, lam):
    """argmax over label paths of  sum_t lp[t, y_t] + lam * (logS[y_0] + sum_t logT[y_{t-1}, y_t])."""
    n, C = lp.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    score = lp[0] + lam * logS
    back = np.zeros((n, C), dtype=np.int16)
    ar = np.arange(C)
    for t in range(1, n):
        cand = score[:, None] + lam * logT
        idx = cand.argmax(0)
        back[t] = idx
        score = cand[idx, ar] + lp[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_all(docs, probs, bias, logT, logS, lam):
    out = []
    for p in probs:
        lp = np.log(np.clip(p, 1e-9, 1.0))
        if bias is not None:
            lp = lp + bias[None, :]
        out.append(viterbi_seq(lp, logT, logS, lam) if (lam > 0 and logT is not None) else lp.argmax(1))
    return out


def score_docs(docs, preds, C, domain=None):
    ys, ps = [], []
    for d, p in zip(docs, preds):
        if domain is None or d.domain == domain:
            ys.append(d.lab)
            ps.append(p)
    if not ys:
        return macro_scores(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), C)
    return macro_scores(np.concatenate(ys), np.concatenate(ps), C)


def calibrate(dv, probs, C, logT, logS, cfg):
    """Tune per-class thresholds, then the weight of the transition prior, on the dev set."""
    base = pooled(dv, probs, C)["macro_f1"]
    bias, f_b = tune_bias(dv, probs, C) if cfg.class_bias else (None, base)
    if bias is not None and f_b <= base + 0.002:
        bias = None
    cur = pooled(dv, probs, C, bias)["macro_f1"]
    lam = 0.0
    if cfg.trans_decode:
        for l_ in (0.1, 0.25, 0.5, 1.0):
            f = score_docs(dv, decode_all(dv, probs, bias, logT, logS, l_), C)["macro_f1"]
            log(f"[calibrate] transition weight {l_:.2f}: dev macro-F1 {f:.4f} (current {cur:.4f})")
            if f > cur + 0.003:
                lam, cur = l_, f
    return {"bias": bias, "lam": lam, "f1": cur, "f1_base": base}


# ---- whole-document sequence mode ("docseq") ---------------------------------------------------------
def pack_windows(d, a, b, max_len):
    """Partition sentences [a, b) into consecutive windows of <= max_len tokens (no overlap)."""
    cost, wins, i = d.lens + 1, [], a
    while i < b:
        j, tot = i, 0
        while j < b and (tot + cost[j] <= max_len or j == i):
            tot += cost[j]
            j += 1
        wins.append((i, j))
        i = j
    return wins


def docseq_items(docs, cfg, rng):
    """Training crops: runs of up to crop_windows consecutive windows of a document (random offset every epoch)."""
    k, items = max(1, cfg.crop_windows), []
    for di, d in enumerate(docs):
        if d.n == 0:
            continue
        wins = pack_windows(d, 0, d.n, cfg.max_len)
        if len(wins) <= k:
            items.append((di, wins))
            continue
        step = max(1, k // 2)
        starts = sorted(set([0] + list(range(rng.randrange(step), len(wins) - k + 1, step)) + [len(wins) - k]))
        items += [(di, wins[s0:s0 + k]) for s0 in starts]
    return items


def collate_docseq(docs, crops, dev, eos):
    """All windows of all crops go through the backbone together; fi maps (crop, sentence) -> row of the flattened sentence vectors."""
    win_items = [(di, a, b) for di, wins in crops for a, b in wins]
    wb = collate3(docs, win_items, dev, eos)[:8]
    M = wb[2].shape[1]
    rows, labs, j = [], [], 0
    for di, wins in crops:
        row, lb = [], []
        for a, b in wins:
            row += [j * M + m for m in range(b - a)]
            if docs[di].lab is not None:
                lb += docs[di].lab[a:b].tolist()
            j += 1
        rows.append(row)
        labs.append(lb)
    S = max(len(r) for r in rows)
    fi = np.zeros((len(crops), S), dtype=np.int64)
    smask = np.zeros((len(crops), S), dtype=bool)
    lab = np.full((len(crops), S), -100, dtype=np.int64)
    for c, (row, lb) in enumerate(zip(rows, labs)):
        fi[c, :len(row)] = row
        smask[c, :len(row)] = True
        if lb:
            lab[c, :len(lb)] = lb
    t = lambda x: torch.from_numpy(x).to(dev)
    return wb, t(fi), t(smask), t(lab)


@torch.no_grad()
def docseq_probs(net, docs, cfg, dev, amp, C, eos):
    """Backbone over the windows of a document (no grad), then ONE BiGRU pass over all its sentences."""
    net.eval()
    out = []
    for di, d in enumerate(docs):
        if d.n == 0:
            out.append(np.zeros((0, C), dtype=np.float32))
            continue
        wins, vecs = pack_windows(d, 0, d.n, cfg.max_len), []
        for b0 in range(0, len(wins), cfg.eval_batch):
            chunk = [(di, a, b) for a, b in wins[b0:b0 + cfg.eval_batch]]
            with amp():
                s = net.sent_vectors(*collate3(docs, chunk, dev, eos)[:8])
            vecs += [s[j, :b - a].float() for j, (_, a, b) in enumerate(chunk)]
        with amp():
            lg = net.seq_logits(torch.cat(vecs, 0).unsqueeze(0), torch.tensor([d.n]))
        out.append(lg[0].float().softmax(-1).cpu().numpy())
    return out


def doc_probs_any(net, docs, cfg, dev, amp, C, eos):
    if cfg.ctx_mode == "docseq":
        return docseq_probs(net, docs, cfg, dev, amp, C, eos)
    return doc_probs3(net, docs, cfg, dev, amp, C, eos)


def epoch_batches(docs, cfg, rng, dev, eos, items_all):
    if cfg.ctx_mode == "docseq":
        crops = docseq_items(docs, cfg, rng)
        rng.shuffle(crops)
        for i in range(0, len(crops), cfg.batch_docs):
            wb, fi, smask, lab = collate_docseq(docs, crops[i:i + cfg.batch_docs], dev, eos)
            yield ("docseq", wb, fi, smask, lab)
    else:
        items = list(items_all)
        rng.shuffle(items)
        for i in range(0, len(items), cfg.batch_size):
            *args, lab = collate3(docs, items[i:i + cfg.batch_size], dev, eos)
            yield ("win", tuple(args), lab)


def logits_of(net, b):
    if b[0] == "docseq":
        _, wb, fi, smask, lab = b
        s = net.sent_vectors(*wb)
        return net.seq_logits(s.reshape(-1, s.size(-1))[fi], smask.sum(1)), lab
    return net(*b[1]), b[2]


class EMA:
    """Exponential moving average of the trainable weights: much more stable on small, noisy dev sets."""

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


def report_text(labels, sc):
    for i, l in enumerate(labels):
        log(f"   {l:<24} P={sc['P'][i]:.3f} R={sc['R'][i]:.3f} F1={sc['F1'][i]:.3f}  support={sc['support'][i]}")


def run_train3(cfg: RCfg):
    rd = run_dir3(cfg)
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
            raise SystemExit('method "kd" needs "teacher_run": the name of a finished RR run')
        lm, mc = N.make_student(lm, mc, cfg)
    data = load_all(cfg, tok)
    labels, eos, tr, dv = data["labels"], data["eos"], data["train"], data["dev"]
    C = len(labels)
    assert cfg.max_len <= mc["ctx"], "max_len exceeds the Phase-1 context length"
    log(f"[data] train {len(tr)} docs / {sum(d.n for d in tr)} sentences | dev {len(dv)} docs / {sum(d.n for d in dv)} sentences | {C} labels")
    teacher, teacher_params = None, 0
    if cfg.method == "kd":
        teacher, teacher_params, t_mc = load_teacher3(cfg, dev, labels, rd / "tokenizer.json")
        assert cfg.max_len <= t_mc["ctx"]

    net = build_net3(cfg, lm, C)
    apply_method3(net, cfg)
    net.to(dev)
    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"[model] total params {n_total / 1e6:.1f}M | trainable {n_train / 1e6:.3f}M | method={cfg.method}")

    peft = cfg.method in ("lora", "rslora", "lorma", "ia3")
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
    cw = class_weights3(tr, C, cfg.class_weight)
    cw = cw.to(dev) if cw is not None else None
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay > 0 else None
    logT, logS = transition_logs(tr, C)

    items_all = [(i, a, b) for i, d in enumerate(tr) for a, b in make_windows3(d, cfg)]
    if cfg.ctx_mode == "docseq":
        per_epoch = math.ceil(len(docseq_items(tr, cfg, random.Random(0))) / cfg.batch_docs)
    else:
        per_epoch = math.ceil(len(items_all) / cfg.batch_size)
    total_steps = max(1, cfg.epochs * per_epoch)
    log(f"[train] ctx_mode={cfg.ctx_mode}: {per_epoch} steps per epoch, {total_steps} steps in total, EMA {cfg.ema_decay}")
    rng = random.Random(cfg.seed)
    best, step, t_start, bad = {"f1": -1.0}, 0, time.time(), 0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logf = open(rd / "log.jsonl", "a")
    for ep in range(cfg.epochs):
        net.train()
        run_loss, n_b, t_ep = 0.0, 0, time.time()
        for b in epoch_batches(tr, cfg, rng, dev, eos, items_all):
            for g in opt.param_groups:
                g["lr"] = N.lr_at(step, total_steps, g["base"], cfg.warmup_frac)
            with amp():
                logits, lab = logits_of(net, b)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, _ = logits_of(teacher, b)
            loss = F.cross_entropy(logits.float().view(-1, C), lab.view(-1), weight=cw, ignore_index=-100,
                                   label_smoothing=cfg.label_smoothing)
            if teacher is not None:
                loss = cfg.kd_alpha * loss + (1.0 - cfg.kd_alpha) * N.kd_loss(logits, t_logits, lab, cfg.kd_temp)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(step)
            step += 1
            run_loss += float(loss)
            n_b += 1
            if step % 20 == 0:
                log(f"  ep {ep + 1} step {step}/{total_steps} loss {run_loss / n_b:.4f}")
        with (ema.swapped() if ema is not None else contextlib.nullcontext()):    # evaluate (and save) the averaged weights
            probs = doc_probs_any(net, dv, cfg, dev, amp, C, eos)
            sc = pooled(dv, probs, C)
            log(f"[epoch {ep + 1}/{cfg.epochs}] loss {run_loss / max(1, n_b):.4f} | dev macro-F1 {sc['macro_f1']:.4f} "
                f"(P {sc['macro_p']:.4f} R {sc['macro_r']:.4f}) | {time.time() - t_ep:.0f}s")
            logf.write(json.dumps({"epoch": ep + 1, "loss": run_loss / max(1, n_b), "dev_f1": sc["macro_f1"]}) + "\n")
            logf.flush()
            if sc["macro_f1"] > best["f1"]:
                best, bad = {"f1": sc["macro_f1"], "epoch": ep + 1}, 0
                torch.save({"net": net.state_dict(), "mc": mc, "labels": labels, "cfg": jd, "bias": None}, rd / "best.pt")
            else:
                bad += 1
        if bad >= cfg.patience:
            log(f"[train] early stopping: no dev improvement for {bad} epochs (best epoch {best['epoch']})")
            break
    logf.close()
    train_s = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0

    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"])
    if dev == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time()
    probs = doc_probs_any(net, dv, cfg, dev, amp, C, eos)
    if dev == "cuda":
        torch.cuda.synchronize()
    infer_sent_s = sum(d.n for d in dv) / max(1e-9, time.time() - t_inf)
    cal = calibrate(dv, probs, C, logT, logS, cfg)
    preds = decode_all(dv, probs, cal["bias"], logT, logS, cal["lam"])
    sc = score_docs(dv, preds, C)
    sc_cl, sc_it = score_docs(dv, preds, C, 0), score_docs(dv, preds, C, 1)
    ck.update(bias=None if cal["bias"] is None else [float(x) for x in cal["bias"]], lam=cal["lam"],
              logT=logT.tolist(), logS=logS.tolist())
    torch.save(ck, rd / "best.pt")
    log(f"[dev] macro-F1 {cal['f1_base']:.4f} (argmax) -> {sc['macro_f1']:.4f} (class thresholds: "
        f"{'used' if cal['bias'] is not None else 'not used'}, transition weight {cal['lam']}) | P {sc['macro_p']:.4f} R {sc['macro_r']:.4f}")
    log(f"[dev] per-domain macro-F1: CL {sc_cl['macro_f1']:.4f} | IT {sc_it['macro_f1']:.4f}")
    report_text(labels, sc)
    cm = sc["cm"].copy()
    np.fill_diagonal(cm, 0)
    top = sorted(((int(cm[i, j]), labels[i], labels[j]) for i in range(C) for j in range(C) if cm[i, j] > 0), reverse=True)[:12]
    log("[dev] most frequent confusions (count, true -> predicted):")
    for c_, t_, p_ in top:
        log(f"   {c_:>5}  {t_} -> {p_}")
    variant = cfg.lorma_variant if cfg.method == "lorma" else ""
    P1.jdump({"best_epoch": best["epoch"], "macro_f1": sc["macro_f1"], "macro_f1_no_bias": cal["f1_base"],
              "macro_p": sc["macro_p"], "macro_r": sc["macro_r"], "dev_cl_f1": sc_cl["macro_f1"], "dev_it_f1": sc_it["macro_f1"],
              "per_label": {l: {"P": sc["P"][i], "R": sc["R"][i], "F1": sc["F1"][i], "support": sc["support"][i]}
                            for i, l in enumerate(labels)},
              "confusions": [{"count": c_, "true": t_, "pred": p_} for c_, t_, p_ in top], "bias": ck["bias"], "lam": cal["lam"],
              "params_total": n_total, "params_trainable": n_train, "train_seconds": train_s, "peak_gpu_mb": peak,
              "infer_sentences_per_s": infer_sent_s, "p1_arch": mc["arch"], "method": cfg.method,
              "teacher_run": cfg.teacher_run, "teacher_params": teacher_params}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": mc["arch"], "method": cfg.method, "variant": variant, "ctx_mode": cfg.ctx_mode,
                  "head": cfg.head, "bidir": net.bidir, "params_total_M": round(n_total / 1e6, 2),
                  "params_trainable_M": round(n_train / 1e6, 3), "dev_macro_f1": round(sc["macro_f1"], 4),
                  "dev_macro_p": round(sc["macro_p"], 4), "dev_macro_r": round(sc["macro_r"], 4),
                  "dev_cl_f1": round(sc_cl["macro_f1"], 4), "dev_it_f1": round(sc_it["macro_f1"], 4),
                  "train_s": round(train_s), "peak_gpu_mb": round(peak), "infer_sent_s": round(infer_sent_s),
                  "teacher_run": cfg.teacher_run})
    if cfg.predict_after:
        run_predict3(cfg)


def write_submission3(cfg: RCfg, rd: Path, preds: dict, test_docs):
    """preds: doc id -> list of label strings. Exactly one label per sentence, original order."""
    for d in test_docs:
        assert len(preds[d.id]) == d.n, f"{d.id}: {len(preds[d.id])} predictions for {d.n} sentences"
    ids = [d.id for d in test_docs]
    if len(set(ids)) != len(ids):
        log("[predict] WARNING: duplicate document ids across CL_test / IT_test - the output format cannot represent them separately")
    rd.mkdir(parents=True, exist_ok=True)
    pj = rd / "prediction.json"
    P1.jdump({d.id: preds[d.id] for d in test_docs}, pj)
    zp = rd / "submission.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    hist = Path(cfg.work_dir) / "rr_submissions"
    hist.mkdir(parents=True, exist_ok=True)
    shutil.copy(zp, hist / f"{cfg.run}_{time.strftime('%m%d_%H%M%S')}.zip")
    log(f"[predict] wrote {pj} ({len(test_docs)} documents, {sum(d.n for d in test_docs)} sentences) and {zp}")
    return str(zp)


def run_predict3(cfg: RCfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, amp = P1.amp_setup(dev)
    r = load_run3(cfg, cfg.run, dev)
    saved, labels, ck = r["saved"], r["labels"], r["ck"]
    data = load_all(saved, r["tok"], with_test=True)
    C, eos = len(labels), data["eos"]
    bias = np.array(ck["bias"], dtype=np.float32) if ck.get("bias") else None
    lam = float(ck.get("lam", 0.0))
    logT = np.array(ck["logT"]) if ck.get("logT") else None
    logS = np.array(ck["logS"]) if ck.get("logS") else None
    log(f"[predict] ctx_mode={saved.ctx_mode}, class thresholds {'on' if bias is not None else 'off'}, transition weight {lam}")
    probs = doc_probs_any(r["net"], data["test"], saved, dev, amp, C, eos)
    idx = decode_all(data["test"], probs, bias, logT, logS, lam)
    preds = {d.id: [labels[i] for i in p] for d, p in zip(data["test"], idx)}
    return write_submission3(cfg, r["rd"], preds, data["test"])


# ============================================================================================
# 6. Ensemble (average sentence probabilities of finished runs)
# ============================================================================================
def _ens_candidates(cfg: RCfg):
    out = []
    for f in sorted((Path(cfg.work_dir) / "rr_runs").glob("*/best.pt")):
        n = f.parent.name
        if cfg.ens_prefix:
            if not n.startswith(cfg.ens_prefix):
                continue
        elif n.startswith(("selftest", "check")):
            continue
        out.append(n)
    return out


def run_ensemble3(cfg: RCfg):
    names = list(cfg.ens_runs) or (_ens_candidates(cfg) if cfg.ens_auto else [])
    if len(names) < 2:
        raise SystemExit('need >= 2 finished runs: set "ens_runs": [...] or "ens_auto": true')
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, amp = P1.amp_setup(dev)
    vp, ref, dv, logT, logS = {}, {}, None, None, None
    for n in names:
        r = load_run3(cfg, n, dev)
        if not ref:
            data = load_all(r["saved"], r["tok"])
            ref = {"labels": r["labels"], "tok": r["tok"].to_str(), "eos": data["eos"]}
            dv = data["dev"]
            logT, logS = transition_logs(data["train"], len(r["labels"]))
        if r["labels"] != ref["labels"] or r["tok"].to_str() != ref["tok"]:
            log(f"[ensemble] skipping {n}: different labels / tokenizer")
        else:
            vp[n] = doc_probs_any(r["net"], dv, r["saved"], dev, amp, len(ref["labels"]), ref["eos"])
        del r
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    labels, eos = ref["labels"], ref["eos"]
    C = len(labels)
    mean = lambda keys: [sum(vp[k][i] for k in keys) / len(keys) for i in range(len(dv))]
    single = {k: pooled(dv, vp[k], C)["macro_f1"] for k in vp}
    for k, v in sorted(single.items(), key=lambda kv: -kv[1]):
        log(f"[ensemble] single model {k:<28} dev macro-F1 {v:.4f}")
    if cfg.ens_runs or not cfg.ens_auto:
        chosen = list(vp)
    else:
        order = sorted(single, key=lambda k: -single[k])
        chosen, cur = [order[0]], single[order[0]]
        for k in order[1:]:
            if len(chosen) >= cfg.ens_max:
                break
            f = pooled(dv, mean(chosen + [k]), C)["macro_f1"]
            log(f"[ensemble] try adding {k:<28} -> {f:.4f} (current {cur:.4f})")
            if f > cur + 0.002:
                chosen.append(k)
                cur = f
    log(f"[ensemble] members: {chosen}")
    avg = mean(chosen)
    cal = calibrate(dv, avg, C, logT, logS, cfg)
    preds_dev = decode_all(dv, avg, cal["bias"], logT, logS, cal["lam"])
    sc = score_docs(dv, preds_dev, C)
    sc_cl, sc_it = score_docs(dv, preds_dev, C, 0), score_docs(dv, preds_dev, C, 1)
    log(f"[ensemble] dev macro-F1 {cal['f1_base']:.4f} (argmax) -> {sc['macro_f1']:.4f} (class thresholds "
        f"{'used' if cal['bias'] is not None else 'not used'}, transition weight {cal['lam']})")
    runs = [load_run3(cfg, n, dev) for n in chosen]
    data = load_all(runs[0]["saved"], runs[0]["tok"], with_test=True)
    plist = [doc_probs_any(r["net"], data["test"], r["saved"], dev, amp, C, eos) for r in runs]
    avg_t = [sum(p[i] for p in plist) / len(plist) for i in range(len(data["test"]))]
    idx = decode_all(data["test"], avg_t, cal["bias"], logT, logS, cal["lam"])
    preds = {d.id: [labels[i] for i in p] for d, p in zip(data["test"], idx)}
    rd = run_dir3(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    P1.jdump({"members": chosen, "single_model_f1": single, "macro_f1": sc["macro_f1"], "dev_cl_f1": sc_cl["macro_f1"],
              "dev_it_f1": sc_it["macro_f1"], "lam": cal["lam"],
              "bias": None if cal["bias"] is None else [float(x) for x in cal["bias"]]}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": "ensemble", "method": "ensemble", "dev_macro_f1": round(sc["macro_f1"], 4),
                  "dev_macro_p": round(sc["macro_p"], 4), "dev_macro_r": round(sc["macro_r"], 4),
                  "dev_cl_f1": round(sc_cl["macro_f1"], 4), "dev_it_f1": round(sc_it["macro_f1"], 4), "teacher_run": "+".join(chosen)})
    return write_submission3(cfg, rd, preds, data["test"])


# ============================================================================================
# 7. Data diagnostics
# ============================================================================================
def diagnose3(cfg: RCfg):
    tok, _, mc = N.load_p1(cfg)
    data = load_all(cfg, tok, with_test=True)
    labels = data["labels"]
    log(f"[check] {len(labels)} labels: {labels}")
    if sorted(labels) != sorted(OFFICIAL_LABELS):
        log("[check] WARNING: label_vocab.json differs from the official 13 labels listed in the task description")
    for split in ("train", "dev", "test"):
        docs = data[split]
        for di, dom in enumerate(DOMAINS):
            dd = [d for d in docs if d.domain == di]
            if not dd:
                continue
            ns = np.array([d.n for d in dd])
            tl = np.concatenate([d.lens for d in dd]) if len(dd) else np.zeros(1)
            nw = sum(len(make_windows3(d, cfg)) for d in dd)
            log(f"[check] {split:<5} {dom}: {len(dd):>4} docs | sentences/doc mean {ns.mean():.0f} max {ns.max()} | "
                f"tokens/sentence mean {tl.mean():.1f} p95 {np.percentile(tl, 95):.0f} max {tl.max()} | windows {nw}")
    for split in ("train", "dev"):
        y = np.concatenate([d.lab for d in data[split]])
        cnt = np.bincount(y, minlength=len(labels))
        log(f"[check] {split} label distribution: " + ", ".join(f"{l}={c} ({100 * c / cnt.sum():.1f}%)" for l, c in zip(labels, cnt)))
    ids = [d.id for d in data["test"]]
    cl_ids, it_ids = {d.id for d in data["test"] if d.domain == 0}, {d.id for d in data["test"] if d.domain == 1}
    log(f"[check] test ids: {len(ids)} total, overlap between CL and IT: {len(cl_ids & it_ids)}")
    longs = sum(1 for d in data["train"] for s in d.sents if len(s) >= cfg.max_sent_tokens)
    log(f"[check] sentences truncated at {cfg.max_sent_tokens} tokens in train: {longs}")


# ============================================================================================
# 8. Self-test
# ============================================================================================
def _synthetic(tmp, rng):
    role_words = {l: [f"{l.lower()}{i}" for i in range(6)] for l in OFFICIAL_LABELS}
    common = "the court of appeal held that under section as per judgment order petitioner respondent".split()

    def sent(l):
        return " ".join(rng.choice(role_words[l] + common) for _ in range(rng.randint(6, 16))) + " ."

    def doc():
        order = ["Fact"] * rng.randint(3, 6) + ["Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "PrecedentReliedUpon",
                                                "RatioOfTheDecision", "RulingByPresentCourt"]
        order += [rng.choice(["None", "Dissent", "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt"])]
        return [sent(l) for l in order], order

    os.makedirs(tmp, exist_ok=True)
    allsent = []
    for dom in DOMAINS:
        for split, n in (("train", 10), ("dev", 4), ("test", 4)):
            docs = []
            for i in range(n):
                s, l = doc()
                allsent += s
                rec = {"id": f"{dom}_{split}_{i}", "text": s}
                if split != "test":
                    rec["labels"] = l
                docs.append(rec)
            json.dump(docs, open(os.path.join(tmp, f"{dom}_{split}.json"), "w"))
    json.dump(OFFICIAL_LABELS, open(os.path.join(tmp, "label_vocab.json"), "w"))
    return allsent


def selftest() -> bool:
    import tempfile
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    ok = True
    rng = random.Random(0)
    tmp = tempfile.mkdtemp(prefix="cs781_rr_selftest_")
    dd = os.path.join(tmp, "data")
    allsent = _synthetic(dd, rng)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([" ".join(s.split()) for s in allsent], trainers.BpeTrainer(
        vocab_size=500, special_tokens=[P1.EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    V = P1.pad64(tok.get_vocab_size())

    log("=== 1. data, windows, metrics ===")
    cfg0 = apply_overrides(RCfg(), {"local_data_dir": dd, "max_len": 96})
    data = load_all(cfg0, tok, with_test=True)
    labels, eos, tr, dv = data["labels"], data["eos"], data["train"], data["dev"]
    good = len(tr) == 20 and len(dv) == 8 and len(data["test"]) == 8 and labels == OFFICIAL_LABELS
    cover = all(sorted({i for a, b in make_windows3(d, cfg0) for i in range(a, b)}) == list(range(d.n)) for d in tr)
    toklim = all(sum(int(d.lens[i]) + 1 for i in range(a, b)) <= 96 or b - a == 1 for d in tr for a, b in make_windows3(d, cfg0))
    ok &= good and cover and toklim
    log(f"  parsing {'PASS' if good else 'FAIL'} | every sentence covered by a window {'PASS' if cover else 'FAIL'} | window token budget {'PASS' if toklim else 'FAIL'}")
    yt = np.array([0, 1, 1, 2, 2, 2, 0])
    yp = np.array([0, 1, 2, 2, 2, 1, 0])
    sc = macro_scores(yt, yp, 4)
    tp = [2, 1, 2, 0]; fp = [0, 1, 1, 0]; fn = [0, 1, 1, 0]
    f_manual = np.mean([2 * t / (2 * t + a + b) if t else 0.0 for t, a, b in zip(tp, fp, fn)])
    good = abs(sc["macro_f1"] - f_manual) < 1e-9
    ok &= good
    log(f"  macro-F1 (absent class counts as 0, as in the official scorer) {'PASS' if good else 'FAIL'}")
    items = [(0, 0, 3), (1, 1, 6)]
    batch = collate3(tr, items, "cpu", eos)
    shapes_ok = batch[2].shape[:2] == (2, 5) and batch[0].shape[0] == 2 and int(batch[4].sum()) == 8
    ok &= shapes_ok
    log(f"  collate shapes {'PASS' if shapes_ok else 'FAIL'}")

    log("=== 1b. transition prior / Viterbi / document-sequence crops / EMA ===")
    logT, logS = transition_logs(tr, len(labels))
    g1 = np.allclose(np.exp(logT).sum(1), 1.0) and abs(float(np.exp(logS).sum()) - 1.0) < 1e-6
    lp = np.log(np.array([[.6, .4], [.45, .55], [.6, .4]]))
    sticky, st0 = np.log(np.array([[.9, .1], [.1, .9]])), np.log(np.array([.5, .5]))
    g2 = list(viterbi_seq(lp, sticky, st0, 0.0)) == [0, 1, 0] and list(viterbi_seq(lp, sticky, st0, 1.0)) == [0, 0, 0]
    g3 = all([i for a, b in pack_windows(d, 0, d.n, 96) for i in range(a, b)] == list(range(d.n)) for d in tr)
    crops = docseq_items(tr, apply_overrides(RCfg(), {"max_len": 96, "crop_windows": 2}), random.Random(0))
    g4 = all({i for c_di, wins in crops if c_di == di for a, b in wins for i in range(a, b)} == set(range(d.n))
             for di, d in enumerate(tr))
    _, fi, smask, lab2 = collate_docseq(tr, crops[:2], "cpu", eos)
    g5 = fi.shape == smask.shape == lab2.shape and int(smask.sum()) == sum(b - a for _, wins in crops[:2] for a, b in wins)
    lin = nn.Linear(2, 2)
    ema_ = EMA(lin, 0.5)
    w0 = lin.weight.detach().clone()
    with torch.no_grad():
        lin.weight.add_(1.0)
    ema_.update(10 ** 6)
    with ema_.swapped():
        inside = lin.weight.detach().clone()
    g6 = torch.allclose(inside, 0.5 * w0 + 0.5 * (w0 + 1)) and torch.allclose(lin.weight.detach(), w0 + 1)
    ok &= all([g1, g2, g3, g4, g5, g6])
    log(f"  transition matrices {'PASS' if g1 else 'FAIL'} | viterbi smoothing {'PASS' if g2 else 'FAIL'} | windows partition {'PASS' if g3 else 'FAIL'} | "
        f"crops cover every sentence {'PASS' if g4 else 'FAIL'} | docseq batch {'PASS' if g5 else 'FAIL'} | EMA {'PASS' if g6 else 'FAIL'}")

    log("=== 2. adapters are the IDENTITY at initialisation; only adapters + head are trainable ===")
    for arch in P1.ARCH_PRESETS:
        for meth, var in (("lora", "plus"), ("rslora", "plus"), ("ia3", "plus"), ("lorma", "plus"), ("lorma", "pi")):
            tag = f"{arch}/{meth}" + (f"-{var}" if meth == "lorma" else "")
            try:
                torch.manual_seed(0)
                c = apply_overrides(RCfg(), {"method": meth, "lorma_variant": var, "peft_r": 4, "peft_alpha": 8.0, "peft_dropout": 0.0})
                net = build_net3(c, P1.LM(N._tiny_mc(arch, V)), len(labels)).eval()
                b = collate3(tr, [(0, 0, 4), (1, 0, 3)], "cpu", eos)
                inp = b[:8]
                with torch.no_grad():
                    ref = net(*inp)
                apply_method3(net, c)
                net.eval()
                with torch.no_grad():
                    out = net(*inp)
                diff = (ref - out).abs().max().item()
                n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
                n_all = sum(p.numel() for p in net.parameters())
                net.train()
                net(*inp).float().pow(2).mean().backward()
                grads = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.parameters() if p.requires_grad)
                frozen = all(p.grad is None for p in net.parameters() if not p.requires_grad)
                good = diff < 1e-4 and n_tr < n_all and grads and frozen
                ok &= good
                log(f"  {tag:<18} |adapted - base| = {diff:.1e}  trainable {n_tr:,}/{n_all:,}  {'PASS' if good else 'FAIL'}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  {tag:<18} FAIL: {type(e).__name__}: {e}")
                ok = False

    log("=== 3. end-to-end: every architecture / method, then ensembles ===")
    p1dirs = {}
    for arch in P1.ARCH_PRESETS:
        p1 = os.path.join(tmp, f"p1_{arch}")
        os.makedirs(p1, exist_ok=True)
        torch.save({"model": P1.LM(N._tiny_mc(arch, V)).state_dict(), "mc": N._tiny_mc(arch, V), "state": {"step": 0}},
                   os.path.join(p1, "ckpt_best.pt"))
        tok.save(os.path.join(p1, "tokenizer.json"))
        p1dirs[arch] = p1
    base = {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "max_len": 96, "epochs": 2, "batch_size": 8,
            "eval_batch": 8, "lr": 1e-3, "peft_lr": 3e-3, "head_lr": 3e-3, "peft_r": 4, "peft_alpha": 8.0, "sent_dim": 32,
            "head_hidden": 16, "patience": 5, "ema_decay": 0.9, "crop_windows": 2, "batch_docs": 2}
    cases = [(f"full_{a}", a, {"method": "full"}) for a in P1.ARCH_PRESETS] + [
        ("lora", "llama", {"method": "lora"}), ("rslora", "gpt2", {"method": "rslora"}), ("ia3", "moe", {"method": "ia3"}),
        ("lorma_plus", "gpt2", {"method": "lorma", "lorma_variant": "plus"}),
        ("lorma_pi", "retnet", {"method": "lorma", "lorma_variant": "pi"}),
        ("sentence_only", "llama", {"method": "full", "ctx_mode": "sentence"}),
        ("window_mode", "llama", {"method": "full", "ctx_mode": "doc"}),
        ("kd", "gpt2", {"method": "kd", "teacher_run": "selftest_full_llama", "student_layers": 1})]
    for tag, arch, extra in cases:
        try:
            cfg = apply_overrides(RCfg(), {**base, **extra, "run": f"selftest_{tag}", "p1_dir": p1dirs[arch]})
            run_train3(cfg)
            with zipfile.ZipFile(run_dir3(cfg) / "submission.zip") as z:
                assert z.namelist() == ["prediction.json"]
                pr = json.loads(z.read("prediction.json"))
            tests = {d.id: d.n for d in data["test"]}
            assert set(pr) == set(tests), "document ids"
            assert all(len(pr[k]) == tests[k] for k in tests), "exactly one label per sentence"
            assert all(x in OFFICIAL_LABELS for v in pr.values() for x in v), "labels from the vocabulary"
            rep = P1.jload(run_dir3(cfg) / "dev_report.json")
            if extra["method"] in ("lora", "rslora", "ia3", "lorma"):
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
        for tag, ov in (("ens", {"ens_runs": ["selftest_full_llama", "selftest_full_gpt2"]}),
                        ("ens_auto", {"ens_auto": True, "ens_prefix": "selftest_"})):
            cfg = apply_overrides(RCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "run": f"selftest_{tag}", **ov})
            zp = run_ensemble3(cfg)
            with zipfile.ZipFile(zp) as z:
                pr = json.loads(z.read("prediction.json"))
            assert all(len(pr[d.id]) == d.n for d in data["test"])
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
    ap.add_argument("stage", choices=["selftest", "check", "train", "predict", "ensemble"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--cfg", default="", help="JSON string or JSON file with config overrides")
    args = ap.parse_args(argv)
    if args.stage == "selftest":
        return 0 if selftest() else 1
    cfg = RCfg()
    ov = {}
    if args.cfg:
        ov = json.loads(args.cfg) if args.cfg.strip().startswith("{") else P1.jload(args.cfg)
    if args.run:
        ov["run"] = args.run
    apply_overrides(cfg, ov)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    {"check": diagnose3, "train": run_train3, "predict": run_predict3, "ensemble": run_ensemble3}[args.stage](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
