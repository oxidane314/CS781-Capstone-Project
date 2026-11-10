#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS781 Phase 7 - Prior Case Retrieval (PCR) on top of YOUR Phase-1 language models.

Needs cs781_phase1.py, cs781_phase2_ner.py and cs781_phase4_cjpe.py in the same folder (model classes, adapters, distillation helper).

Model: a DUAL ENCODER (the same Phase-1 backbone encodes queries and candidates)
  case text -> BPE tokens -> chunks of max_len tokens (a few random chunks per case while training, up to eval_chunks spread over the case at inference)
  -> Phase-1 backbone (bidirectional attention; causal for retnet) -> chunk vector = proj([mean of token states ; closing <eos> state])
  -> document representation ("pool"):
       mean    mean of chunk vectors                     -> cosine similarity
       attn    learned attention pooling over chunks     -> cosine similarity
       maxsim  late interaction: every query chunk is matched with its best candidate chunk (and vice versa), averaged
  Training: InfoNCE with in-batch negatives + HARD negatives mined with BM25 from the split's own candidate pool (false negatives masked),
  or a pairwise triplet loss. Distillation: the student matches the teacher's score distribution over the same candidates.
  Ranking: neural score, optionally FUSED with a lexical BM25 score (z-normalised per query, weight tuned on dev). Output: 7 unique ids per query.

Adaptation methods (cfg "method")
  full | partial | bitfit | adapter | lora | lorma (LoRMA+ / LoRMAπ) | ia3 | kd (pruned student <- teacher scores)

Stages:  selftest | check | train | predict | ensemble
  python cs781_phase7_pcr.py check --run pc_check --cfg '{"p1_dir": ".../explore_llama"}'
  python cs781_phase7_pcr.py train --run pc_full_llama --cfg cfg_pc.json
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
K = 7


# ============================================================================================
# 0. Config
# ============================================================================================
@dataclass
class PCfg:
    run: str = "pc_full_v1"
    work_dir: str = P1.DEFAULT_WORK
    p1_dir: str = ""
    p1_ckpt: str = "ckpt_best.pt"
    hf_repo: str = "Exploration-Lab/CS-781-Capstone"
    local_data_dir: str = ""          # folder with train.json, train_candidates.json, dev.json, dev_candidates.json, test.json, test_candidates.json
    train_prefix: str = "Phase_7_PCR/Train"
    dev_prefix: str = "Phase_7_PCR/Dev"
    test_prefix: str = "Phase_7_PCR/Test"
    max_train_queries: int = 0
    # ---- document representation
    max_len: int = 512
    train_chunks: int = 2
    eval_chunks: int = 4
    train_select: str = "random"      # random | uniform | head | tail
    eval_select: str = "uniform"
    pool: str = "mean"                # mean | attn | maxsim
    dim: int = 256
    drop: float = 0.1
    bidir: str = "auto"
    # ---- method
    method: str = "full"              # full | partial | bitfit | adapter | lora | lorma | ia3 | kd
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
    # ---- objective
    loss: str = "infonce"             # infonce | triplet
    tau: float = 0.05
    margin: float = 0.2
    batch_q: int = 6                  # queries per step
    hard_negs: int = 2                # BM25 hard negatives per query
    hard_pool: int = 30               # hard negatives are sampled from the top-N lexical neighbours that are not relevant
    # ---- optimisation
    epochs: int = 6
    lr: float = 2e-4
    llrd: float = 0.85
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    ema_decay: float = 0.995
    time_budget_min: float = 45.0
    eval_every_steps: int = 300
    dev_eval_queries: int = 60
    dev_eval_extra: int = 600         # extra random candidates in the reduced dev pool used during training
    enc_docs: int = 8                 # documents per forward pass at inference
    patience: int = 5
    seed: int = 1337
    # ---- lexical fusion
    fuse: bool = True
    alpha_min: float = 0.2            # minimum weight of the neural (Phase-1) score in the fusion: the submission never degenerates to pure BM25
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    predict_after: bool = True
    # ---- ensemble stage
    ens_runs: list = field(default_factory=list)
    ens_auto: bool = False
    ens_prefix: str = ""
    ens_max: int = 6


def apply_overrides(cfg: PCfg, ov: dict) -> PCfg:
    names = {f.name for f in fields(PCfg)}
    for k, v in ov.items():
        if k not in names:
            raise SystemExit(f"Unknown config key {k!r}. Valid: {sorted(names)}")
        setattr(cfg, k, v)
    return cfg


def run_dir7(cfg: PCfg) -> Path:
    return Path(cfg.work_dir) / "pc_runs" / cfg.run


PC_FIELDS = ["time", "run", "p1_arch", "method", "pool", "loss", "params_total_M", "params_trainable_M", "dev_f1_neural", "dev_f1_bm25", "dev_f1",
             "dev_precision", "dev_recall", "alpha", "train_s", "peak_gpu_mb", "enc_docs_s", "teacher_run"]


def log_csv(cfg: PCfg, row: dict):
    p = Path(cfg.work_dir) / "pc_experiments.csv"
    if p.exists() and open(p).readline().strip().split(",") != PC_FIELDS:
        p.rename(p.with_suffix(".old.csv"))
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PC_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row})


# ============================================================================================
# 1. Data
# ============================================================================================
def _first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _text_of(v):
    t = v.get("text") if isinstance(v, dict) else v
    if t is None and isinstance(v, dict):
        t = _first(v, ("document", "content", "sentences"))
    if isinstance(t, list):
        t = " ".join(str(x) for x in t)
    return " ".join(str(t or "").split())


def parse_docs(raw):
    """-> (ordered ids, dict id -> text, dict id -> relevant list or None)"""
    items = [(str(_first(r, ("id", "case_id", "doc_id", "query_id"))), r) for r in raw] if isinstance(raw, list) else [(str(k), v) for k, v in raw.items()]
    ids, texts, rel = [], {}, {}
    for did, v in items:
        ids.append(did)
        texts[did] = _text_of(v)
        r = _first(v, ("relevant_candidates", "relevant", "gold")) if isinstance(v, dict) else None
        rel[did] = [str(x) for x in r if str(x).strip()] if isinstance(r, list) else None
    return ids, texts, rel


class PSplit:
    pass


def load_split7(cfg: PCfg, prefix: str, name: str, labelled: bool, tok, tok_path: str, max_q: int = 0):
    pq = P1.resolve_file(cfg, f"{prefix}/{name}.json")
    pc = P1.resolve_file(cfg, f"{prefix}/{name}_candidates.json")
    key = hashlib.md5(open(tok_path, "rb").read() + f"{os.path.getsize(pq)}|{os.path.getsize(pc)}|{max_q}".encode()).hexdigest()[:12]
    cache = Path(cfg.work_dir) / "cache" / f"pcr_{name}_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            s = pickle.load(f)
        log(f"[data] {name}: {len(s.qids)} queries, {len(s.pool_ids)} candidates (token cache)")
        return s
    qids, qtext, rel = parse_docs(P1.jload(pq))
    if max_q:
        qids = qids[:max_q]
    pool_ids, pool_text, _ = parse_docs(P1.jload(pc))
    assert tok.get_vocab_size() < 65536
    enc = lambda texts: [np.asarray(e.ids, dtype=np.uint16) for b in range(0, len(texts), 64) for e in tok.encode_batch(texts[b:b + 64])]
    s = PSplit()
    s.name, s.qids, s.pool_ids = name, qids, pool_ids
    s.qtext, s.pool_text = [qtext[q] for q in qids], [pool_text[c] for c in pool_ids]
    s.qtok, s.pool_tok = enc(s.qtext), enc(s.pool_text)
    s.rel = {q: (rel[q] if (labelled and rel[q] is not None) else []) for q in qids}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(s, f, protocol=4)
    log(f"[data] {name}: {len(qids)} queries, {len(pool_ids)} candidates tokenised")
    return s


# ============================================================================================
# 2. Lexical retrieval (BM25) + metrics + fusion
# ============================================================================================
class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        import scipy.sparse as sp
        self.vec = CountVectorizer(token_pattern=r"(?u)\b[a-z0-9]{2,}\b", lowercase=True, min_df=2, max_df=0.6, max_features=300000, dtype=np.float32)
        tf = self.vec.fit_transform(texts).tocoo()
        dl = np.asarray(tf.tocsr().sum(1)).ravel()
        avg = max(1.0, float(dl.mean()))
        df = np.asarray((tf.tocsr() > 0).sum(0)).ravel()
        n = tf.shape[0]
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        w = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / avg)) * self.idf[tf.col]
        self.W = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def score(self, qtexts):
        q = self.vec.transform(qtexts).tocsr().astype(np.float32)
        q.data = np.log1p(q.data)                                        # damp repeated query terms (queries are whole cases)
        return np.asarray((q @ self.W.T).todense(), dtype=np.float32)


def topk_ids(S, qids, pool_ids, k=K):
    """Top-k candidate ids per query from a score matrix; a candidate with the query's own id is never returned."""
    S = np.array(S, dtype=np.float32, copy=True)
    pos = {c: i for i, c in enumerate(pool_ids)}
    for r, q in enumerate(qids):
        if q in pos:
            S[r, pos[q]] = -np.inf
    top = np.argpartition(-S, min(k, S.shape[1] - 1), axis=1)[:, :k]
    out = {}
    for r, q in enumerate(qids):
        order = top[r][np.argsort(-S[r, top[r]])]
        out[q] = [pool_ids[i] for i in order]
    return out


def f1_at_k(gold, pred, k=K):
    """Micro-F1/precision/recall @k exactly as in the official evaluator (gold set is neither truncated nor padded)."""
    tp = fp = fn = 0
    for q, labels in gold.items():
        g, p = set(labels), set(pred[q][:k])
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def recall_at(gold, S, qids, pool_ids, k):
    pred = topk_ids(S, qids, pool_ids, k)
    tp = sum(len(set(gold[q]) & set(pred[q])) for q in qids)
    tot = sum(len(set(gold[q])) for q in qids)
    return tp / tot if tot else 0.0


def zrow(S):
    S = np.asarray(S, dtype=np.float32)
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)


def tune_alpha(Sn, Sb, gold, qids, pool_ids, a_min=0.0):
    """Weight of the neural score in  alpha * z(neural) + (1 - alpha) * z(BM25), chosen on dev by Micro-F1@7 among alpha >= a_min
    (ties are broken towards a balanced mixture)."""
    zn, zb, table = zrow(Sn), zrow(Sb), []
    grid = sorted({round(float(a_min), 2)} | {round(float(a), 2) for a in np.arange(0.1, 1.0001, 0.1) if a >= a_min - 1e-9})
    for a in grid:
        table.append((a, f1_at_k(gold, topk_ids(a * zn + (1 - a) * zb, qids, pool_ids))["f1"]))
    best = max(table, key=lambda t: (round(t[1], 6), -abs(t[0] - 0.5)))
    return best[0], table


def validate_pred(pred, qids, pool_ids):
    """The official evaluator's validation rules: every query, exactly 7 unique ids from the pool."""
    errs, pool = [], set(pool_ids)
    if set(pred) != set(qids):
        errs.append("query ids differ")
    for q, v in pred.items():
        if not isinstance(v, list) or len(v) != K or len(set(v)) != K or any((not isinstance(c, str)) or c not in pool for c in v):
            errs.append(f"bad prediction for {q}")
    return errs


# ============================================================================================
# 3. Methods: reuse Phase 4 + IA3
# ============================================================================================
class IA3Out(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.l


class IA3In(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


def apply_method7(net, cfg: PCfg):
    if cfg.method != "ia3":
        J.apply_method4(net, cfg)
        return
    for p in net.lm.parameters():
        p.requires_grad = False
    n = 0
    for blk in net.lm.blocks:
        for part in (blk.mix, blk.ffn):
            for name, child in list(part.named_children()):
                if isinstance(child, nn.Linear) and name in ("k", "v"):
                    setattr(part, name, IA3Out(child))
                    n += 1
                elif isinstance(child, nn.Linear) and name == "w2":
                    setattr(part, name, IA3In(child))
                    n += 1
    log(f"[ia3] learned scaling vectors on {n} projections")


# ============================================================================================
# 4. Dual encoder
# ============================================================================================
def pick_chunks(m, k, mode, rng):
    if m <= k:
        return list(range(m))
    if mode == "random":
        return sorted(rng.sample(range(m), k))
    if mode == "head":
        return list(range(k))
    if mode == "tail":
        return list(range(m - k, m))
    return sorted({int(x) for x in np.linspace(0, m - 1, k).round()})


def collate_docs7(toks, idxs, cfg: PCfg, dev, eos, n_chunks, mode, rng=None):
    chunks, owner, slot, rel, counts = [], [], [], [], []
    for b, di in enumerate(idxs):
        t = toks[di]
        bounds = J.chunk_bounds(len(t), cfg.max_len - 1, 0)
        sel = pick_chunks(len(bounds), n_chunks, mode, rng or random)
        for k_, j in enumerate(sel):
            s, e = bounds[j]
            chunks.append(t[s:e])
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
    t_ = lambda x: torch.from_numpy(x).to(dev)
    return t_(ids), t_(km), t_(np.array(rel)), t_(np.array(owner)), t_(np.array(slot)), t_(np.array(counts))


class PCRNet(nn.Module):
    def __init__(self, lm, bidir, pool, dim, drop):
        super().__init__()
        self.lm, self.bidir, self.pool = lm, bidir, pool
        d = lm.mc["n_embd"]
        self.proj = nn.Linear(2 * d, dim)
        self.pos = nn.Embedding(8, dim)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        self.att = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)

    def embed(self, ids, km, rel, owner, slot, counts):
        h = J.backbone_hidden(self.lm, ids, km, self.bidir)
        kf = km.unsqueeze(-1).to(h.dtype)
        mean = (h * kf).sum(1) / kf.sum(1).clamp(min=1.0)
        lastv = h[torch.arange(h.size(0), device=h.device), km.sum(1) - 1]
        c = self.drop(self.proj(torch.cat([mean, lastv], dim=-1)) + self.pos(rel))                  # (N, D)
        B, Cm = counts.numel(), int(counts.max())
        pad = c.new_zeros(B, Cm, c.size(-1)).index_put((owner, slot), c).float()
        mask = torch.arange(Cm, device=c.device).unsqueeze(0) < counts.unsqueeze(1)
        if self.pool == "maxsim":
            return {"pad": F.normalize(pad, dim=-1), "mask": mask}
        if self.pool == "mean":
            v = (pad * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).float()
        else:
            sc = self.att(pad).squeeze(-1).float()
            sc = sc.masked_fill(~mask, torch.finfo(torch.float32).min)
            weights = torch.softmax(sc, dim=-1)
            v = (weights.unsqueeze(-1) * pad.float()).sum(1)
        return {"vec": F.normalize(v, dim=-1)}


def sim_matrix(Q, D, pool):
    """Query representations x candidate representations -> (nq, nd) similarity (cosine / late-interaction)."""
    if pool != "maxsim":
        return Q["vec"] @ D["vec"].T
    s = torch.einsum("bid,mjd->bmij", Q["pad"], D["pad"])                                           # (B, M, Cq, Cd)
    qm, dm = Q["mask"].float(), D["mask"].float()
    s_q = s.masked_fill(~D["mask"][None, :, None, :], -2.0).max(3).values                           # best candidate chunk for every query chunk
    s_q = (s_q * qm[:, None, :]).sum(2) / qm.sum(1)[:, None]
    s_d = s.masked_fill(~Q["mask"][:, None, :, None], -2.0).max(2).values                           # best query chunk for every candidate chunk
    s_d = (s_d * dm[None, :, :]).sum(2) / dm.sum(1)[None, :]
    return 0.5 * (s_q + s_d)


def build_net7(cfg: PCfg, lm):
    arch = lm.mc["arch"]
    bidir = (arch != "retnet") if cfg.bidir == "auto" else (cfg.bidir in ("true", "yes", "1", True))
    if arch == "retnet":
        bidir = False
    log(f"[model] arch={arch} bidirectional_attention={bidir} pool={cfg.pool} loss={cfg.loss} chunks train/eval {cfg.train_chunks}/{cfg.eval_chunks}")
    return PCRNet(lm, bidir, cfg.pool, cfg.dim, cfg.drop)


def load_run7(cfg: PCfg, run_name: str, dev):
    from tokenizers import Tokenizer
    rd = Path(cfg.work_dir) / "pc_runs" / run_name
    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    saved = PCfg()
    apply_overrides(saved, {k: v for k, v in ck["cfg"].items() if k in {f.name for f in fields(PCfg)}})
    saved.work_dir = cfg.work_dir
    if cfg.local_data_dir:
        saved.local_data_dir = cfg.local_data_dir
    net = build_net7(saved, P1.LM(ck["mc"]))
    apply_method7(net, saved)
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    return {"rd": rd, "ck": ck, "saved": saved, "net": net, "tok": Tokenizer.from_file(str(rd / "tokenizer.json"))}


@torch.no_grad()
def encode_set(net, toks, cfg: PCfg, dev, amp, eos, idxs=None):
    """Representations of many documents (inference): vec (n, D) or padded chunk vectors (n, eval_chunks, D) + mask, on the CPU in fp16."""
    net.eval()
    idxs = list(range(len(toks))) if idxs is None else list(idxs)
    order = sorted(range(len(idxs)), key=lambda i: len(toks[idxs[i]]))
    key = "pad" if cfg.pool == "maxsim" else "vec"
    D = net.proj.out_features
    out = torch.zeros((len(idxs), cfg.eval_chunks, D) if key == "pad" else (len(idxs), D), dtype=torch.float16)
    mask = torch.zeros((len(idxs), cfg.eval_chunks), dtype=torch.bool) if key == "pad" else None
    for b0 in range(0, len(order), cfg.enc_docs):
        sel = order[b0:b0 + cfg.enc_docs]
        batch = collate_docs7(toks, [idxs[i] for i in sel], cfg, dev, eos, cfg.eval_chunks, cfg.eval_select)
        with amp():
            r = net.embed(*batch)
        if key == "pad":
            Cm = r["pad"].shape[1]
            out[sel, :Cm] = r["pad"].half().cpu()
            mask[sel, :Cm] = r["mask"].cpu()
        else:
            out[sel] = r["vec"].half().cpu()
    return {key: out, **({"mask": mask} if mask is not None else {})}


@torch.no_grad()
def score_all(Qr, Dr, pool, dev, bq=4):
    n = (Qr["pad"] if pool == "maxsim" else Qr["vec"]).shape[0]
    D = {k: v.to(dev).float() if v.dtype != torch.bool else v.to(dev) for k, v in Dr.items()}
    rows = []
    for i in range(0, n, bq if pool == "maxsim" else 512):
        Q = {k: (v[i:i + (bq if pool == "maxsim" else 512)].to(dev).float() if v.dtype != torch.bool else v[i:i + (bq if pool == "maxsim" else 512)].to(dev))
             for k, v in Qr.items()}
        rows.append(sim_matrix(Q, D, pool).float().cpu().numpy())
    return np.concatenate(rows, 0)


def neural_scores(net, split, cfg, dev, amp, eos, pool_idx=None, q_idx=None):
    Qr = encode_set(net, split.qtok, cfg, dev, amp, eos, q_idx)
    Dr = encode_set(net, split.pool_tok, cfg, dev, amp, eos, pool_idx)
    return score_all(Qr, Dr, cfg.pool, dev)


# ============================================================================================
# 5. Training
# ============================================================================================
def run_train7(cfg: PCfg):
    rd = run_dir7(cfg)
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
            raise SystemExit('method "kd" needs "teacher_run": the name of a finished retrieval run')
        lm, mc = N.make_student(lm, mc, cfg)
    eos = tok.token_to_id(P1.EOS)
    assert cfg.max_len <= mc["ctx"], "max_len exceeds the Phase-1 context length"
    tr = load_split7(cfg, cfg.train_prefix, "train", True, tok, tok_path, cfg.max_train_queries)
    dv = load_split7(cfg, cfg.dev_prefix, "dev", True, tok, tok_path)

    # ---- training queries, positives inside the pool, BM25 hard negatives
    pidx = {c: i for i, c in enumerate(tr.pool_ids)}
    pos = {q: [pidx[c] for c in tr.rel[q] if c in pidx] for q in tr.qids}
    train_q = [i for i, q in enumerate(tr.qids) if pos[q]]
    log(f"[data] {len(train_q)}/{len(tr.qids)} training queries have a relevant candidate in the pool | mean relevant {np.mean([len(pos[tr.qids[i]]) for i in train_q]):.2f}")
    t0 = time.time()
    bm_tr = BM25(tr.pool_text, cfg.bm25_k1, cfg.bm25_b)
    Sb_tr = bm_tr.score([tr.qtext[i] for i in train_q])
    hard = {}
    for r, i in enumerate(train_q):
        q = tr.qids[i]
        banned = set(pos[q]) | ({pidx[q]} if q in pidx else set())
        order = np.argsort(-Sb_tr[r])[:cfg.hard_pool + len(banned)]
        hard[i] = [int(j) for j in order if int(j) not in banned][:cfg.hard_pool]
    del Sb_tr
    bm_dv = BM25(dv.pool_text, cfg.bm25_k1, cfg.bm25_b)
    log(f"[data] BM25 indexes + hard negatives ready ({time.time() - t0:.0f}s)")

    teacher = None
    teacher_params = 0
    if cfg.method == "kd":
        trd = Path(cfg.work_dir) / "pc_runs" / cfg.teacher_run
        if not (trd / "best.pt").exists():
            raise SystemExit(f"teacher run not found: {trd}/best.pt (train it first)")
        if (trd / "tokenizer.json").read_bytes() != (rd / "tokenizer.json").read_bytes():
            raise SystemExit("teacher and student use different tokenizers")
        tr_ = load_run7(cfg, cfg.teacher_run, dev)
        teacher = tr_["net"]
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_params = sum(p.numel() for p in teacher.parameters())
        log(f"[kd] teacher {cfg.teacher_run}: arch={tr_['ck']['mc']['arch']} method={tr_['saved'].method} pool={tr_['saved'].pool} params={teacher_params / 1e6:.1f}M")

    net = build_net7(cfg, lm)
    apply_method7(net, cfg)
    net.to(dev)
    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"[model] total params {n_total / 1e6:.1f}M | trainable {n_train / 1e6:.3f}M | method={cfg.method}")

    peft = cfg.method in ("lora", "lorma", "adapter", "bitfit", "ia3")
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
        base, wd = (body_lr * lr_scale(n_), body_wd if p_.dim() >= 2 else 0.0) if n_.startswith("lm.") else (cfg.head_lr, cfg.weight_decay if p_.dim() >= 2 else 0.0)
        buckets.setdefault((round(base, 12), wd), []).append(p_)
    opt = torch.optim.AdamW([{"params": ps, "lr": b_, "base": b_, "weight_decay": wd} for (b_, wd), ps in buckets.items()], betas=(0.9, 0.98), eps=1e-8)
    scaler = P1.make_scaler(dev == "cuda" and amp_dtype == torch.float16)
    ema = J.EMA(net, cfg.ema_decay) if cfg.ema_decay > 0 else None

    # ---- reduced dev pool for model selection during training
    rng = np.random.default_rng(cfg.seed)
    dq = [i for i, q in enumerate(dv.qids) if dv.rel[q]][:cfg.dev_eval_queries]
    dpidx = {c: i for i, c in enumerate(dv.pool_ids)}
    need = {dpidx[c] for i in dq for c in dv.rel[dv.qids[i]] if c in dpidx}
    extra = [int(i) for i in rng.permutation(len(dv.pool_ids))[:cfg.dev_eval_extra + len(need)] if int(i) not in need][:cfg.dev_eval_extra]
    sub_pool = sorted(need | set(extra))
    sub_ids = [dv.pool_ids[i] for i in sub_pool]
    sub_gold = {dv.qids[i]: dv.rel[dv.qids[i]] for i in dq}

    steps_per_epoch = max(1, math.ceil(len(train_q) / cfg.batch_q))
    total_steps = cfg.epochs * steps_per_epoch
    budget = cfg.time_budget_min * 60.0
    log(f"[train] {len(train_q)} queries -> {steps_per_epoch} steps/epoch x {cfg.epochs} epochs ({cfg.batch_q} queries + {cfg.hard_negs} hard negatives each per step); budget {cfg.time_budget_min} min")
    best, step, bad, t_start, stop = {"f1": -1.0}, 0, 0, time.time(), False
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logf = open(rd / "log.jsonl", "a")
    prng = random.Random(cfg.seed)

    def do_eval(tag, loss_val):
        nonlocal best, bad
        with (ema.swapped() if ema is not None else contextlib.nullcontext()):
            S = neural_scores(net, dv, cfg, dev, amp, eos, sub_pool, dq)
            m = f1_at_k(sub_gold, topk_ids(S, [dv.qids[i] for i in dq], sub_ids))
            log(f"[{tag}] loss {loss_val:.4f} | reduced dev pool ({len(dq)} queries, {len(sub_ids)} candidates): Micro-F1@7 {m['f1']:.4f} "
                f"(P {m['precision']:.4f} R {m['recall']:.4f}) | {time.time() - t_start:.0f}s")
            logf.write(json.dumps({"tag": tag, "step": step, "loss": loss_val, "dev_f1": m["f1"]}) + "\n")
            logf.flush()
            if m["f1"] > best["f1"]:
                best, bad = {"f1": m["f1"], "epoch": tag}, 0
                torch.save({"net": net.state_dict(), "mc": mc, "cfg": jd, "alpha": 1.0}, rd / "best.pt")
            else:
                bad += 1
        net.train()

    for ep in range(cfg.epochs):
        net.train()
        order = list(train_q)
        prng.shuffle(order)
        run_loss, n_b, early = 0.0, 0, False
        for g0 in range(0, len(order), cfg.batch_q):
            prog = max(step / total_steps, (time.time() - t_start) / budget if budget > 0 else 0.0)
            if prog >= 1.0:
                stop = True
                break
            for g in opt.param_groups:
                g["lr"] = J.lr_prog(g["base"], prog, cfg.warmup_frac)
            qs = order[g0:g0 + cfg.batch_q]
            docs, col, target = [], {}, []
            for i in qs:                                              # one random positive per query ...
                p = prng.choice(pos[tr.qids[i]])
                if p not in col:
                    col[p] = len(docs)
                    docs.append(p)
                target.append(col[p])
            for i in qs:                                              # ... plus BM25 hard negatives
                for h in prng.sample(hard[i], min(cfg.hard_negs, len(hard[i]))):
                    if h not in col:
                        col[h] = len(docs)
                        docs.append(h)
            relmask = torch.zeros((len(qs), len(docs)), dtype=torch.bool)
            for r, i in enumerate(qs):
                for p in pos[tr.qids[i]]:
                    if p in col and col[p] != target[r]:
                        relmask[r, col[p]] = True                     # other relevant documents are not negatives
            relmask = relmask.to(dev)
            tgt = torch.tensor(target, device=dev)
            bq = collate_docs7(tr.qtok, qs, cfg, dev, eos, cfg.train_chunks, cfg.train_select, prng)
            bd = collate_docs7(tr.pool_tok, docs, cfg, dev, eos, cfg.train_chunks, cfg.train_select, prng)       # one chunk sample for student AND teacher
            with amp():
                Q, D = net.embed(*bq), net.embed(*bd)
                if teacher is not None:
                    with torch.no_grad():
                        Qt, Dt = teacher.embed(*bq), teacher.embed(*bd)
            S = sim_matrix(Q, D, cfg.pool).float()
            if cfg.loss == "triplet":
                sp = S.gather(1, tgt[:, None])
                neg = S.masked_fill(relmask, -2.0)
                neg = neg.scatter(1, tgt[:, None], -2.0)
                loss = F.relu(cfg.margin - sp + neg).masked_fill(neg <= -2.0, 0.0).sum() / max(1, int((neg > -2.0).sum()))
            else:
                logits = (S / cfg.tau).masked_fill(relmask, -1e4)
                loss = F.cross_entropy(logits, tgt)
            if teacher is not None:
                with torch.no_grad():
                    St = sim_matrix(Qt, Dt, teacher.pool).float() / cfg.tau
                s_log = F.log_softmax((S / cfg.tau).masked_fill(relmask, -1e4) / cfg.kd_temp, dim=-1)
                t_p = F.softmax(St.masked_fill(relmask, -1e4) / cfg.kd_temp, dim=-1)
                kd = (t_p * (torch.log(t_p.clamp_min(1e-9)) - s_log)).sum(-1).mean() * cfg.kd_temp ** 2
                loss = cfg.kd_alpha * loss + (1.0 - cfg.kd_alpha) * kd
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(step)
            step += 1
            run_loss += float(loss.detach())
            n_b += 1
            if step % 20 == 0:
                log(f"  ep {ep + 1} step {step}/{total_steps} loss {run_loss / n_b:.4f} ({time.time() - t_start:.0f}s)")
            if cfg.eval_every_steps and step % cfg.eval_every_steps == 0:
                do_eval(f"step {step}", run_loss / max(1, n_b))
                if bad >= cfg.patience:
                    early = stop = True
                    break
        if not early:
            do_eval(f"epoch {ep + 1}/{cfg.epochs}", run_loss / max(1, n_b))
        if early or bad >= cfg.patience:
            log(f"[train] early stopping (best: {best['epoch']})")
            break
        if stop:
            log("[train] time budget reached")
            break
    logf.close()
    train_s = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0

    # ---- final evaluation on the FULL dev pool: neural, BM25, fused
    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"])
    if dev == "cuda":
        torch.cuda.synchronize()
    t_enc = time.time()
    Sn = neural_scores(net, dv, cfg, dev, amp, eos)
    if dev == "cuda":
        torch.cuda.synchronize()
    enc_docs_s = (len(dv.pool_ids) + len(dv.qids)) / max(1e-9, time.time() - t_enc)
    Sb = bm_dv.score(dv.qtext)
    gold = {q: dv.rel[q] for q in dv.qids}
    m_n = f1_at_k(gold, topk_ids(Sn, dv.qids, dv.pool_ids))
    m_b = f1_at_k(gold, topk_ids(Sb, dv.qids, dv.pool_ids))
    alpha, table = tune_alpha(Sn, Sb, gold, dv.qids, dv.pool_ids, cfg.alpha_min) if cfg.fuse else (1.0, [])
    m_f = f1_at_k(gold, topk_ids(alpha * zrow(Sn) + (1 - alpha) * zrow(Sb), dv.qids, dv.pool_ids)) if cfg.fuse else m_n
    ck["alpha"] = alpha
    torch.save(ck, rd / "best.pt")
    np.save(rd / "dev_scores.npy", Sn.astype(np.float16))
    np.save(rd / "dev_bm25.npy", Sb.astype(np.float16))
    P1.jdump({"qids": dv.qids, "pool": dv.pool_ids}, rd / "dev_ids.json")
    log(f"[dev] FULL pool ({len(dv.qids)} queries x {len(dv.pool_ids)} candidates): Micro-F1@7 neural {m_n['f1']:.4f} | BM25 {m_b['f1']:.4f} | "
        f"fused (alpha={alpha}) {m_f['f1']:.4f}  [P {m_f['precision']:.4f} R {m_f['recall']:.4f}]")
    if table:
        log("[dev] fusion weight alpha -> Micro-F1@7: " + ", ".join(f"{a}: {f:.3f}" for a, f in table))
    log(f"[dev] recall@7 {recall_at(gold, Sn, dv.qids, dv.pool_ids, 7):.4f} | recall@20 {recall_at(gold, Sn, dv.qids, dv.pool_ids, 20):.4f} | "
        f"recall@50 {recall_at(gold, Sn, dv.qids, dv.pool_ids, 50):.4f} (neural only: how many relevant cases are in the top-N at all)")
    variant = cfg.lorma_variant if cfg.method == "lorma" else ""
    P1.jdump({"best": best, "dev_neural": m_n, "dev_bm25": m_b, "dev_fused": m_f, "alpha": alpha, "alpha_table": table,
              "params_total": n_total, "params_trainable": n_train, "train_seconds": train_s, "peak_gpu_mb": peak, "enc_docs_per_s": enc_docs_s,
              "p1_arch": mc["arch"], "method": cfg.method, "teacher_run": cfg.teacher_run, "teacher_params": teacher_params}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": mc["arch"], "method": cfg.method + (f"-{variant}" if variant else ""), "pool": cfg.pool, "loss": cfg.loss,
                  "params_total_M": round(n_total / 1e6, 2), "params_trainable_M": round(n_train / 1e6, 3), "dev_f1_neural": round(m_n["f1"], 4),
                  "dev_f1_bm25": round(m_b["f1"], 4), "dev_f1": round(m_f["f1"], 4), "dev_precision": round(m_f["precision"], 4),
                  "dev_recall": round(m_f["recall"], 4), "alpha": alpha, "train_s": round(train_s), "peak_gpu_mb": round(peak),
                  "enc_docs_s": round(enc_docs_s, 1), "teacher_run": cfg.teacher_run})
    if cfg.predict_after:
        run_predict7(cfg)


def write_submission7(cfg: PCfg, rd: Path, pred: dict, qids, pool_ids):
    errs = validate_pred(pred, qids, pool_ids)
    assert not errs, errs[:3]
    rd.mkdir(parents=True, exist_ok=True)
    pj = rd / "prediction.json"
    P1.jdump({q: pred[q] for q in qids}, pj)
    zp = rd / "submission.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    hist = Path(cfg.work_dir) / "pc_submissions"
    hist.mkdir(parents=True, exist_ok=True)
    shutil.copy(zp, hist / f"{cfg.run}_{time.strftime('%m%d_%H%M%S')}.zip")
    log(f"[predict] wrote {pj} ({len(qids)} queries x {K} candidates) and {zp}")
    return str(zp)


def run_predict7(cfg: PCfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, amp = P1.amp_setup(dev)
    r = load_run7(cfg, cfg.run, dev)
    saved, ck = r["saved"], r["ck"]
    tp = str(r["rd"] / "tokenizer.json")
    eos = r["tok"].token_to_id(P1.EOS)
    te = load_split7(saved, saved.test_prefix, "test", False, r["tok"], tp)
    Sn = neural_scores(r["net"], te, saved, dev, amp, eos)
    Sb = BM25(te.pool_text, saved.bm25_k1, saved.bm25_b).score(te.qtext)
    np.save(r["rd"] / "test_scores.npy", Sn.astype(np.float16))
    np.save(r["rd"] / "test_bm25.npy", Sb.astype(np.float16))
    P1.jdump({"qids": te.qids, "pool": te.pool_ids}, r["rd"] / "test_ids.json")
    alpha = float(ck.get("alpha", 1.0))
    S = alpha * zrow(Sn) + (1 - alpha) * zrow(Sb) if saved.fuse else Sn
    log(f"[predict] {len(te.qids)} test queries x {len(te.pool_ids)} candidates, fusion alpha {alpha if saved.fuse else 1.0}")
    return write_submission7(cfg, r["rd"], topk_ids(S, te.qids, te.pool_ids), te.qids, te.pool_ids)


# ============================================================================================
# 6. Ensemble of runs (average z-normalised neural scores, then fuse with BM25)
# ============================================================================================
def run_ensemble7(cfg: PCfg):
    root = Path(cfg.work_dir) / "pc_runs"
    if cfg.ens_runs:
        names = list(cfg.ens_runs)
    elif cfg.ens_auto:
        names = [f.parent.name for f in sorted(root.glob("*/test_scores.npy")) if (f.parent / "dev_scores.npy").exists()
                 and (f.parent.name.startswith(cfg.ens_prefix) if cfg.ens_prefix else not f.parent.name.startswith(("selftest", "check")))]
    else:
        names = []
    if len(names) < 2:
        raise SystemExit('need >= 2 finished runs: set "ens_runs": [...] or "ens_auto": true')
    dids, tids = P1.jload(root / names[0] / "dev_ids.json"), P1.jload(root / names[0] / "test_ids.json")
    dev_bm, te_bm = np.load(root / names[0] / "dev_bm25.npy").astype(np.float32), np.load(root / names[0] / "test_bm25.npy").astype(np.float32)
    saved = PCfg()
    apply_overrides(saved, {k: v for k, v in P1.jload(root / names[0] / "config.json").items() if k in {f.name for f in fields(PCfg)}})
    saved.work_dir = cfg.work_dir
    if cfg.local_data_dir:
        saved.local_data_dir = cfg.local_data_dir
    pq = P1.resolve_file(saved, f"{saved.dev_prefix}/dev.json")
    _, _, rel = parse_docs(P1.jload(pq))
    gold = {q: [c for c in (rel[q] or [])] for q in dids["qids"]}
    dv_s, te_s = {}, {}
    for n in names:
        if P1.jload(root / n / "dev_ids.json") != dids or P1.jload(root / n / "test_ids.json") != tids:
            log(f"[ensemble] skipping {n}: different queries / candidate pool order")
            continue
        dv_s[n], te_s[n] = zrow(np.load(root / n / "dev_scores.npy").astype(np.float32)), zrow(np.load(root / n / "test_scores.npy").astype(np.float32))
    f1 = lambda S: f1_at_k(gold, topk_ids(S, dids["qids"], dids["pool"]))["f1"]
    single = {k: f1(v) for k, v in dv_s.items()}
    for k, v in sorted(single.items(), key=lambda kv: -kv[1]):
        log(f"[ensemble] single run {k:<28} neural-only dev Micro-F1@7 {v:.4f}")
    mean = lambda keys, d: sum(d[k] for k in keys) / len(keys)
    if cfg.ens_runs or not cfg.ens_auto:
        chosen = list(dv_s)
    else:
        order = sorted(single, key=lambda k: -single[k])
        chosen, cur = [order[0]], single[order[0]]
        for k in order[1:]:
            if len(chosen) >= cfg.ens_max:
                break
            f = f1(mean(chosen + [k], dv_s))
            log(f"[ensemble] try adding {k:<28} -> {f:.4f} (current {cur:.4f})")
            if f > cur + 0.001:
                chosen.append(k)
                cur = f
    log(f"[ensemble] members: {chosen}")
    Dn, Tn = mean(chosen, dv_s), mean(chosen, te_s)
    alpha, table = tune_alpha(Dn, dev_bm, gold, dids["qids"], dids["pool"], cfg.alpha_min) if cfg.fuse else (1.0, [])
    S_dev = alpha * zrow(Dn) + (1 - alpha) * zrow(dev_bm)
    m = f1_at_k(gold, topk_ids(S_dev, dids["qids"], dids["pool"]))
    log(f"[ensemble] dev Micro-F1@7 {m['f1']:.4f} (P {m['precision']:.4f} R {m['recall']:.4f}) with fusion alpha {alpha}")
    rd = run_dir7(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    P1.jdump({"members": chosen, "single_neural_f1": single, "alpha": alpha, "dev_fused": m}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": "ensemble", "method": "ensemble", "dev_f1": round(m["f1"], 4), "dev_precision": round(m["precision"], 4),
                  "dev_recall": round(m["recall"], 4), "alpha": alpha, "teacher_run": "+".join(chosen)})
    S_te = alpha * zrow(Tn) + (1 - alpha) * zrow(te_bm)
    return write_submission7(cfg, rd, topk_ids(S_te, tids["qids"], tids["pool"]), tids["qids"], tids["pool"])


# ============================================================================================
# 7. Diagnostics
# ============================================================================================
def diagnose7(cfg: PCfg):
    tok, _, mc = N.load_p1(cfg)
    tp = str(Path(cfg.p1_dir) / "tokenizer.json")
    sp = {"train": load_split7(cfg, cfg.train_prefix, "train", True, tok, tp, cfg.max_train_queries),
          "dev": load_split7(cfg, cfg.dev_prefix, "dev", True, tok, tp), "test": load_split7(cfg, cfg.test_prefix, "test", False, tok, tp)}
    for name, s in sp.items():
        qn, pn = np.array([len(t) for t in s.qtok]), np.array([len(t) for t in s.pool_tok])
        nch = np.array([len(J.chunk_bounds(len(t), cfg.max_len - 1, 0)) for t in s.pool_tok])
        line = (f"[check] {name:<5}: {len(s.qids)} queries x {len(s.pool_ids)} candidates | query tokens median {int(np.median(qn))} p90 {int(np.percentile(qn, 90))} | "
                f"candidate tokens median {int(np.median(pn))} p90 {int(np.percentile(pn, 90))} max {int(pn.max())} | candidate chunks median {int(np.median(nch))} "
                f"(eval_chunks={cfg.eval_chunks} covers {100 * np.mean(nch <= cfg.eval_chunks):.0f}% fully)")
        if name != "test":
            rl = np.array([len(s.rel[q]) for q in s.qids])
            pset = set(s.pool_ids)
            inpool = np.mean([c in pset for q in s.qids for c in s.rel[q]]) if rl.sum() else 0.0
            line += f" | relevant per query mean {rl.mean():.2f} max {rl.max()} (queries without any: {int((rl == 0).sum())}) | relevant ids found in the pool: {100 * inpool:.0f}%"
        log(line)
        log(f"[check] {name}: query ids that are also in the candidate pool: {len(set(s.qids) & set(s.pool_ids))}")
    dv = sp["dev"]
    gold = {q: dv.rel[q] for q in dv.qids}
    t0 = time.time()
    Sb = BM25(dv.pool_text, cfg.bm25_k1, cfg.bm25_b).score(dv.qtext)
    m = f1_at_k(gold, topk_ids(Sb, dv.qids, dv.pool_ids))
    log(f"[check] LEXICAL BASELINE (BM25, no learning) on dev: Micro-F1@7 {m['f1']:.4f} (P {m['precision']:.4f} R {m['recall']:.4f}) | "
        f"recall@7 {recall_at(gold, Sb, dv.qids, dv.pool_ids, 7):.4f} recall@20 {recall_at(gold, Sb, dv.qids, dv.pool_ids, 20):.4f} recall@50 {recall_at(gold, Sb, dv.qids, dv.pool_ids, 50):.4f} ({time.time() - t0:.0f}s)")
    log("[check] the Phase-1-based neural retriever has to beat or complement this number; the fused ranking (neural + BM25) is tuned on dev automatically")
    ex = np.mean([len(set(dv.rel[q]) & set(sp["train"].pool_ids)) for q in dv.qids])
    log(f"[check] relevant dev candidates that also occur in the TRAIN pool (would be leakage): {ex:.2f} per query")


# ============================================================================================
# 8. Self-test
# ============================================================================================
def _synthetic(tmp, rng):
    topics = [[f"topic{t}w{i}" for i in range(12)] for t in range(8)]
    filler = "the court held that the appeal petitioner respondent order section learned counsel submitted impugned judgment evidence".split()

    def doc(t):
        words = [rng.choice(filler) for _ in range(rng.randint(150, 500))]
        for _ in range(rng.randint(15, 30)):
            words.insert(rng.randint(0, len(words)), rng.choice(topics[t]))
        sents = []
        for i in range(0, len(words), 12):
            sents.append(" ".join(words[i:i + 12]) + ".")
        return sents

    os.makedirs(tmp, exist_ok=True)
    allt = []
    for split, nq, npool in (("train", 24, 60), ("dev", 10, 40), ("test", 8, 40)):
        pool, ptopic = {}, {}
        for i in range(npool):
            t = rng.randrange(8)
            pool[f"{split[0]}{i:05d}"] = {"text": doc(t)}
            ptopic[f"{split[0]}{i:05d}"] = t
        qs = {}
        for i in range(nq):
            t = rng.randrange(8)
            rel = [c for c, tt in ptopic.items() if tt == t]
            rng.shuffle(rel)
            q = {"text": doc(t)}
            if split != "test":
                q["relevant_candidates"] = rel[:rng.randint(1, 4)]
            qs[f"q{split[0]}{i:05d}"] = q
        json.dump(qs, open(os.path.join(tmp, f"{split}.json"), "w"))
        json.dump(pool, open(os.path.join(tmp, f"{split}_candidates.json"), "w"))
        allt += [" ".join(v["text"]) for v in list(qs.values()) + list(pool.values())]
    return allt


def selftest() -> bool:
    import tempfile
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    ok = True
    rng = random.Random(0)
    tmp = tempfile.mkdtemp(prefix="cs781_pc_selftest_")
    dd = os.path.join(tmp, "data")
    texts = _synthetic(dd, rng)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=[P1.EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    V = P1.pad64(tok.get_vocab_size())
    eos = tok.token_to_id(P1.EOS)
    p1x = os.path.join(tmp, "p1_x")
    os.makedirs(p1x, exist_ok=True)
    tok.save(os.path.join(p1x, "tokenizer.json"))

    log("=== 1. data, BM25, metrics (official definitions), fusion ===")
    cfg0 = apply_overrides(PCfg(), {"local_data_dir": dd, "work_dir": os.path.join(tmp, "work0"), "max_len": 64, "p1_dir": p1x})
    tr = load_split7(cfg0, cfg0.train_prefix, "train", True, tok, os.path.join(p1x, "tokenizer.json"))
    tr2 = load_split7(cfg0, cfg0.train_prefix, "train", True, tok, os.path.join(p1x, "tokenizer.json"))                 # from the cache
    dv = load_split7(cfg0, cfg0.dev_prefix, "dev", True, tok, os.path.join(p1x, "tokenizer.json"))
    g1 = len(tr.qids) == 24 and len(tr.pool_ids) == 60 and len(dv.pool_ids) == 40 and all(np.array_equal(a, b) for a, b in zip(tr.qtok, tr2.qtok))
    gold = {q: dv.rel[q] for q in dv.qids}
    Sb = BM25(dv.pool_text, 1.2, 0.75).score(dv.qtext)
    m_b = f1_at_k(gold, topk_ids(Sb, dv.qids, dv.pool_ids))
    g2 = m_b["recall"] > 0.9                                                  # topic words make lexical retrieval easy (F1@7 is capped by ~2 relevant per query)
    pred = topk_ids(Sb, dv.qids, dv.pool_ids)
    g3 = not validate_pred(pred, dv.qids, dv.pool_ids) and all(len(set(v)) == K for v in pred.values())
    gg = {"a": ["x", "y", "z"], "b": ["w"]}
    pp = {"a": ["x", "q", "r", "s", "t", "u", "v"], "b": ["w", "x", "y", "z", "q", "r", "s"]}
    r_ = f1_at_k(gg, pp)
    g4 = (r_["tp"], r_["fp"], r_["fn"]) == (2, 12, 2) and abs(r_["f1"] - 2 * 2 / (2 * 2 + 12 + 2)) < 1e-12
    S = np.random.default_rng(0).random((3, 12)).astype(np.float32)
    ids12 = [f"c{i}" for i in range(12)]
    S[0, 4] = 99.0
    tk = topk_ids(S, ["c4", "q1", "q2"], ids12)                              # query 0 is itself candidate c4 -> must not be returned
    g5 = "c4" not in tk["c4"] and len(tk["c4"]) == K and tk["q1"][0] != tk["q1"][1]
    a_, table = tune_alpha(np.random.default_rng(1).random((10, 40)), Sb, gold, dv.qids, dv.pool_ids)
    a2_, table2 = tune_alpha(np.random.default_rng(1).random((10, 40)), Sb, gold, dv.qids, dv.pool_ids, 0.2)
    g6 = 0.0 <= a_ <= 1.0 and len(table) == 11 and a2_ >= 0.2 and min(t[0] for t in table2) == 0.2
    bad = {q: v[:6] for q, v in pred.items()}
    g7 = bool(validate_pred(bad, dv.qids, dv.pool_ids))
    ok &= all([g1, g2, g3, g4, g5, g6, g7])
    log(f"  data + token cache {'PASS' if g1 else 'FAIL'} | BM25 finds the synthetic topics (recall@7 {m_b['recall']:.2f}) {'PASS' if g2 else 'FAIL'} | output validity {'PASS' if g3 else 'FAIL'} | "
        f"micro-F1@7 = evaluator definition {'PASS' if g4 else 'FAIL'} | query itself never retrieved {'PASS' if g5 else 'FAIL'} | fusion grid {'PASS' if g6 else 'FAIL'} | "
        f"validator rejects 6 candidates {'PASS' if g7 else 'FAIL'}")

    log("=== 2. similarity functions + adapters at identity / trainable subsets ===")
    for arch in P1.ARCH_PRESETS:
        for meth, var in (("adapter", "plus"), ("lora", "plus"), ("lorma", "plus"), ("lorma", "pi"), ("bitfit", "plus"), ("partial", "plus"), ("ia3", "plus")):
            tag = f"{arch}/{meth}" + (f"-{var}" if meth == "lorma" else "")
            try:
                torch.manual_seed(0)
                c = apply_overrides(PCfg(), {"method": meth, "lorma_variant": var, "peft_r": 4, "peft_alpha": 8.0, "peft_dropout": 0.0, "adapter_dim": 8,
                                             "freeze_layers": 1, "dim": 32, "max_len": 64})
                for pool in ("mean", "attn", "maxsim") if (arch == "llama" and meth == "adapter") else ("mean",):
                    c.pool = pool
                    net = build_net7(c, P1.LM(N._tiny_mc(arch, V))).eval()
                    Qb = collate_docs7(tr.qtok, [0, 1], c, "cpu", eos, 3, "uniform")
                    Db = collate_docs7(tr.pool_tok, [0, 1, 2], c, "cpu", eos, 3, "uniform")
                    with torch.no_grad():
                        s0 = sim_matrix(net.embed(*Qb), net.embed(*Db), pool)
                    assert s0.shape == (2, 3) and float(s0.abs().max()) <= 1.0001
                    if pool != "mean":
                        continue
                    apply_method7(net, c)
                    net.eval()
                    with torch.no_grad():
                        s1 = sim_matrix(net.embed(*Qb), net.embed(*Db), pool)
                    diff = (s0 - s1).abs().max().item()
                    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    n_all = sum(p.numel() for p in net.parameters())
                    net.train()
                    sim_matrix(net.embed(*Qb), net.embed(*Db), pool).sum().backward()
                    grads = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.parameters() if p.requires_grad)
                    good = diff < 1e-4 and n_tr < n_all and grads
                    ok &= good
                    log(f"  {tag:<18} |adapted - base| = {diff:.1e}  trainable {n_tr:,}/{n_all:,}  {'PASS' if good else 'FAIL'}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  {tag:<18} FAIL: {type(e).__name__}: {e}")
                ok = False

    log("=== 3. end-to-end: architectures / methods / pooling / losses, then ensemble ===")
    p1dirs = {}
    for arch in P1.ARCH_PRESETS:
        p1 = os.path.join(tmp, f"p1_{arch}")
        os.makedirs(p1, exist_ok=True)
        torch.save({"model": P1.LM(N._tiny_mc(arch, V)).state_dict(), "mc": N._tiny_mc(arch, V), "state": {"step": 0}}, os.path.join(p1, "ckpt_best.pt"))
        tok.save(os.path.join(p1, "tokenizer.json"))
        p1dirs[arch] = p1
    base = {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "max_len": 64, "train_chunks": 2, "eval_chunks": 3, "epochs": 2, "batch_q": 4,
            "hard_negs": 2, "hard_pool": 8, "lr": 1e-3, "peft_lr": 3e-3, "head_lr": 3e-3, "peft_r": 4, "peft_alpha": 8.0, "adapter_dim": 8, "dim": 32,
            "ema_decay": 0.9, "time_budget_min": 5, "eval_every_steps": 0, "dev_eval_queries": 6, "dev_eval_extra": 15, "enc_docs": 4, "patience": 5}
    cases = [(f"full_{a}", a, {"method": "full"}) for a in P1.ARCH_PRESETS] + [
        ("lorma_plus", "gpt2", {"method": "lorma", "lorma_variant": "plus"}), ("lorma_pi", "retnet", {"method": "lorma", "lorma_variant": "pi"}),
        ("lora", "llama", {"method": "lora"}), ("adapter", "moe", {"method": "adapter"}), ("bitfit", "gpt2", {"method": "bitfit"}),
        ("partial", "llama", {"method": "partial", "freeze_layers": 1}), ("ia3", "moe", {"method": "ia3"}),
        ("pool_attn", "llama", {"method": "full", "pool": "attn"}), ("pool_maxsim", "llama", {"method": "full", "pool": "maxsim"}),
        ("loss_triplet", "llama", {"method": "full", "loss": "triplet"}),
        ("kd", "gpt2", {"method": "kd", "teacher_run": "selftest_full_llama", "student_layers": 1})]
    te_ids = None
    for tag, arch, extra in cases:
        try:
            cfg = apply_overrides(PCfg(), {**base, **extra, "run": f"selftest_{tag}", "p1_dir": p1dirs[arch]})
            run_train7(cfg)
            with zipfile.ZipFile(run_dir7(cfg) / "submission.zip") as z:
                assert z.namelist() == ["prediction.json"]
                pr = json.loads(z.read("prediction.json"))
            ids_ = P1.jload(run_dir7(cfg) / "test_ids.json")
            assert not validate_pred(pr, ids_["qids"], ids_["pool"]), "official validation rules"
            rep = P1.jload(run_dir7(cfg) / "dev_report.json")
            if extra["method"] in ("lora", "lorma", "adapter", "bitfit", "partial", "ia3"):
                assert rep["params_trainable"] < rep["params_total"]
            if extra["method"] == "kd":
                assert rep["params_total"] < rep["teacher_params"]
            log(f"  e2e {tag:<13} PASS  (dev Micro-F1@7 neural {rep['dev_neural']['f1']:.3f}, BM25 {rep['dev_bm25']['f1']:.3f}, fused {rep['dev_fused']['f1']:.3f}; "
                f"trainable {rep['params_trainable']:,}/{rep['params_total']:,})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  e2e {tag:<13} FAIL: {type(e).__name__}: {e}")
            ok = False
    try:
        for tag, ov in (("ens", {"ens_runs": ["selftest_full_llama", "selftest_full_gpt2"]}), ("ens_auto", {"ens_auto": True, "ens_prefix": "selftest_full_"})):
            cfg = apply_overrides(PCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "run": f"selftest_{tag}", **ov})
            zp = run_ensemble7(cfg)
            with zipfile.ZipFile(zp) as z:
                pr = json.loads(z.read("prediction.json"))
            ids_ = P1.jload(run_dir7(apply_overrides(PCfg(), {"work_dir": os.path.join(tmp, "work"), "run": "selftest_full_llama"})) / "test_ids.json")
            assert not validate_pred(pr, ids_["qids"], ids_["pool"])
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
    cfg = PCfg()
    ov = {}
    if args.cfg:
        ov = json.loads(args.cfg) if args.cfg.strip().startswith("{") else P1.jload(args.cfg)
    if args.run:
        ov["run"] = args.run
    apply_overrides(cfg, ov)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    {"check": diagnose7, "train": run_train7, "predict": run_predict7, "ensemble": run_ensemble7}[args.stage](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
