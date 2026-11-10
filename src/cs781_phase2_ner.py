#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS781 Phase 2 - Legal NER on top of YOUR Phase-1 language models  [FINAL VERSION]
(four adaptation strategies: full fine-tuning | LoRA | LoRMA (multiplicative low-rank) | knowledge distillation)

Needs cs781_phase1.py in the same folder (it provides the model classes) and a Phase-1 run folder that
contains ckpt_best.pt + tokenizer.json (e.g. <work_dir>/runs/turbo_llama_v1).

Pipeline
  text --(collapse whitespace, keep char map)--> Phase-1 BPE tokens (+char offsets) --> sliding windows
       --> Phase-1 backbone (bidirectional attention) + token-classification head (BIO, 25 classes)
       --> overlap-averaged probabilities --> greedy / constrained-Viterbi BIO decoding
       --> token spans --> ORIGINAL character offsets --> prediction.json / submission.zip

Methods (cfg "method")
  full     fine-tune every weight of the Phase-1 backbone + head
  partial  freeze embeddings and the first N blocks
  lora     additive low-rank adapters:           W = W0 + (alpha/r) B A                       (Hu et al.)
  lorma    multiplicative low-rank adapters:     W = I(BA) W0   (LoRMA+ : (alpha/r BA + I) W0 ;  LoRMAπ : Iπ(BA) W0)   (Bihany et al. 2025)
  kd       knowledge distillation: a smaller student (layer-pruned Phase-1 model) learns from a fine-tuned teacher

What makes this version strong (all switchable in the config)
  * layout features: line-break structure + position-in-document buckets, injected through ZERO-initialised embeddings (the model
    starts exactly as the Phase-1 model, then learns to use them) - court preambles are line-structured, we collapse whitespace otherwise
  * layer-wise learning-rate decay (full fine-tuning), early stopping, class-balanced loss
  * decoding calibrated on validation: decoder, background-class threshold and (macro-F1!) per-class thresholds
  * ensemble stage that greedily picks the best subset of your finished runs

Stages
  recalibrate  re-tune the entity/background decision threshold + decoder of a FINISHED run on validation and re-predict (no retraining)
  ensemble     average the token probabilities of several finished runs (same tokenizer) and decode once
  selftest   offsets + adapter identity-at-init + tiny end-to-end run of every method on synthetic data (~3-6 min)
  check      REAL data diagnostics: gold format, end-offset convention, token-boundary alignment, oracle strict F1
  train      fine-tune, validate (strict macro-F1), error analysis, predict test, write submission.zip
  predict    re-predict the test set from a saved run (best.pt)

Examples
  python cs781_phase2_ner.py check  --run ner_full_llama --cfg '{"p1_dir": "/kaggle/working/cs781/runs/turbo_llama_v1"}'
  python cs781_phase2_ner.py train  --run ner_full_llama --cfg cfg_ner.json
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
import zlib
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cs781_phase1 as P1

log = P1.log
LABELS_DEFAULT = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]


# ============================================================================================
# 0. Config
# ============================================================================================
@dataclass
class NCfg:
    run: str = "ner_full_v1"
    work_dir: str = P1.DEFAULT_WORK
    p1_dir: str = ""                  # Phase-1 run dir (contains ckpt_best.pt, tokenizer.json)
    p1_ckpt: str = "ckpt_best.pt"
    hf_repo: str = "Exploration-Lab/CS-781-Capstone"
    local_data_dir: str = ""          # folder holding fold1-text.json, fold1-gold.json, ..., labels.txt, fold3-text.json
    train_prefix: str = "Phase_2_LNER/Train"
    test_file: str = "Phase_2_LNER/Test/fold3-text.json"
    # ---- data / context windows
    max_len: int = 512                # tokens per window (<= Phase-1 context)
    stride: int = 384                 # window step (overlap = max_len - stride)
    val_frac: float = 0.10
    split_seed: int = 1234
    neg_keep: float = 0.7             # fraction of entity-free windows kept per epoch (low values make the model over-predict entities)
    max_train_docs: int = 0           # debug only
    # ---- model
    method: str = "full"              # full | partial | lora | lorma | kd
    freeze_layers: int = 0            # partial: freeze embeddings + first N blocks
    # ---- parameter-efficient adapters (lora / lorma)
    peft_r: int = 16                  # rank
    peft_alpha: float = 32.0          # scale = alpha / r  (LoRA and LoRMA+; LoRMAπ uses scale 1 as in the paper)
    peft_dropout: float = 0.05        # LoRA only
    peft_lr: float = 5e-4             # learning rate of the adapter weights
    peft_targets: str = "all"         # attn (q,k,v,o + retention gate) | all (+ dense feed-forward linears)
    peft_experts: bool = False        # also adapt the MoE experts (off: only attention is adapted in MoE models)
    lorma_variant: str = "plus"       # plus (LoRMA+, additive rank inflation) | pi (LoRMAπ, permutation inflation)
    # ---- knowledge distillation
    teacher_run: str = ""             # name of a finished NER run (same labels + same tokenizer) used as teacher
    student_layers: int = 0           # student keeps this many evenly spaced Phase-1 blocks (0 = half)
    kd_alpha: float = 0.5             # loss = alpha * CE(hard labels) + (1 - alpha) * T^2 * KL(teacher || student)
    kd_temp: float = 2.0
    bidir: str = "auto"               # auto = bidirectional attention for gpt2/llama/moe, causal for retnet
    head: str = "auto"                # linear | bigru | auto (bigru when the backbone stays causal)
    head_dropout: float = 0.1
    head_hidden: int = 256            # BiGRU hidden size per direction (head = bigru)
    head_layers: int = 1              # BiGRU layers
    class_weight: str = "sqrt"        # none | sqrt  (up-weights rare entity classes)
    use_feats: bool = True            # line-break + document-position feature embeddings (zero-initialised)
    class_bias: bool = True           # tune a per-class decision threshold on validation (macro-F1 treats classes equally)
    # ---- optimisation
    epochs: int = 10
    patience: int = 4                 # early stopping: epochs without validation improvement
    batch_size: int = 16
    lr: float = 3e-4                  # backbone
    llrd: float = 0.85                # layer-wise lr decay (full / partial / kd); 1.0 = off
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    seed: int = 1337
    eval_batch: int = 32
    decode: str = "auto"              # greedy | viterbi | auto (best on validation)
    predict_after: bool = True
    ens_runs: list = field(default_factory=list)   # ensemble stage: names of finished runs whose probabilities are averaged
    ens_auto: bool = False            # ensemble stage: pick the best subset of ALL finished runs automatically
    ens_prefix: str = ""              # auto ensemble: only consider runs whose name starts with this
    ens_max: int = 6


def apply_overrides(cfg: NCfg, ov: dict) -> NCfg:
    names = {f.name for f in fields(NCfg)}
    for k, v in ov.items():
        if k not in names:
            raise SystemExit(f"Unknown config key {k!r}. Valid: {sorted(names)}")
        setattr(cfg, k, v)
    return cfg


NER_FIELDS = ["time", "run", "p1_arch", "method", "variant", "head", "bidir", "params_total_M", "params_trainable_M",
              "val_macro_f1", "val_macro_p", "val_macro_r", "decode", "train_s", "peak_gpu_mb", "infer_tok_s", "teacher_run"]


def run_dir(cfg: NCfg) -> Path:
    return Path(cfg.work_dir) / "ner_runs" / cfg.run


# ============================================================================================
# 1. Raw data loading (tolerant to the exact JSON layout) + end-offset convention
# ============================================================================================
def _doc_text(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("text", "document", "content"):
            if isinstance(v.get(k), str):
                return v[k]
    return ""


def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_gold_items(items):
    """-> list of (start, end, label, text_or_None)  (end exactly as stored in the file)"""
    if isinstance(items, dict):
        items = _first(items, ("entities", "annotations", "labels", "spans"), [])
    out = []
    for it in items or []:
        if isinstance(it, (list, tuple)):
            if len(it) < 3:
                continue
            s, e, l = it[0], it[1], it[2]
            t = it[3] if len(it) > 3 and isinstance(it[3], str) else None
        elif isinstance(it, dict):
            s = _first(it, ("start", "start_char", "start_offset", "begin", "start_pos"))
            e = _first(it, ("end", "end_char", "end_offset", "stop", "end_pos"))
            l = _first(it, ("label", "type", "entity", "tag", "entity_type", "category"))
            t = _first(it, ("text", "entity_text", "span", "value", "mention"))
            if s is None or e is None or l is None:
                continue
        else:
            continue
        out.append((int(s), int(e), str(l), t if isinstance(t, str) else None))
    return out


def _load_map(path):
    raw = P1.jload(path)
    if isinstance(raw, list):                       # [{"id":..,"text":..}, ...]
        raw = {str(_first(r, ("id", "doc_id", "document_id"), i)): r for i, r in enumerate(raw)}
    return {str(k): v for k, v in raw.items()}


def load_raw(cfg: NCfg):
    res = lambda rel: P1.resolve_file(cfg, rel)
    lp = res(f"{cfg.train_prefix}/labels.txt")
    labels = [x.strip() for x in open(lp, encoding="utf-8") if x.strip()] or LABELS_DEFAULT
    texts, gold = {}, {}
    for k in (1, 2):
        t = _load_map(res(f"{cfg.train_prefix}/fold{k}-text.json"))
        g = _load_map(res(f"{cfg.train_prefix}/fold{k}-gold.json"))
        for did, v in t.items():
            texts[did] = _doc_text(v)
            gold[did] = parse_gold_items(g.get(did, []))
    unknown = sorted({l for gs in gold.values() for _, _, l, _ in gs} - set(labels))
    if unknown:
        raise SystemExit(f"Gold labels not in labels.txt: {unknown}  (labels.txt: {labels})")
    return labels, texts, gold


def load_test(cfg: NCfg):
    t = _load_map(P1.resolve_file(cfg, cfg.test_file))
    return {k: _doc_text(v) for k, v in t.items()}


def detect_end_shift(texts, gold):
    """0 if gold end is exclusive (text[s:e] == entity text), 1 if inclusive."""
    ex = inc = n = 0
    for did, gs in gold.items():
        tx = texts[did]
        for s, e, _, t in gs:
            if t is None:
                continue
            n += 1
            ex += tx[s:e] == t
            inc += tx[s:e + 1] == t
    if n == 0:
        return 0, {"n_with_text": 0}
    shift = 1 if (inc > ex and ex < 0.5 * n) else 0
    return shift, {"n_with_text": n, "exclusive_match": ex / n, "inclusive_match": inc / n}


# ============================================================================================
# 2. Normalisation with a character map, tokenisation, BIO labels, windows
# ============================================================================================
_WORD = re.compile(r"\S+")


def normalize_with_feats(text: str):
    """Collapse whitespace runs (the Phase-1 model never saw newlines / double spaces) and keep
      n2o[i] : index of the original character of normalised character i
      nlb[i] : newlines (0, 1, 2+) in the whitespace BEFORE the word that starts at normalised char i (2 for the first word)
      nla[i] : newlines (0, 1, 2+) in the whitespace AFTER the word that ends at normalised char i (2 for the last word)
    The layout of a court preamble (party names on their own lines, 'Versus', judge names ...) is thus not lost."""
    words = list(_WORD.finditer(text))
    if not words:
        z = np.zeros(0, dtype=np.uint8)
        return "", np.zeros(0, dtype=np.int64), z, z
    nlw = np.empty(len(words) + 1, dtype=np.uint8)
    nlw[0], nlw[len(words)] = 2, 2
    for i in range(1, len(words)):
        nlw[i] = min(2, text.count("\n", words[i - 1].end(), words[i].start()))
    total = sum(m.end() - m.start() for m in words) + len(words) - 1
    nlb, nla = np.zeros(total, dtype=np.uint8), np.zeros(total, dtype=np.uint8)
    n2o, chars, pos = np.empty(total, dtype=np.int64), [], 0
    for i, m in enumerate(words):
        if i:
            chars.append(" ")
            n2o[pos] = m.start() - 1                    # separator -> a whitespace char before the word
            pos += 1
        w = m.group()
        n = len(w)
        chars.append(w)
        n2o[pos:pos + n] = np.arange(m.start(), m.end())
        nlb[pos] = nlw[i]
        nla[pos + n - 1] = nlw[i + 1]
        pos += n
    return "".join(chars), n2o, nlb, nla


def normalize_with_map(text: str):
    """(normalised text, char map) - see normalize_with_feats."""
    n, m, _, _ = normalize_with_feats(text)
    return n, m


POS_EDGES = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192])     # document-position buckets (in tokens), 9 buckets


class Doc:
    pass


def build_labels(n_tok, ts, te, ents, label2i, empty):
    lab = np.zeros(n_tok, dtype=np.int64)
    lab[empty] = -100
    mono = bool(np.all(np.diff(te) >= 0)) and bool(np.all(np.diff(ts) >= 0))
    unaligned = 0
    for cs, ce, l in sorted(ents, key=lambda x: -(x[1] - x[0])):         # long first, nested short ones overwrite
        if mono:
            i0 = int(np.searchsorted(te, cs, side="right"))
            i1 = int(np.searchsorted(ts, ce, side="left"))
            idx = [i for i in range(i0, i1) if not empty[i]]
        else:
            idx = np.nonzero((te > cs) & (ts < ce) & ~empty)[0].tolist()
        if not idx:
            unaligned += 1
            continue
        b = 1 + 2 * label2i[l]
        lab[idx[0]] = b
        lab[idx[1:]] = b + 1
    return lab, unaligned


def make_docs(tok, texts: dict, gold: dict | None, label2i: dict, end_shift: int, stats: dict):
    ids_order = list(texts)
    norms, maps, nlbs, nlas = [], [], [], []
    for d in ids_order:
        n, m, nb_, na_ = normalize_with_feats(texts[d])
        norms.append(n)
        maps.append(m)
        nlbs.append(nb_)
        nlas.append(na_)
    encs = tok.encode_batch([n if n else " " for n in norms])
    docs = []
    for did, norm, n2o, nlb, nla, enc in zip(ids_order, norms, maps, nlbs, nlas, encs):
        d = Doc()
        d.id, d.text, d.norm, d.n2o = did, texts[did], norm, n2o
        if not norm:
            enc_ids, off = [], np.zeros((0, 2), dtype=np.int64)
        else:
            enc_ids, off = enc.ids, np.asarray(enc.offsets, dtype=np.int64).reshape(-1, 2)
        d.ids = np.asarray(enc_ids, dtype=np.int64)
        ts, te = off[:, 0].copy(), off[:, 1].copy()
        if len(d.ids):
            nb = np.frombuffer(norm.encode("utf-32-le"), dtype=np.uint32)
            lead = (ts < te) & (nb[np.minimum(ts, len(nb) - 1)] == 32)    # byte-level tokens carry their leading space
            ts[lead] += 1
        d.ts, d.te = ts, te
        empty = te <= ts
        d.keep = np.nonzero(~empty)[0]
        d.feat = np.zeros(len(d.ids), dtype=np.int64)
        if len(d.ids):                                     # layout feature id = newlines-before * 3 + newlines-after
            d.feat = (nlb[np.minimum(ts, len(nlb) - 1)].astype(np.int64) * 3
                      + nla[np.clip(te - 1, 0, len(nla) - 1)].astype(np.int64))
            d.feat[empty] = 0
        d.gold, d.lab = None, None
        if gold is not None:
            ents_orig, ents_norm = [], []
            o2n = np.full(len(d.text), -1, dtype=np.int64)
            if len(n2o):
                nbn = np.frombuffer(norm.encode("utf-32-le"), dtype=np.uint32)
                nonsep = np.nonzero(nbn != 32)[0]
                o2n[n2o[nonsep]] = nonsep
            for s, e, l, _ in gold.get(did, []):
                e = e + end_shift
                stats["n_gold"] = stats.get("n_gold", 0) + 1
                if s < 0 or e > len(d.text) or e <= s:
                    stats["n_bad_offsets"] = stats.get("n_bad_offsets", 0) + 1
                    continue
                ents_orig.append((s, e, l))
                a, b = s, e
                while a < b and o2n[a] < 0:
                    a += 1
                while b > a and o2n[b - 1] < 0:
                    b -= 1
                if a != s or b != e:
                    stats["n_ws_edge"] = stats.get("n_ws_edge", 0) + 1
                if b <= a:
                    stats["n_empty_after_trim"] = stats.get("n_empty_after_trim", 0) + 1
                    continue
                ents_norm.append((int(o2n[a]), int(o2n[b - 1]) + 1, l))
            srt = sorted(ents_norm)
            stats["n_overlap_gold"] = stats.get("n_overlap_gold", 0) + sum(
                1 for x, y in zip(srt, srt[1:]) if y[0] < x[1])
            d.gold = set(ents_orig)
            d.lab, un = build_labels(len(d.ids), ts, te, ents_norm, label2i, empty)
            stats["n_unaligned"] = stats.get("n_unaligned", 0) + un
        docs.append(d)
    return docs


def make_windows(n, W, S):
    if n <= W:
        return [(0, n)] if n else []
    starts = list(range(0, n - W + 1, S))
    if starts[-1] + W < n:
        starts.append(n - W)
    return [(s, s + W) for s in starts]


# ============================================================================================
# 3. Decoding (greedy / constrained Viterbi) -> ORIGINAL character offsets
# ============================================================================================
def build_trans(L):
    C = 1 + 2 * L
    allowed = np.zeros((C, C), dtype=bool)
    for j in range(C):
        if j == 0 or (j - 1) % 2 == 0:
            allowed[:, j] = True                                  # O / B-x can follow anything
        else:
            lab = (j - 1) // 2
            allowed[1 + 2 * lab, j] = allowed[2 + 2 * lab, j] = True   # I-x only after B-x / I-x
    start = np.array([0.0 if (j == 0 or (j - 1) % 2 == 0) else -1e9 for j in range(C)])
    return np.where(allowed, 0.0, -1e9), start


def viterbi(logp, trans, start):
    T, C = logp.shape
    score = logp[0] + start
    back = np.zeros((T, C), dtype=np.int16)
    ar = np.arange(C)
    for t in range(1, T):
        cand = score[:, None] + trans
        idx = cand.argmax(0)
        back[t] = idx
        score = cand[idx, ar] + logp[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def path_to_spans(path, ts, te, n2o):
    spans, cur = [], None

    def close():
        if cur is not None:
            spans.append((int(n2o[cur[1]]), int(n2o[cur[2] - 1]) + 1, cur[0]))

    for i, p in enumerate(path):
        if p == 0:
            close()
            cur = None
            continue
        lab, is_b = (int(p) - 1) // 2, (int(p) - 1) % 2 == 0
        if cur is not None and not is_b and cur[0] == lab:
            cur[2] = int(te[i])
            continue
        close()
        cur = [lab, int(ts[i]), int(te[i])]
    close()
    return spans


def decode_doc(doc, probs, mode, trans, start, o_bias=0.0, cbias=None):
    """o_bias is added to the log-probability of the background class 'O': > 0 -> fewer / more confident entities
    (higher precision), < 0 -> more entities (higher recall). It is tuned on validation data."""
    k = doc.keep
    if len(k) == 0:
        return []
    lp = np.log(np.clip(probs[k], 1e-9, 1.0))
    lp[:, 0] += o_bias
    if cbias is not None:
        lp += cbias[None, :]
    path = viterbi(lp, trans, start) if mode == "viterbi" else lp.argmax(1)
    return path_to_spans(path, doc.ts[k], doc.te[k], doc.n2o)


# ============================================================================================
# 4. Strict metrics (same formulas as evaluate_phase2.py) + error analysis
# ============================================================================================
def strict_scores(gold: dict, pred: dict, labels):
    stats = {l: [0, 0, 0] for l in labels}
    for did, g in gold.items():
        p = pred.get(did, set())
        for l in labels:
            gs = {x for x in g if x[2] == l}
            ps = {x for x in p if x[2] == l}
            stats[l][0] += len(gs & ps)
            stats[l][1] += len(ps - gs)
            stats[l][2] += len(gs - ps)
    per, ps_, rs_, fs_ = {}, [], [], []
    for l, (tp, fp, fn) in stats.items():
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[l] = {"P": pr, "R": rc, "F1": f, "tp": tp, "fp": fp, "fn": fn}
        ps_.append(pr), rs_.append(rc), fs_.append(f)
    n = len(labels)
    return {"per_label": per, "macro_p": sum(ps_) / n, "macro_r": sum(rs_) / n, "macro_f1": sum(fs_) / n}


def present_f1(sc):
    """mean F1 over the labels that occur in the gold data (strict_scores' macro_f1 counts absent labels as 0)"""
    v = [x["F1"] for x in sc["per_label"].values() if x["tp"] + x["fn"] > 0]
    return sum(v) / max(1, len(v))


def analyze_errors(gold: dict, pred: dict):
    c = {"exact": 0, "boundary_error": 0, "label_confusion": 0, "overlap_other_label": 0, "spurious": 0, "missed": 0}
    for did, g in gold.items():
        p = pred.get(did, set())
        for x in p:
            if x in g:
                c["exact"] += 1
                continue
            ov = [y for y in g if y[0] < x[1] and x[0] < y[1]]
            if any(y[2] == x[2] for y in ov):
                c["boundary_error"] += 1
            elif any((y[0], y[1]) == (x[0], x[1]) for y in ov):
                c["label_confusion"] += 1
            elif ov:
                c["overlap_other_label"] += 1
            else:
                c["spurious"] += 1
        for y in g:
            if not any(x[0] < y[1] and y[0] < x[1] for x in p):
                c["missed"] += 1
    return c


def to_sets(spans_by_doc: dict, labels):
    return {d: {(s, e, labels[l]) for s, e, l in sp} for d, sp in spans_by_doc.items()}


# ============================================================================================
# 5. Model: Phase-1 backbone + token-classification head
# ============================================================================================
def load_p1(cfg: NCfg):
    from tokenizers import Tokenizer
    d = Path(cfg.p1_dir)
    ck_path = d / cfg.p1_ckpt
    if not ck_path.exists():
        raise SystemExit(f"Phase-1 checkpoint not found: {ck_path}  (set p1_dir to your Phase-1 run folder)")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    lm = P1.LM(ck["mc"])
    lm.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    log(f"[p1] loaded {ck_path} arch={ck['mc']['arch']} params={P1.count_params(lm) / 1e6:.1f}M")
    return tok, lm, ck["mc"]


class NERNet(nn.Module):
    def __init__(self, lm, n_classes, bidir, head, head_dropout, head_hidden=256, head_layers=1, use_feats=False):
        super().__init__()
        self.lm, self.bidir, self.head_type, self.use_feats = lm, bidir, head, use_feats
        d = lm.mc["n_embd"]
        if use_feats:                                   # zero-init: the network starts exactly as the Phase-1 model
            self.f_emb = nn.Embedding(9, d)
            self.p_emb = nn.Embedding(len(POS_EDGES) + 1, d)
            nn.init.zeros_(self.f_emb.weight)
            nn.init.zeros_(self.p_emb.weight)
        out_d = d
        if head == "bigru":
            self.rnn = nn.GRU(d, head_hidden, num_layers=head_layers, batch_first=True, bidirectional=True,
                              dropout=0.1 if head_layers > 1 else 0.0)
            out_d = 2 * head_hidden
        self.drop = nn.Dropout(head_dropout)
        self.cls = nn.Linear(out_d, n_classes)
        nn.init.normal_(self.cls.weight, 0.0, 0.02)
        nn.init.zeros_(self.cls.bias)

    def forward(self, ids, km, feat=None, apos=None):
        lm = self.lm
        B, T = ids.shape
        dev = ids.device
        pos = torch.arange(T, device=dev).unsqueeze(0)
        x = lm.wte(ids)
        if lm.wpe is not None:
            x = x + lm.wpe(pos)
        if self.use_feats and feat is not None:
            x = x + self.f_emb(feat) + self.p_emb(apos)
        rope = None
        if lm.rope_cos is not None:
            rope = (lm.rope_cos[pos].unsqueeze(1), lm.rope_sin[pos].unsqueeze(1))
        if self.bidir:
            amask = km[:, None, None, :]                                       # every token sees all real tokens
        else:
            qi, kj = torch.arange(T, device=dev).unsqueeze(1), torch.arange(T, device=dev).unsqueeze(0)
            amask = ((kj <= qi)[None, None] & km[:, None, None, :]) | (kj == qi)[None, None]
        for blk in lm.blocks:
            x = blk(x, rope, amask, km, None)
        h = lm.norm_f(x)
        if self.head_type == "bigru":
            lengths = km.sum(1).cpu()
            pk = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            h, _ = self.rnn(pk)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=T)
        return self.cls(self.drop(h))


def build_net(cfg: NCfg, lm, n_classes):
    arch = lm.mc["arch"]
    bidir = (arch != "retnet") if cfg.bidir == "auto" else (cfg.bidir in ("true", "yes", "1", True))
    if arch == "retnet":
        bidir = False                               # retention is inherently causal
    head = cfg.head if cfg.head != "auto" else ("linear" if bidir else "bigru")
    log(f"[model] arch={arch} bidirectional_attention={bidir} head={head}")
    return NERNet(lm, n_classes, bidir, head, cfg.head_dropout, cfg.head_hidden, cfg.head_layers, cfg.use_feats)


# --------------------------------------------------------------------------------------------
# Parameter-efficient adapters
# --------------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Additive low-rank adapter (LoRA):  y = W0 x + b + (alpha/r) * B A x ;  B = 0 at init so the model starts unchanged."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


_PI_IDX = {}


def _pi_index(d, device):
    """idx[i, j] = (j - i) mod d : gathering row i of a matrix with idx rotates that row to the right by i."""
    key = (d, str(device))
    if key not in _PI_IDX:
        i = torch.arange(d, device=device)
        _PI_IDX[key] = (i.unsqueeze(0) - i.unsqueeze(1)) % d
    return _PI_IDX[key]


class LoRMALinear(nn.Module):
    """Low-Rank Multiplicative Adaptation (LoRMA, Bihany et al. 2025): the frozen weight W0 (d x k) is multiplied
    from the left by a low-rank-parametrised d x d matrix, h = (I(BA) W0) x, with A in R^{r x d}, B in R^{d x r}.
      plus : I+(BA)  = (alpha/r) BA + I_d          init B = 0, A random      -> identity at start (LoRMA+)
      pi   : Ipi(BA) = rows of BA rotated by 0,1,2.. init B[:,0] = 1, A[0,0] = 1, rest of A = 0 -> identity at start (LoRMAπ)
    Operations are re-ordered as in the paper: W0 x is computed first, then the low-rank map acts on that d-dim vector,
    so the cost is comparable to LoRA. A bias (if any) is added after the multiplicative update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, variant: str):
        super().__init__()
        if variant not in ("plus", "pi"):
            raise SystemExit(f"lorma_variant must be 'plus' or 'pi', got {variant!r}")
        self.base, self.variant = base, variant
        d = base.out_features
        self.scale = alpha / r if variant == "plus" else 1.0
        self.A = nn.Parameter(torch.zeros(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        if variant == "plus":
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            nn.init.normal_(self.B, 0.0, 0.02)
            with torch.no_grad():
                self.B[:, 0] = 1.0
                self.A[0, 0] = 1.0

    def forward(self, x):
        y = F.linear(x, self.base.weight)                       # W0 x
        if self.variant == "plus":
            y = y + self.scale * F.linear(F.linear(y, self.A), self.B)       # (I + s B A) W0 x
        else:
            m = (self.B @ self.A).gather(1, _pi_index(self.A.shape[1], self.A.device))   # Ipi(BA)  (d x d)
            y = F.linear(y, m)                                  # Ipi(BA) W0 x
        if self.base.bias is not None:
            y = y + self.base.bias
        return y


PEFT_ALL = {"q", "k", "v", "o", "g", "fc", "w1", "w2", "w3"}
PEFT_ATTN = {"q", "k", "v", "o", "g"}


def inject_peft(lm, cfg: NCfg) -> int:
    """Freeze the whole Phase-1 backbone and wrap its linear layers with LoRA / LoRMA adapters. Returns #layers wrapped."""
    for p in lm.parameters():
        p.requires_grad = False
    names = PEFT_ATTN if cfg.peft_targets == "attn" else PEFT_ALL
    wrap = (lambda base: LoRALinear(base, cfg.peft_r, cfg.peft_alpha, cfg.peft_dropout)) if cfg.method == "lora" else \
           (lambda base: LoRMALinear(base, cfg.peft_r, cfg.peft_alpha, cfg.lorma_variant))
    n = 0
    for blk in lm.blocks:
        for part in (blk.mix, blk.ffn):
            for name, child in list(part.named_children()):
                if isinstance(child, nn.Linear) and name in names:
                    setattr(part, name, wrap(child))
                    n += 1
                elif name == "experts" and cfg.peft_experts:
                    for ex in child:
                        for en, ec in list(ex.named_children()):
                            if isinstance(ec, nn.Linear) and en in names:
                                setattr(ex, en, wrap(ec))
                                n += 1
    return n


def apply_method(net: NERNet, cfg: NCfg):
    """Prepare the network for the chosen adaptation method (must run before the optimizer is built)."""
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
    if m in ("lora", "lorma"):
        n = inject_peft(net.lm, cfg)
        log(f"[peft] {m}{'-' + cfg.lorma_variant if m == 'lorma' else ''}: wrapped {n} linear layers "
            f"(r={cfg.peft_r}, targets={cfg.peft_targets})")
        return
    raise SystemExit(f"unknown method {cfg.method!r} (use full | partial | lora | lorma | kd)")


# --------------------------------------------------------------------------------------------
# Knowledge distillation
# --------------------------------------------------------------------------------------------
def remap_student_keys(sd: dict, keep: list) -> dict:
    """Keep only the Phase-1 blocks listed in `keep` and renumber them 0..len(keep)-1."""
    out = {}
    for k, v in sd.items():
        if k.startswith("blocks."):
            _, i, rest = k.split(".", 2)
            if int(i) in keep:
                out[f"blocks.{keep.index(int(i))}.{rest}"] = v
        else:
            out[k] = v
    return out


def make_student(lm, mc: dict, cfg: NCfg):
    """Student = Phase-1 model with fewer (evenly spaced) blocks, initialised from the Phase-1 weights."""
    L = mc["n_layer"]
    n = cfg.student_layers or max(1, L // 2)
    if n >= L:
        raise SystemExit(f"student_layers ({n}) must be smaller than the Phase-1 depth ({L})")
    keep = [int(x) for x in np.linspace(0, L - 1, n).round()]
    mc2 = dict(mc)
    mc2["n_layer"] = n
    st = P1.LM(mc2)
    st.load_state_dict(remap_student_keys(lm.state_dict(), keep))
    log(f"[kd] student keeps Phase-1 blocks {keep} of {L}: {P1.count_params(lm) / 1e6:.1f}M -> {P1.count_params(st) / 1e6:.1f}M params")
    return st, mc2


def kd_loss(s_logits, t_logits, lab, T: float):
    """T^2 * KL(softmax(teacher/T) || softmax(student/T)) averaged over labelled (non-padding) tokens."""
    m = (lab != -100).float()
    sl = F.log_softmax(s_logits.float() / T, dim=-1)
    tp = F.softmax(t_logits.float() / T, dim=-1)
    kl = (tp * (torch.log(tp.clamp_min(1e-9)) - sl)).sum(-1)
    return (kl * m).sum() / m.sum().clamp_min(1.0) * (T * T)


def load_teacher(cfg: NCfg, dev, labels, tok_path):
    """Load a finished NER run as the (frozen) teacher; it must use the same labels and the same tokenizer."""
    tr = Path(cfg.work_dir) / "ner_runs" / cfg.teacher_run
    if not (tr / "best.pt").exists():
        raise SystemExit(f"teacher run not found: {tr}/best.pt (train it first)")
    ck = torch.load(tr / "best.pt", map_location="cpu", weights_only=False)
    if ck["labels"] != labels:
        raise SystemExit("teacher and student use different label sets")
    if (tr / "tokenizer.json").read_bytes() != Path(tok_path).read_bytes():
        raise SystemExit("teacher and student use different tokenizers (distillation needs identical token ids)")
    saved = NCfg()
    apply_overrides(saved, {k: v for k, v in ck["cfg"].items() if k in {f.name for f in fields(NCfg)}})
    net = build_net(saved, P1.LM(ck["mc"]), 1 + 2 * len(labels))
    apply_method(net, saved)
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in net.parameters())
    log(f"[kd] teacher {cfg.teacher_run}: arch={ck['mc']['arch']} method={saved.method} params={n / 1e6:.1f}M")
    return net, n, ck["mc"]


# ============================================================================================
# 6. Batching, inference over windows, training
# ============================================================================================
def collate(docs, items, dev):
    B = len(items)
    L = max(e - s for _, s, e in items)
    ids = np.zeros((B, L), dtype=np.int64)
    lab = np.full((B, L), -100, dtype=np.int64)
    km = np.zeros((B, L), dtype=bool)
    feat = np.zeros((B, L), dtype=np.int64)
    apos = np.zeros((B, L), dtype=np.int64)
    for j, (di, s, e) in enumerate(items):
        d, n = docs[di], e - s
        ids[j, :n] = d.ids[s:e]
        km[j, :n] = True
        feat[j, :n] = d.feat[s:e]
        apos[j, :n] = np.searchsorted(POS_EDGES, np.arange(s, e), side="right")   # position of each token in its DOCUMENT
        if d.lab is not None:
            lab[j, :n] = d.lab[s:e]
    t = lambda a: torch.from_numpy(a).to(dev)
    return t(ids), t(km), t(lab), t(feat), t(apos)


@torch.no_grad()
def doc_probs(net, docs, cfg, dev, amp, C):
    net.eval()
    items = [(i, s, e) for i, d in enumerate(docs) for s, e in make_windows(len(d.ids), cfg.max_len, cfg.stride)]
    order = sorted(range(len(items)), key=lambda k: items[k][2] - items[k][1])
    probs = [np.zeros((len(d.ids), C), dtype=np.float32) for d in docs]
    wsum = [np.zeros(len(d.ids), dtype=np.float32) for d in docs]
    for b in range(0, len(order), cfg.eval_batch):
        chunk = [items[k] for k in order[b:b + cfg.eval_batch]]
        ids, km, _, ft, ap = collate(docs, chunk, dev)
        with amp():
            logits = net(ids, km, ft, ap)
        pr = logits.float().softmax(-1).cpu().numpy()
        for j, (di, s, e) in enumerate(chunk):
            L = e - s
            w = np.minimum(np.arange(1, L + 1), np.arange(L, 0, -1)).astype(np.float32)   # centre tokens count more
            probs[di][s:e] += pr[j, :L] * w[:, None]
            wsum[di][s:e] += w
    return [p / np.maximum(w, 1e-6)[:, None] for p, w in zip(probs, wsum)]


def predict_docs(net, docs, cfg, dev, amp, C, mode, trans, start, o_bias=0.0, cbias=None, group=64):
    out = {}
    for g in range(0, len(docs), group):
        sub = docs[g:g + group]
        for d, p in zip(sub, doc_probs(net, sub, cfg, dev, amp, C)):
            out[d.id] = decode_doc(d, p, mode, trans, start, o_bias, cbias)
    return out


def tune_class_bias(va, probs, labels, trans, start, o_bias, gold, min_support=5, grid=(-1.0, -0.5, 0.5, 1.0), passes=2):
    """Coordinate ascent on one additive log-prob bias per entity class (applied to its B and I states), greedy decoding.
    Macro-F1 weighs every class equally, so rare classes profit from their own threshold. To limit over-fitting to a small
    validation set, only classes with >= min_support gold spans are tuned and a change must gain >= 0.005 macro-F1."""
    C = 1 + 2 * len(labels)
    support = {l: sum(1 for g in gold.values() for x in g if x[2] == l) for l in labels}

    def score(cb):
        pr = {d.id: decode_doc(d, p, "greedy", trans, start, o_bias, cb) for d, p in zip(va, probs)}
        return strict_scores(gold, to_sets(pr, labels), labels)["macro_f1"]

    cb = np.zeros(C, dtype=np.float32)
    cur = score(cb)
    for _ in range(passes):
        improved = False
        for li, l in enumerate(labels):
            if support[l] < min_support:
                continue
            best_b, best_s = 0.0, cur
            for b in grid:
                trial = cb.copy()
                trial[1 + 2 * li] += b
                trial[2 + 2 * li] += b
                sc = score(trial)
                if sc > best_s + 0.005:
                    best_s, best_b = sc, b
            if best_b != 0.0:
                cb[1 + 2 * li] += best_b
                cb[2 + 2 * li] += best_b
                cur, improved = best_s, True
        if not improved:
            break
    return cb, cur


O_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
O_GRID_VIT = [0.0, 1.0, 2.0, 3.0]


def tune_decoding(va, probs, cfg, labels, trans, start):
    """Pick (decoder, O-bias) with the best strict macro-F1 on the validation documents."""
    gold = eval_gold_sets(va)
    best, table = None, []
    for mode, grid in (("greedy", O_GRID), ("viterbi", O_GRID_VIT)):
        if cfg.decode in ("greedy", "viterbi") and cfg.decode != mode:
            continue
        for b in grid:
            pr = {d.id: decode_doc(d, p, mode, trans, start, b) for d, p in zip(va, probs)}
            sets = to_sets(pr, labels)
            sc = strict_scores(gold, sets, labels)
            table.append({"decode": mode, "o_bias": b, "macro_f1": sc["macro_f1"], "macro_p": sc["macro_p"],
                          "macro_r": sc["macro_r"]})
            log(f"[val] decode={mode:<8} o_bias={b:+.1f}  macro-F1 {sc['macro_f1']:.4f} (P {sc['macro_p']:.4f} R {sc['macro_r']:.4f})")
            if best is None or sc["macro_f1"] > best["sc"]["macro_f1"] + 1e-9:
                best = {"sc": sc, "sets": sets, "mode": mode, "o_bias": b}
    best.setdefault("cbias", None)
    if cfg.class_bias:
        g_best = max((t for t in table if t["decode"] == "greedy"), key=lambda t: t["macro_f1"], default=None)
        if g_best is not None:
            cb, f_cb = tune_class_bias(va, probs, labels, trans, start, g_best["o_bias"], gold)
            log(f"[val] per-class thresholds: macro-F1 {f_cb:.4f} (best global setting {best['sc']['macro_f1']:.4f}); "
                f"bias per class {[round(float(cb[1 + 2 * i]), 1) for i in range(len(labels))]}")
            if f_cb > best["sc"]["macro_f1"] + 0.005:
                pr = {d.id: decode_doc(d, p, "greedy", trans, start, g_best["o_bias"], cb) for d, p in zip(va, probs)}
                sets = to_sets(pr, labels)
                best = {"sc": strict_scores(gold, sets, labels), "sets": sets, "mode": "greedy",
                        "o_bias": g_best["o_bias"], "cbias": cb}
    return best, table


def error_examples(va, gold, pred, n=40):
    """A random sample of concrete mistakes (type, predicted text, gold text) to see what the model gets wrong."""
    byid = {d.id: d for d in va}
    rows = []
    for did, g in gold.items():
        text, p = byid[did].text, pred.get(did, set())
        for x in sorted(p):
            if x in g:
                continue
            ov = [y for y in g if y[0] < x[1] and x[0] < y[1]]
            same = [y for y in ov if y[2] == x[2]]
            if same:
                kind, y = "boundary_error", same[0]
            elif ov:
                kind, y = ("label_confusion" if (ov[0][0], ov[0][1]) == (x[0], x[1]) else "overlap_other_label"), ov[0]
            else:
                kind, y = "spurious", None
            rows.append({"doc": did, "type": kind, "pred": [x[0], x[1], x[2], text[x[0]:x[1]][:80]],
                         "gold": [y[0], y[1], y[2], text[y[0]:y[1]][:80]] if y else None})
        for y in g:
            if not any(x[0] < y[1] and y[0] < x[1] for x in p):
                rows.append({"doc": did, "type": "missed", "pred": None,
                             "gold": [y[0], y[1], y[2], text[y[0]:y[1]][:80]]})
    random.Random(0).shuffle(rows)
    return rows[:n]


def lr_at(step, total, base, warm):
    w = max(1, int(total * warm))
    return base * (step + 1) / w if step < w else base * max(0.0, (total - step) / max(1, total - w))


def class_weights(docs, C, mode):
    if mode == "none":
        return None
    lab = np.concatenate([d.lab[d.lab >= 0] for d in docs if d.lab is not None])
    cnt = np.bincount(lab, minlength=C).astype(np.float64)
    w = np.clip(np.sqrt(cnt.max() / (cnt + 1.0)), 1.0, 6.0)
    return torch.tensor(w, dtype=torch.float32)


def eval_gold_sets(docs):
    return {d.id: d.gold for d in docs}


# ============================================================================================
# 7. Experiment driver
# ============================================================================================
def prepare_data(cfg: NCfg, tok, labels):
    labels_, texts, gold = load_raw(cfg)
    labels = labels_
    label2i = {l: i for i, l in enumerate(labels)}
    end_shift, info = detect_end_shift(texts, gold)
    log(f"[data] gold end-offset convention: {'INCLUSIVE (+1 applied)' if end_shift else 'exclusive'}  {info}")
    ids = sorted(texts)
    if cfg.max_train_docs:
        ids = ids[:cfg.max_train_docs]
    is_val = lambda d: (zlib.crc32(f"{cfg.split_seed}:{d}".encode()) % 1000) < cfg.val_frac * 1000
    va_ids = [d for d in ids if is_val(d)] or ids[:1]
    tr_ids = [d for d in ids if d not in set(va_ids)] or ids
    stats = {}
    t0 = time.time()
    tr = make_docs(tok, {d: texts[d] for d in tr_ids}, gold, label2i, end_shift, stats)
    va = make_docs(tok, {d: texts[d] for d in va_ids}, gold, label2i, end_shift, stats)
    log(f"[data] {len(tr)} train docs, {len(va)} val docs, "
        f"{sum(len(d.ids) for d in tr):,} train tokens ({time.time() - t0:.0f}s)  stats={stats}")
    return labels, label2i, end_shift, tr, va, stats


def oracle(docs, labels, trans=None):
    """gold -> token labels -> decode -> ORIGINAL offsets. Measures what the tokenisation / alignment can ever reach."""
    pred = {}
    for d in docs:
        k = d.keep
        pred[d.id] = path_to_spans(d.lab[k], d.ts[k], d.te[k], d.n2o) if len(k) else []
    return strict_scores(eval_gold_sets(docs), to_sets(pred, labels), labels)


def diagnose(cfg: NCfg):
    tok, _, mc = load_p1(cfg)
    labels, label2i, end_shift, tr, va, stats = prepare_data(cfg, tok, None)
    docs = tr + va
    lens = np.array([len(d.ids) for d in docs])
    nwin = sum(len(make_windows(n, cfg.max_len, cfg.stride)) for n in lens)
    log(f"[check] tokens/doc: mean {lens.mean():.0f}  median {np.median(lens):.0f}  max {lens.max()}  "
        f"| windows of {cfg.max_len}: {nwin}")
    if mc["ctx"] < cfg.max_len:
        log(f"[check] WARNING: max_len {cfg.max_len} > Phase-1 context {mc['ctx']}")
    o = oracle(docs, labels)
    log(f"[check] ORACLE strict macro-F1 (upper bound of tokenisation + offset mapping): {o['macro_f1']:.4f} "
        f"(over labels present in the data: {present_f1(o):.4f})")
    cnt = {l: o["per_label"][l]["tp"] + o["per_label"][l]["fn"] for l in labels}
    for l in labels:
        pl = o["per_label"][l]
        log(f"   {l:<10} gold={cnt[l]:>6}  oracle-recoverable={pl['R']:.4f}")
    if present_f1(o) < 0.97:
        log("[check] WARNING: oracle < 0.97 -> many gold boundaries fall inside BPE tokens or have whitespace edges; "
            "look at n_ws_edge / n_unaligned in the stats above.")
    return o


def run_train(cfg: NCfg):
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    jd = asdict(cfg)
    P1.jdump(jd, rd / "config.json")
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype, amp = P1.amp_setup(dev)
    tok, lm, mc = load_p1(cfg)
    shutil.copy(Path(cfg.p1_dir) / "tokenizer.json", rd / "tokenizer.json")
    if cfg.method == "kd":
        if not cfg.teacher_run:
            raise SystemExit('method "kd" needs "teacher_run": the name of a finished NER run')
        lm, mc = make_student(lm, mc, cfg)
    labels, label2i, end_shift, tr, va, stats = prepare_data(cfg, tok, None)
    L = len(labels)
    C = 1 + 2 * L
    assert cfg.max_len <= mc["ctx"], "max_len exceeds the Phase-1 context length"
    teacher, teacher_params = None, 0
    if cfg.method == "kd":
        teacher, teacher_params, t_mc = load_teacher(cfg, dev, labels, rd / "tokenizer.json")
        assert cfg.max_len <= t_mc["ctx"], "max_len exceeds the teacher's context length"
    log(f"[oracle] strict macro-F1 upper bound on val docs: {oracle(va, labels)['macro_f1']:.4f}")

    net = build_net(cfg, lm, C)
    apply_method(net, cfg)
    net.to(dev)
    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"[model] total params {n_total / 1e6:.1f}M | trainable {n_train / 1e6:.2f}M | method={cfg.method}")

    peft = cfg.method in ("lora", "lorma")
    body_lr, body_wd = (cfg.peft_lr, 0.0) if peft else (cfg.lr, cfg.weight_decay)   # no weight decay on adapters (it would pull LoRMAπ away from identity)
    n_blocks = mc["n_layer"]

    def lr_scale(name):                       # layer-wise lr decay: lower layers (and embeddings) move less
        if peft or cfg.llrd >= 1.0 or not name.startswith("lm."):
            return 1.0
        m_ = re.match(r"lm\.blocks\.(\d+)\.", name)
        if m_:
            return cfg.llrd ** (n_blocks - 1 - int(m_.group(1)))
        return cfg.llrd ** n_blocks if name.startswith(("lm.wte", "lm.wpe")) else 1.0

    buckets = {}
    for n_, p_ in net.named_parameters():
        if not p_.requires_grad:
            continue
        if n_.startswith("lm."):
            base, wd = body_lr * lr_scale(n_), (body_wd if p_.dim() >= 2 else 0.0)
        else:                                  # NER head, BiGRU, feature embeddings
            base, wd = cfg.head_lr, (cfg.weight_decay if p_.dim() >= 2 else 0.0)
        buckets.setdefault((round(base, 12), wd), []).append(p_)
    groups = [{"params": ps, "lr": b_, "base": b_, "weight_decay": wd} for (b_, wd), ps in buckets.items()]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
    scaler = P1.make_scaler(dev == "cuda" and amp_dtype == torch.float16)
    cw = class_weights(tr, C, cfg.class_weight)
    cw = cw.to(dev) if cw is not None else None

    all_items = [(i, s, e) for i, d in enumerate(tr) for s, e in make_windows(len(d.ids), cfg.max_len, cfg.stride)]
    has_ent = [bool((tr[i].lab[s:e] > 0).any()) for i, s, e in all_items]
    pos_items = [it for it, h in zip(all_items, has_ent) if h]
    neg_items = [it for it, h in zip(all_items, has_ent) if not h]
    per_epoch = len(pos_items) + int(cfg.neg_keep * len(neg_items))
    total_steps = max(1, cfg.epochs * math.ceil(per_epoch / cfg.batch_size))
    log(f"[train] windows: {len(pos_items)} with entities, {len(neg_items)} without (keep {cfg.neg_keep}) "
        f"-> ~{per_epoch}/epoch, {total_steps} steps")
    trans, start = build_trans(L)
    rng = random.Random(cfg.seed)
    best, step, t_start, bad = {"f1": -1.0}, 0, time.time(), 0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logf = open(rd / "log.jsonl", "a")
    for ep in range(cfg.epochs):
        net.train()
        items = pos_items + [it for it in neg_items if rng.random() < cfg.neg_keep]
        rng.shuffle(items)
        run_loss, n_b, t_ep = 0.0, 0, time.time()
        for b in range(0, len(items), cfg.batch_size):
            ids, km, lab, ft, ap = collate(tr, items[b:b + cfg.batch_size], dev)
            if int((lab != -100).sum()) == 0:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, g["base"], cfg.warmup_frac)
            with amp():
                logits = net(ids, km, ft, ap)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(ids, km, ft, ap)
            loss = F.cross_entropy(logits.float().view(-1, C), lab.view(-1), weight=cw, ignore_index=-100)
            if teacher is not None:
                loss = cfg.kd_alpha * loss + (1.0 - cfg.kd_alpha) * kd_loss(logits, t_logits, lab, cfg.kd_temp)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += float(loss)
            n_b += 1
            if step % 20 == 0:
                log(f"  ep {ep + 1} step {step}/{total_steps} loss {run_loss / n_b:.4f}")
        pred = predict_docs(net, va, cfg, dev, amp, C, "greedy", trans, start)
        sc = strict_scores(eval_gold_sets(va), to_sets(pred, labels), labels)
        log(f"[epoch {ep + 1}/{cfg.epochs}] loss {run_loss / max(1, n_b):.4f} | val strict macro-F1 {sc['macro_f1']:.4f} "
            f"(P {sc['macro_p']:.4f} R {sc['macro_r']:.4f}) | {time.time() - t_ep:.0f}s")
        logf.write(json.dumps({"epoch": ep + 1, "loss": run_loss / max(1, n_b), "val_f1": sc["macro_f1"]}) + "\n")
        logf.flush()
        if sc["macro_f1"] > best["f1"]:
            best = {"f1": sc["macro_f1"], "epoch": ep + 1}
            torch.save({"net": net.state_dict(), "mc": mc, "labels": labels, "end_shift": end_shift,
                        "cfg": jd, "decode": "greedy"}, rd / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"[train] early stopping: no validation improvement for {bad} epochs (best epoch {best['epoch']})")
                break
    logf.close()
    train_s = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0

    # ---- reload best epoch, choose decoder, full analysis on validation
    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"])
    if dev == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time()
    probs = doc_probs(net, va, cfg, dev, amp, C)
    if dev == "cuda":
        torch.cuda.synchronize()
    infer_tok_s = sum(len(d.ids) for d in va) / max(1e-9, time.time() - t_inf)
    best_dec, table = tune_decoding(va, probs, cfg, labels, trans, start)
    sc, ps, mode, o_bias = best_dec["sc"], best_dec["sets"], best_dec["mode"], best_dec["o_bias"]
    err = analyze_errors(eval_gold_sets(va), ps)
    log(f"[val] CHOSEN decoder: {mode} with O-bias {o_bias:+.1f} | macro-F1 {sc['macro_f1']:.4f} "
        f"P {sc['macro_p']:.4f} R {sc['macro_r']:.4f}")
    for l in labels:
        pl = sc["per_label"][l]
        log(f"   {l:<10} P={pl['P']:.3f} R={pl['R']:.3f} F1={pl['F1']:.3f}  (tp {pl['tp']} fp {pl['fp']} fn {pl['fn']})")
    log(f"[val] error breakdown (predictions): {err}")
    ex = error_examples(va, eval_gold_sets(va), ps)
    P1.jdump(ex, rd / "error_examples.json")
    for e in ex[:8]:
        log(f"   [{e['type']}] pred={e['pred']}  gold={e['gold']}")
    ck["decode"], ck["o_bias"] = mode, o_bias
    ck["cbias"] = None if best_dec["cbias"] is None else [float(x) for x in best_dec["cbias"]]
    torch.save(ck, rd / "best.pt")
    variant = cfg.lorma_variant if cfg.method == "lorma" else ""
    P1.jdump({"best_epoch": best["epoch"], "decode": mode, "o_bias": o_bias, "cbias": ck["cbias"], "decode_table": table, "scores": sc, "errors": err, "stats": stats,
              "params_total": n_total, "params_trainable": n_train, "train_seconds": train_s,
              "peak_gpu_mb": peak, "p1_arch": mc["arch"], "method": cfg.method, "variant": variant,
              "infer_tokens_per_s": infer_tok_s, "teacher_run": cfg.teacher_run,
              "teacher_params": teacher_params}, rd / "val_report.json")
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": cfg.run, "p1_arch": mc["arch"], "method": cfg.method,
           "variant": variant, "head": net.head_type, "bidir": net.bidir, "params_total_M": round(n_total / 1e6, 2),
           "params_trainable_M": round(n_train / 1e6, 3), "val_macro_f1": round(sc["macro_f1"], 4),
           "val_macro_p": round(sc["macro_p"], 4), "val_macro_r": round(sc["macro_r"], 4), "decode": f"{mode}/o{o_bias:+.1f}",
           "train_s": round(train_s), "peak_gpu_mb": round(peak), "infer_tok_s": round(infer_tok_s),
           "teacher_run": cfg.teacher_run}
    p = Path(cfg.work_dir) / "ner_experiments.csv"
    if p.exists() and open(p).readline().strip().split(",") != NER_FIELDS:
        p.rename(p.with_suffix(".old.csv"))                 # header of an older version: keep it, start a new table
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NER_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    if cfg.predict_after:
        run_predict(cfg)


def load_run(cfg: NCfg, run_name: str, dev):
    """Rebuild a finished NER run (network, saved config, tokenizer, labels) from its best.pt."""
    from tokenizers import Tokenizer
    rd = Path(cfg.work_dir) / "ner_runs" / run_name
    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    saved = NCfg()
    apply_overrides(saved, {k: v for k, v in ck["cfg"].items() if k in {f.name for f in fields(NCfg)}})
    saved.work_dir = cfg.work_dir
    if "use_feats" not in ck["cfg"]:
        saved.use_feats = False                  # run trained with an older version of this script (no feature embeddings)
    if cfg.local_data_dir:
        saved.local_data_dir = cfg.local_data_dir
    C = 1 + 2 * len(ck["labels"])
    net = build_net(saved, P1.LM(ck["mc"]), C)
    apply_method(net, saved)                     # re-create the LoRA / LoRMA adapters before loading their weights
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    return {"rd": rd, "ck": ck, "saved": saved, "net": net, "labels": ck["labels"], "C": C,
            "tok": Tokenizer.from_file(str(rd / "tokenizer.json"))}


def write_submission(cfg: NCfg, rd: Path, spans: dict, texts: dict, labels, end_shift: int):
    """spans: doc id -> [(start, end_exclusive, label_idx)] in ORIGINAL characters -> prediction.json + submission.zip"""
    out = {}
    for did in texts:
        seen, lst = set(), []
        for s_, e_, l_ in sorted(spans.get(did, [])):
            t = (s_, e_ - end_shift, labels[l_])
            if t not in seen and t[1] > t[0] - (1 if end_shift else 0):
                seen.add(t)
                lst.append([t[0], t[1], t[2]])
        out[did] = lst
    problems = validate_prediction(out, texts, labels, end_shift)
    if problems:
        log(f"[predict] WARNING: {problems[:5]}")
    rd.mkdir(parents=True, exist_ok=True)
    pj = rd / "prediction.json"
    P1.jdump(out, pj)
    zp = rd / "submission.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    hist = Path(cfg.work_dir) / "ner_submissions"
    hist.mkdir(parents=True, exist_ok=True)
    shutil.copy(zp, hist / f"{cfg.run}_{time.strftime('%m%d_%H%M%S')}.zip")
    n_ent = sum(len(v) for v in out.values())
    log(f"[predict] wrote {pj} ({n_ent} entities, {n_ent / max(1, len(out)):.1f} per doc) and {zp}")
    return str(zp)


def run_predict(cfg: NCfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, amp = P1.amp_setup(dev)
    r = load_run(cfg, cfg.run, dev)
    ck, labels = r["ck"], r["labels"]
    mode, o_bias, end_shift = ck["decode"], float(ck.get("o_bias", 0.0)), ck["end_shift"]
    cbias = np.array(ck["cbias"], dtype=np.float32) if ck.get("cbias") else None
    texts = load_test(cfg)
    log(f"[predict] {len(texts)} test documents, decoder={mode}, O-bias={o_bias:+.1f}")
    docs = make_docs(r["tok"], texts, None, {l: i for i, l in enumerate(labels)}, end_shift, {})
    trans, start = build_trans(len(labels))
    spans = predict_docs(r["net"], docs, r["saved"], dev, amp, r["C"], mode, trans, start, o_bias, cbias)
    return write_submission(cfg, r["rd"], spans, texts, labels, end_shift)


def _log_csv_row(cfg: NCfg, row: dict):
    p = Path(cfg.work_dir) / "ner_experiments.csv"
    if p.exists() and open(p).readline().strip().split(",") != NER_FIELDS:
        p.rename(p.with_suffix(".old.csv"))
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NER_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row})


def run_recalibrate(cfg: NCfg):
    """No retraining: re-tune decoder + O-bias of a finished run on its validation documents, then re-predict."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, amp = P1.amp_setup(dev)
    r = load_run(cfg, cfg.run, dev)
    saved, ck, labels = r["saved"], r["ck"], r["labels"]
    saved.decode = cfg.decode
    _, _, end_shift, tr, va, stats = prepare_data(saved, r["tok"], None)
    trans, start = build_trans(len(labels))
    probs = doc_probs(r["net"], va, saved, dev, amp, r["C"])
    best, table = tune_decoding(va, probs, saved, labels, trans, start)
    sc, mode, o_bias = best["sc"], best["mode"], best["o_bias"]
    log(f"[recalibrate] {cfg.run}: CHOSEN {mode} O-bias {o_bias:+.1f} -> val macro-F1 {sc['macro_f1']:.4f} "
        f"(P {sc['macro_p']:.4f} R {sc['macro_r']:.4f})")
    for l in labels:
        pl = sc["per_label"][l]
        log(f"   {l:<10} P={pl['P']:.3f} R={pl['R']:.3f} F1={pl['F1']:.3f}  (tp {pl['tp']} fp {pl['fp']} fn {pl['fn']})")
    err = analyze_errors(eval_gold_sets(va), best["sets"])
    log(f"[recalibrate] error breakdown: {err}")
    P1.jdump(error_examples(va, eval_gold_sets(va), best["sets"]), r["rd"] / "error_examples.json")
    ck["decode"], ck["o_bias"] = mode, o_bias
    ck["cbias"] = None if best["cbias"] is None else [float(x) for x in best["cbias"]]
    torch.save(ck, r["rd"] / "best.pt")
    rep_path = r["rd"] / "val_report.json"
    rep = P1.jload(rep_path) if rep_path.exists() else {}
    rep.update({"decode": mode, "o_bias": o_bias, "cbias": ck["cbias"], "decode_table": table, "scores": sc, "errors": err})
    P1.jdump(rep, rep_path)
    _log_csv_row(cfg, {"run": cfg.run, "p1_arch": ck["mc"]["arch"], "method": saved.method, "variant": "recalibrated",
                       "val_macro_f1": round(sc["macro_f1"], 4), "val_macro_p": round(sc["macro_p"], 4),
                       "val_macro_r": round(sc["macro_r"], 4), "decode": f"{mode}/o{o_bias:+.1f}"})
    del r["net"]
    return run_predict(cfg)


def _ens_candidates(cfg: NCfg):
    out = []
    for f in sorted((Path(cfg.work_dir) / "ner_runs").glob("*/best.pt")):
        n = f.parent.name
        if cfg.ens_prefix:
            if not n.startswith(cfg.ens_prefix):
                continue
        elif n.startswith(("selftest", "check")):
            continue
        out.append(n)
    return out


def run_ensemble(cfg: NCfg):
    """Average the per-token class probabilities of several finished runs (they share the Phase-1 tokenizer and the
    validation split), tune the decoder on validation, write one submission.
      "ens_runs": [...]   explicit members            "ens_auto": true   greedy forward selection among ALL finished runs"""
    names = list(cfg.ens_runs) or (_ens_candidates(cfg) if cfg.ens_auto else [])
    if len(names) < 2:
        raise SystemExit('need >= 2 finished runs: set "ens_runs": [...] or "ens_auto": true')
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, amp = P1.amp_setup(dev)
    ref, vp, va = {}, {}, None
    for n in names:                                   # validation probabilities of every candidate (one network in memory at a time)
        r = load_run(cfg, n, dev)
        if not ref:
            ref = {"labels": r["labels"], "tok": r["tok"].to_str(), "C": r["C"], "end_shift": r["ck"]["end_shift"],
                   "split": (r["saved"].val_frac, r["saved"].split_seed)}
            _, _, _, tr, va, _ = prepare_data(r["saved"], r["tok"], None)
        if (r["labels"] != ref["labels"] or r["tok"].to_str() != ref["tok"]
                or (r["saved"].val_frac, r["saved"].split_seed) != ref["split"]):
            log(f"[ensemble] skipping {n}: different labels / tokenizer / validation split")
        else:
            vp[n] = doc_probs(r["net"], va, r["saved"], dev, amp, ref["C"])
        del r
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    labels, C, end_shift = ref["labels"], ref["C"], ref["end_shift"]
    trans, start = build_trans(len(labels))
    gold = eval_gold_sets(va)

    def mean(keys):
        return [sum(vp[k][i] for k in keys) / len(keys) for i in range(len(va))]

    def quick(plist):                                 # tuned greedy macro-F1 (cheap)
        top = 0.0
        for b_ in (-0.5, 0.0, 0.5, 1.0):
            pr = {d.id: decode_doc(d, p, "greedy", trans, start, b_) for d, p in zip(va, plist)}
            top = max(top, strict_scores(gold, to_sets(pr, labels), labels)["macro_f1"])
        return top

    single = {k: quick(vp[k]) for k in vp}
    for k, v in sorted(single.items(), key=lambda kv: -kv[1]):
        log(f"[ensemble] single model {k:<28} val macro-F1 {v:.4f}")
    if cfg.ens_runs or not cfg.ens_auto:
        chosen = list(vp)
    else:
        order = sorted(single, key=lambda k: -single[k])
        chosen, cur = [order[0]], single[order[0]]
        for k in order[1:]:
            if len(chosen) >= cfg.ens_max:
                break
            f = quick(mean(chosen + [k]))
            log(f"[ensemble] try adding {k:<28} -> {f:.4f} (current {cur:.4f})")
            if f > cur + 0.002:
                chosen.append(k)
                cur = f
    log(f"[ensemble] members: {chosen}")
    best, table = tune_decoding(va, mean(chosen), cfg, labels, trans, start)
    sc, mode, o_bias, cbias = best["sc"], best["mode"], best["o_bias"], best["cbias"]
    log(f"[ensemble] val macro-F1 {sc['macro_f1']:.4f} (P {sc['macro_p']:.4f} R {sc['macro_r']:.4f}) with {mode}, "
        f"O-bias {o_bias:+.1f}{', per-class thresholds' if cbias is not None else ''}")
    runs = [load_run(cfg, n, dev) for n in chosen]    # now the test set, with only the chosen networks
    texts = load_test(cfg)
    docs = make_docs(runs[0]["tok"], texts, None, {l: i for i, l in enumerate(labels)}, end_shift, {})
    spans = {}
    for g in range(0, len(docs), 16):
        sub = docs[g:g + 16]
        probs = [doc_probs(r["net"], sub, r["saved"], dev, amp, C) for r in runs]
        for i, d in enumerate(sub):
            spans[d.id] = decode_doc(d, sum(p[i] for p in probs) / len(runs), mode, trans, start, o_bias, cbias)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    P1.jdump({"members": chosen, "single_model_f1": single, "decode": mode, "o_bias": o_bias,
              "cbias": None if cbias is None else [float(x) for x in cbias], "scores": sc, "decode_table": table},
             rd / "val_report.json")
    _log_csv_row(cfg, {"run": cfg.run, "p1_arch": "ensemble", "method": "ensemble", "val_macro_f1": round(sc["macro_f1"], 4),
                       "val_macro_p": round(sc["macro_p"], 4), "val_macro_r": round(sc["macro_r"], 4),
                       "decode": f"{mode}/o{o_bias:+.1f}", "teacher_run": "+".join(chosen)})
    return write_submission(cfg, rd, spans, texts, labels, end_shift)


def validate_prediction(pred, texts, labels, end_shift=0):
    """Same checks as evaluate_phase2.py (+ bounds against the document text)."""
    bad, allowed = [], set(labels)
    if set(pred) != set(texts):
        bad.append("id mismatch")
    for did, lst in pred.items():
        seen = set()
        for j, s in enumerate(lst):
            if not (isinstance(s, list) and len(s) in (3, 4)):
                bad.append(f"{did}[{j}] bad shape")
                continue
            a, b, l = s[:3]
            if not (isinstance(a, int) and isinstance(b, int)) or a < 0 or b <= a:
                bad.append(f"{did}[{j}] bad offsets {a},{b}")
            if b > len(texts[did]):
                bad.append(f"{did}[{j}] beyond text")
            if l not in allowed:
                bad.append(f"{did}[{j}] bad label {l}")
            if (a, b, l) in seen:
                bad.append(f"{did}[{j}] duplicate")
            seen.add((a, b, l))
    return bad


# ============================================================================================
# 8. Self-test (synthetic data, tiny models, all 4 Phase-1 architectures)
# ============================================================================================
def _synthetic(tmp, rng):
    first = ["Ram", "Shyam", "Anita", "Suresh", "Meena", "Kiran", "Vijay", "Latha", "Arun", "Geeta"]
    last = ["Kumar", "Sharma", "Reddy", "Singh", "Iyer", "Das", "Patel", "Nair"]
    courts = ["SUPREME COURT OF INDIA", "High Court of Kerala", "High Court of Delhi"]
    months = ["January", "March", "June", "October"]
    filler = "the learned counsel submitted that the impugned order was passed without hearing and hence liable to be set aside".split()

    def doc():
        text, ents = "", []

        def add(s, label=None):
            nonlocal text
            if label:
                ents.append([len(text), len(text) + len(s), label, s])
            text += s
        add("IN THE ")
        add(rng.choice(courts), "COURT")
        add("\n\n  CIVIL APPEAL NO. ")
        add(f"{rng.randint(100, 999)} OF {rng.randint(1990, 2020)}", "CASENO")
        add("\n\n")
        add(f"{rng.choice(first)} {rng.choice(last)}", "APP")
        add("   ...Appellant\n  Versus \n")
        add(f"State of {rng.choice(last)}", "RESP")
        add("  ...Respondent\n\nJudgment by ")
        add(f"{rng.choice(first)[0]}. {rng.choice(last).upper()}", "JUDGE")
        add(", J. on ")
        add(f"{rng.randint(1, 28)} {rng.choice(months)} {rng.randint(1990, 2020)}", "DATE")
        add(".\n\n")
        for _ in range(rng.randint(3, 12)):
            add(" ".join(rng.choice(filler) for _ in range(rng.randint(8, 20))) + ".\n")
        if rng.random() < 0.5:
            add("Counsel ")
            add(f"{rng.choice(first)} {rng.choice(last)}", "A.COUNSEL")
            add(" appeared for the appellant.\n")
        return text, ents

    data = {}
    for k in (1, 2):
        texts, gold = {}, {}
        for i in range(20):
            did = f"{k}{i:03d}"
            texts[did], gold[did] = doc()
        data[k] = (texts, gold)
    test = {f"t{i}": doc()[0] for i in range(5)}
    os.makedirs(tmp, exist_ok=True)
    for k, (t, g) in data.items():
        json.dump(t, open(os.path.join(tmp, f"fold{k}-text.json"), "w"))
        json.dump(g, open(os.path.join(tmp, f"fold{k}-gold.json"), "w"))
    json.dump(test, open(os.path.join(tmp, "fold3-text.json"), "w"))
    open(os.path.join(tmp, "labels.txt"), "w").write("\n".join(LABELS_DEFAULT) + "\n")
    return data, test


def _tiny_mc(arch, V):
    return {**P1.ARCH_PRESETS[arch], "n_embd": 64, "n_layer": 2, "n_head": 4, "n_kv_head": 2, "ffn_hidden": 96,
            "n_experts": 4, "top_k": 2, "arch": arch, "ctx": 128, "vocab": V, "dropout": 0.0}


def selftest() -> bool:
    import tempfile
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    ok = True
    rng = random.Random(0)
    tmp = tempfile.mkdtemp(prefix="cs781_ner_selftest_")
    dd = os.path.join(tmp, "data")
    data, test = _synthetic(dd, rng)
    log("=== 1. text normalisation / offset map ===")
    for s in ["a  b\n\nc", "  lead trail  ", "x\u00a0y\tz", ""]:
        n, m = normalize_with_map(s)
        good = all(s[m[i]] == n[i] or (n[i] == " " and s[m[i]].isspace()) for i in range(len(n)))
        ok &= good
        log(f"  {s!r:<22} -> {n!r:<16} {'PASS' if good else 'FAIL'}")

    log("=== 2. tokeniser + oracle (gold -> BIO -> char offsets must reproduce gold exactly) ===")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [normalize_with_map(t)[0] for k in data for t in data[k][0].values()]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=500, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    texts = {**data[1][0], **data[2][0]}
    gold = {**data[1][1], **data[2][1]}
    label2i = {l: i for i, l in enumerate(LABELS_DEFAULT)}
    shift, info = detect_end_shift(texts, gold)
    stats = {}
    docs = make_docs(tok, texts, {d: parse_gold_items(g) for d, g in gold.items()}, label2i, shift, stats)
    o = oracle(docs, LABELS_DEFAULT)
    good = shift == 0 and present_f1(o) > 0.999
    ok &= good
    log(f"  end_shift={shift} {info}\n  oracle strict F1 (labels present) = {present_f1(o):.4f} stats={stats}  "
        f"{'PASS' if good else 'FAIL'}")

    log("=== 2b. layout features (line breaks) and position buckets ===")
    n_, m_, nlb_, nla_ = normalize_with_feats("AAA BBB\n\nCCC DDD\nEEE")
    good = (n_ == "AAA BBB CCC DDD EEE" and [int(nlb_[i]) for i in (0, 4, 8, 12, 16)] == [2, 0, 2, 0, 1]
            and [int(nla_[i]) for i in (2, 6, 10, 14, 18)] == [0, 2, 0, 1, 2])
    good2 = any(int(d.feat.max()) > 0 for d in docs) and int(np.searchsorted(POS_EDGES, 63, side="right")) == 0 \
        and int(np.searchsorted(POS_EDGES, 64, side="right")) == 1 and int(np.searchsorted(POS_EDGES, 10 ** 6, side="right")) == 8
    ok &= good and good2
    log(f"  newline features {'PASS' if good else 'FAIL'} | per-token features / position buckets {'PASS' if good2 else 'FAIL'}")

    log("=== 3. viterbi / windows ===")
    trans, start = build_trans(2)
    lp = np.log(np.array([[.1, .1, .6, .1, .1], [.6, .1, .1, .1, .1], [.1, .1, .1, .1, .6]]))
    path = viterbi(lp, trans, start)
    good = path[0] not in (2, 4) and path[2] not in (2, 4) or path[1] in (1, 2, 3, 4)   # I-x never starts a span
    good = bool(path[0] not in (2, 4))
    ok &= good
    wins = make_windows(1000, 512, 384)
    good2 = wins[0][0] == 0 and wins[-1][1] == 1000 and all(b - a <= 512 for a, b in wins)
    ok &= good2
    log(f"  viterbi start constraint {'PASS' if good else 'FAIL'} | windows {wins} {'PASS' if good2 else 'FAIL'}")

    V = P1.pad64(tok.get_vocab_size())
    log("=== 3b. adapters are the IDENTITY at initialisation; only adapters + head are trainable; KD helpers ===")
    for arch in P1.ARCH_PRESETS:
        for meth, var in (("lora", "plus"), ("lorma", "plus"), ("lorma", "pi")):
            tag = f"{arch}/{meth}" + (f"-{var}" if meth == "lorma" else "")
            try:
                torch.manual_seed(0)
                c = apply_overrides(NCfg(), {"method": meth, "lorma_variant": var, "peft_r": 4, "peft_alpha": 8.0,
                                             "peft_dropout": 0.0})
                net = build_net(c, P1.LM(_tiny_mc(arch, V)), 25).eval()
                ids = torch.randint(1, V, (2, 16))
                km = torch.ones((2, 16), dtype=torch.bool)
                with torch.no_grad():
                    ref = net(ids, km)
                apply_method(net, c)
                net.eval()
                with torch.no_grad():
                    out = net(ids, km)
                diff = (ref - out).abs().max().item()
                n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
                n_all = sum(p.numel() for p in net.parameters())
                net.train()
                net(ids, km).float().pow(2).mean().backward()
                grads = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.parameters() if p.requires_grad)
                frozen_ok = all(p.grad is None for p in net.parameters() if not p.requires_grad)
                good = diff < 1e-4 and n_tr < n_all and grads and frozen_ok
                ok &= good
                log(f"  {tag:<18} |out_adapted - out_base| = {diff:.1e}  trainable {n_tr:,}/{n_all:,}  "
                    f"{'PASS' if good else 'FAIL'}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  {tag:<18} FAIL: {type(e).__name__}: {e}")
                ok = False
    sd = {"wte.weight": 0, "blocks.0.a": 1, "blocks.1.a": 2, "blocks.2.a": 3, "norm_f.weight": 4}
    good = remap_student_keys(sd, [0, 2]) == {"wte.weight": 0, "blocks.0.a": 1, "blocks.1.a": 3, "norm_f.weight": 4}
    tl = torch.randn(2, 5, 25)
    lb = torch.randint(0, 25, (2, 5))
    lb[0, 0] = -100
    good2 = abs(float(kd_loss(tl, tl, lb, 2.0))) < 1e-5 and float(kd_loss(tl, torch.randn(2, 5, 25), lb, 2.0)) > 0
    ok &= good and good2
    log(f"  layer remap {'PASS' if good else 'FAIL'} | kd_loss (0 for identical logits, > 0 otherwise) {'PASS' if good2 else 'FAIL'}")

    log("=== 4. end-to-end fine-tuning on synthetic data, all 4 Phase-1 architectures ===")
    p1dirs = {}
    for arch in P1.ARCH_PRESETS:
        try:
            mc = _tiny_mc(arch, V)
            p1 = os.path.join(tmp, f"p1_{arch}")
            os.makedirs(p1, exist_ok=True)
            p1dirs[arch] = p1
            torch.save({"model": P1.LM(mc).state_dict(), "mc": mc, "state": {"step": 0}}, os.path.join(p1, "ckpt_best.pt"))
            tok.save(os.path.join(p1, "tokenizer.json"))
            cfg = apply_overrides(NCfg(), {"run": f"selftest_{arch}", "work_dir": os.path.join(tmp, "work"), "p1_dir": p1,
                                           "local_data_dir": dd, "max_len": 96, "stride": 64, "epochs": 2,
                                           "batch_size": 8, "eval_batch": 8, "lr": 1e-3, "val_frac": 0.2})
            run_train(cfg)
            with zipfile.ZipFile(run_dir(cfg) / "submission.zip") as z:
                assert z.namelist() == ["prediction.json"]
                pr = json.loads(z.read("prediction.json"))
            assert set(pr) == set(test), "ids"
            assert not validate_prediction(pr, test, LABELS_DEFAULT), "format"
            log(f"  e2e {arch:<7} PASS")
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  e2e {arch:<7} FAIL: {type(e).__name__}: {e}")
            ok = False
    log("=== 5. every adaptation method end-to-end (LoRA, LoRMA+, LoRMAπ, distillation) ===")
    base_ov = {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "max_len": 96, "stride": 64, "epochs": 2,
               "batch_size": 8, "eval_batch": 8, "lr": 1e-3, "peft_lr": 3e-3, "val_frac": 0.2, "peft_r": 4,
               "peft_alpha": 8.0}
    cases = [("lora", "llama", {"method": "lora"}),
             ("lorma_plus", "gpt2", {"method": "lorma", "lorma_variant": "plus"}),
             ("lorma_pi", "retnet", {"method": "lorma", "lorma_variant": "pi"}),
             ("kd", "gpt2", {"method": "kd", "teacher_run": "selftest_llama", "student_layers": 1})]
    for tag, arch, extra in cases:
        try:
            cfg = apply_overrides(NCfg(), {**base_ov, **extra, "run": f"selftest_{tag}", "p1_dir": p1dirs[arch]})
            run_train(cfg)
            with zipfile.ZipFile(run_dir(cfg) / "submission.zip") as z:
                assert z.namelist() == ["prediction.json"]
                pr = json.loads(z.read("prediction.json"))
            assert set(pr) == set(test) and not validate_prediction(pr, test, LABELS_DEFAULT), "format"
            rep_ = P1.jload(run_dir(cfg) / "val_report.json")
            if extra["method"] in ("lora", "lorma"):
                assert rep_["params_trainable"] < rep_["params_total"], "adapters must be a small subset"
            if extra["method"] == "kd":
                assert rep_["params_total"] < rep_["teacher_params"], "student must be smaller than the teacher"
            log(f"  e2e {tag:<11} PASS  (trainable {rep_['params_trainable']:,} / total {rep_['params_total']:,})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  e2e {tag:<11} FAIL: {type(e).__name__}: {e}")
            ok = False
    log("=== 6. recalibrate (no retraining) + ensemble ===")
    try:
        cfg = apply_overrides(NCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "run": "selftest_llama"})
        zp = run_recalibrate(cfg)
        with zipfile.ZipFile(zp) as z:
            assert z.namelist() == ["prediction.json"]
            assert not validate_prediction(json.loads(z.read("prediction.json")), test, LABELS_DEFAULT)
        cfg = apply_overrides(NCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "run": "selftest_ens",
                                       "ens_runs": ["selftest_llama", "selftest_gpt2"]})
        zp = run_ensemble(cfg)
        with zipfile.ZipFile(zp) as z:
            pr = json.loads(z.read("prediction.json"))
        assert set(pr) == set(test) and not validate_prediction(pr, test, LABELS_DEFAULT)
        cfg = apply_overrides(NCfg(), {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "run": "selftest_ens_auto",
                                       "ens_auto": True, "ens_prefix": "selftest_"})
        zp = run_ensemble(cfg)
        with zipfile.ZipFile(zp) as z:
            pr = json.loads(z.read("prediction.json"))
        assert set(pr) == set(test) and not validate_prediction(pr, test, LABELS_DEFAULT)
        log("  recalibrate + explicit ensemble + automatic ensemble PASS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"  recalibrate / ensemble FAIL: {type(e).__name__}: {e}")
        ok = False
    shutil.rmtree(tmp, ignore_errors=True)
    log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


# ============================================================================================
# 9. CLI
# ============================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["selftest", "check", "train", "predict", "recalibrate", "ensemble"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--cfg", default="", help="JSON string or JSON file with config overrides")
    ap.add_argument("--p1_dir", default=None, help="Phase-1 run folder (ckpt_best.pt + tokenizer.json)")
    args = ap.parse_args(argv)
    if args.stage == "selftest":
        return 0 if selftest() else 1
    cfg = NCfg()
    ov = {}
    if args.cfg:
        ov = json.loads(args.cfg) if args.cfg.strip().startswith("{") else P1.jload(args.cfg)
    if args.run:
        ov["run"] = args.run
    if args.p1_dir:
        ov["p1_dir"] = args.p1_dir
    apply_overrides(cfg, ov)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        diagnose(cfg)
    elif args.stage == "train":
        run_train(cfg)
    elif args.stage == "predict":
        run_predict(cfg)
    elif args.stage == "recalibrate":
        run_recalibrate(cfg)
    elif args.stage == "ensemble":
        run_ensemble(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
