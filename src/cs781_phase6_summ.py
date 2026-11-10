#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS781 Phase 6 - Abstractive legal summarisation on top of YOUR Phase-1 language models.

Needs cs781_phase1.py, cs781_phase2_ner.py and cs781_phase4_cjpe.py in the same folder (model classes, adapters, distillation helper).
Local scoring needs `pip install rouge-score bert-score` (the same libraries as the official evaluator).

Pipeline
  1. LONG DOCUMENTS: your Phase-1 models read at most `ctx` (1024) tokens, a judgment has thousands. Each document is compressed by an
     importance-based selector ("select": learned | lead | head_tail | uniform): a small gradient-boosting model, trained on the training set to
     predict how much of the reference summary each sentence covers, keeps the best sentences (original order) within `max_src` tokens.
  2. TRAINING: the Phase-1 decoder-only LM is fine-tuned on   <eos> source <sep> summary <eos>   with the loss on the summary tokens only.
  3. DECODING: batched KV-cache generation (greedy / sampling, repetition penalty, no-repeat n-gram, minimum / maximum length); the decoding
     setup and the generation length are chosen on a held-out dev split by BERTScore F1 (the primary metric); light redundancy removal.
  4. Output: [{"id": ..., "summary": ...}, ...]  (the list format recommended by the task).

Adaptation methods (cfg "method")
  full | partial | bitfit | adapter (Houlsby) | lora | lorma (LoRMA+ / LoRMAπ) | kd (pruned student <- teacher: logit distillation on the summary tokens)

Stages:  selftest | check | train | predict | mbr
  python cs781_phase6_summ.py check --run sm_check --cfg '{"p1_dir": ".../explore_llama"}'
  python cs781_phase6_summ.py train --run sm_full_llama --cfg cfg_sm.json
  python cs781_phase6_summ.py mbr   --run sm_consensus --cfg '{"ens_auto": true, "ens_prefix": "sm_"}'     # consensus of several runs' summaries
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
from collections import Counter
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
SEP_TEXT = " <<SUMMARY>> "


# ============================================================================================
# 0. Config
# ============================================================================================
@dataclass
class SCfg:
    run: str = "sm_full_v1"
    work_dir: str = P1.DEFAULT_WORK
    p1_dir: str = ""
    p1_ckpt: str = "ckpt_best.pt"
    hf_repo: str = "Exploration-Lab/CS-781-Capstone"
    local_data_dir: str = ""          # folder with train.json and test.json
    train_file: str = "Phase_6_Summ/Train/train.json"
    test_file: str = "Phase_6_Summ/Test/test.json"
    max_train_docs: int = 0
    dev_docs: int = 200               # held-out part of train.json (deterministic, by hash of the id)
    # ---- long documents -> model input
    select: str = "learned"           # learned | lead | head_tail | uniform
    max_src: int = 0                  # source tokens kept (0 = ctx - max_sum - small margin)
    max_sum: int = 0                  # summary tokens trained / generated (0 = 75th percentile of reference lengths, capped at 520)
    selector_docs: int = 1500         # training documents used to fit the sentence selector
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
    epochs: int = 4
    micro_bs: int = 4
    accum: int = 4
    lr: float = 2e-4
    llrd: float = 0.85
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    ema_decay: float = 0.995
    time_budget_min: float = 45.0
    eval_every_steps: int = 400
    dev_loss_docs: int = 200
    patience: int = 5
    seed: int = 1337
    # ---- generation / evaluation
    gen_batch: int = 16
    sweep_n: int = 48                 # dev documents used for the decoding sweep
    final_dev_n: int = 120            # dev documents scored with the chosen decoding
    min_new_frac: float = 0.4         # minimum summary length as a fraction of the maximum
    len_scales: list = field(default_factory=lambda: [0.6, 0.8, 1.0])
    decode: dict = field(default_factory=dict)            # fixed decoding config (skips the sweep)
    sweep_grid: list = field(default_factory=lambda: [
        {"name": "greedy", "temperature": 0.0, "rep": 1.0},
        {"name": "greedy_rp1.1", "temperature": 0.0, "rep": 1.1},
        {"name": "greedy_rp1.2", "temperature": 0.0, "rep": 1.2},
        {"name": "greedy_nr3", "temperature": 0.0, "nr": 3},
        {"name": "greedy_rp1.1_nr4", "temperature": 0.0, "rep": 1.1, "nr": 4},
        {"name": "t0.3_p0.9_rp1.1", "temperature": 0.3, "top_p": 0.9, "rep": 1.1},
        {"name": "t0.6_p0.9_rp1.1_nr4", "temperature": 0.6, "top_p": 0.9, "rep": 1.1, "nr": 4}])
    use_bertscore: bool = True
    predict_after: bool = True
    # ---- consensus stage
    ens_runs: list = field(default_factory=list)
    ens_auto: bool = False
    ens_prefix: str = ""
    ens_max: int = 5


def apply_overrides(cfg: SCfg, ov: dict) -> SCfg:
    names = {f.name for f in fields(SCfg)}
    for k, v in ov.items():
        if k not in names:
            raise SystemExit(f"Unknown config key {k!r}. Valid: {sorted(names)}")
        setattr(cfg, k, v)
    return cfg


def run_dir6(cfg: SCfg) -> Path:
    return Path(cfg.work_dir) / "sm_runs" / cfg.run


SM_FIELDS = ["time", "run", "p1_arch", "method", "select", "params_total_M", "params_trainable_M", "dev_bertscore", "dev_rouge1", "dev_rougeL",
             "dev_loss", "decode", "max_new", "avg_words", "ref_words", "rep_rate", "train_s", "peak_gpu_mb", "gen_docs_s", "teacher_run"]


def log_csv(cfg: SCfg, row: dict):
    p = Path(cfg.work_dir) / "sm_experiments.csv"
    if p.exists() and open(p).readline().strip().split(",") != SM_FIELDS:
        p.rename(p.with_suffix(".old.csv"))
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SM_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row})


# ============================================================================================
# 1. Sentences, importance-based selection
# ============================================================================================
_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9("\'])|\n+')
_WORD = re.compile(r"[a-z0-9]+")
CUES = set("held dismissed allowed appeal appellant respondent section article court decree order directed therefore accordingly conclusion "
           "hence thus affirmed set aside rejected petition high supreme act under judgment evidence contract property tax liable".split())
TOK_PER_WORD = 1.4


def split_sents(text: str, glue: bool = True):
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    if not glue:
        return parts or [text.strip()]
    out = []
    for p in parts:                                          # glue tiny fragments ("No.", "v.") to the next sentence
        if out and len(out[-1].split()) < 4:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out or [text.strip()]


def words_of(s: str):
    return _WORD.findall(s.lower())


def sent_features(sents, idf, default_idf):
    wl = [words_of(s) for s in sents]
    tf = Counter()
    for ws in wl:
        tf.update(ws)
    total = max(1, sum(tf.values()))
    n = len(sents)
    rows = []
    for i, ws in enumerate(wl):
        L = max(1, len(ws))
        cent = sum(tf[w] / total * idf.get(w, default_idf) for w in ws) / L
        rows.append([i / max(1, n - 1), min(i, 10), min(n - 1 - i, 10), L, math.log1p(L), cent * 1000.0,
                     sum(w in CUES for w in ws) / L, sum(w.isdigit() for w in ws) / L, math.log1p(n)])
    return np.array(rows, dtype=np.float32), wl


class Selector:
    """Predicts, for every sentence, the idf-weighted share of its words that also occur in the reference summary, from cheap features
    (position, length, centroid similarity, legal cue words, digits). Fitted once on the training set; falls back to a heuristic."""

    def __init__(self):
        self.idf, self.default_idf, self.model = {}, 5.0, None

    def fit(self, docs, refs):
        df = Counter()
        for d in docs:
            df.update(set(words_of(d)))
        N_ = len(docs)
        self.idf = {w: math.log((N_ + 1) / (c + 1)) + 1.0 for w, c in df.items() if c >= 2}
        self.default_idf = math.log(N_ + 1) + 1.0
        X, y = [], []
        for d, r in zip(docs, refs):
            sents = split_sents(d)
            f, wl = sent_features(sents, self.idf, self.default_idf)
            refw = set(words_of(r))
            for ws in wl:
                den = sum(self.idf.get(w, self.default_idf) for w in ws) or 1.0
                y.append(sum(self.idf.get(w, self.default_idf) for w in ws if w in refw) / den)
            X.append(f)
        X, y = np.concatenate(X), np.array(y, dtype=np.float32)
        if len(X) > 400000:
            sel = np.random.default_rng(0).permutation(len(X))[:400000]
            X, y = X[sel], y[sel]
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.1, random_state=0).fit(X, y)
            log(f"[select] sentence selector fitted on {len(X):,} sentences from {N_} documents")
        except ImportError:
            log("[select] scikit-learn missing: using the heuristic selector (centroid similarity + position)")
        return self

    def scores(self, sents):
        f, _ = sent_features(sents, self.idf, self.default_idf)
        if self.model is not None:
            return self.model.predict(f)
        return f[:, 5] * 0.01 + 0.2 * (f[:, 0] < 0.1) + 0.1 * (f[:, 0] > 0.9)


def select_text(doc: str, budget_words: int, mode: str, selector: Selector | None = None) -> str:
    sents = split_sents(doc)
    wc = [len(s.split()) for s in sents]
    if sum(wc) <= budget_words:
        return " ".join(sents)
    n = len(sents)
    if mode == "lead":
        chosen = []
        tot = 0
        for i in range(n):
            if tot + wc[i] > budget_words:
                break
            chosen.append(i)
            tot += wc[i]
    elif mode == "head_tail":
        head, tail, th, tt = [], [], 0, 0
        for i in range(n):                                  # 40% of the budget from the start, 60% from the end
            if th + wc[i] > 0.4 * budget_words:
                break
            head.append(i)
            th += wc[i]
        for i in range(n - 1, len(head) - 1, -1):
            if tt + wc[i] > 0.6 * budget_words:
                break
            tail.append(i)
            tt += wc[i]
        chosen = sorted(set(head) | set(tail))
    elif mode == "uniform":
        step = sum(wc) / budget_words
        chosen, acc, nxt = [], 0.0, 0.0
        for i in range(n):
            if acc >= nxt:
                chosen.append(i)
                nxt += step * wc[i]
            acc += wc[i]
        tot = 0
        keep = []
        for i in chosen:
            if tot + wc[i] <= budget_words:
                keep.append(i)
                tot += wc[i]
        chosen = keep
    else:                                                   # learned
        sc = selector.scores(sents) if selector is not None else np.zeros(n)
        chosen, tot = [], 0
        for i in np.argsort(-np.asarray(sc)):
            if tot + wc[i] <= budget_words:
                chosen.append(int(i))
                tot += wc[i]
        chosen.sort()
    return " ".join(sents[i] for i in chosen) if chosen else " ".join(doc.split()[:budget_words])


def ref_coverage(selected: str, ref: str, idf, default_idf) -> float:
    """idf-weighted fraction of the reference summary's words that survive in the selected source (information kept after compression)."""
    rw, sw = set(words_of(ref)), set(words_of(selected))
    den = sum(idf.get(w, default_idf) for w in rw) or 1.0
    return sum(idf.get(w, default_idf) for w in rw if w in sw) / den


# ============================================================================================
# 2. Data
# ============================================================================================
class Ex:
    pass


def _first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def read_split(cfg: SCfg, rel: str, labelled: bool, max_docs: int = 0):
    raw = P1.jload(P1.resolve_file(cfg, rel))
    items = raw if isinstance(raw, list) else [dict(v, id=k) if isinstance(v, dict) else {"id": k, "document": v} for k, v in raw.items()]
    if max_docs:
        items = items[:max_docs]
    out = []
    for i, r in enumerate(items):
        did = _first(r, ("id", "doc_id", "document_id", "case_id"))
        doc = _first(r, ("document", "text", "judgment", "content")) or ""
        doc = "\n".join(doc) if isinstance(doc, list) else doc
        e = {"id": str(did) if did is not None else str(i), "doc": doc}
        if labelled:
            s = _first(r, ("summary", "abstract", "gold", "reference")) or ""
            e["summary"] = " ".join((" ".join(s) if isinstance(s, list) else s).split())
        out.append(e)
    return out


def build_data(cfg: SCfg, tok, ctx: int):
    """-> dict(train, dev, test lists of Ex) with token ids; the selector, lengths and ids are cached per setting."""
    tp_bytes = tok.to_str().encode()
    path = Path(P1.resolve_file(cfg, cfg.train_file))
    key = hashlib.md5(tp_bytes + f"{cfg.select}|{cfg.max_src}|{cfg.max_sum}|{cfg.dev_docs}|{cfg.max_train_docs}|{cfg.selector_docs}|{ctx}|{path.stat().st_size}".encode()).hexdigest()[:12]
    cache = Path(cfg.work_dir) / "cache" / f"summ_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
        log(f"[data] loaded the prepared data from the cache ({len(data['train'])} train / {len(data['dev'])} dev / {len(data['test'])} test)")
        return data
    t0 = time.time()
    allx = read_split(cfg, cfg.train_file, True, cfg.max_train_docs)
    test = read_split(cfg, cfg.test_file, False)
    allx.sort(key=lambda e: hashlib.md5(e["id"].encode()).hexdigest())                  # deterministic split by id hash
    dev, train = allx[:cfg.dev_docs], allx[cfg.dev_docs:]
    sep_ids = tok.encode(SEP_TEXT).ids
    enc = lambda texts: [e.ids for b in range(0, len(texts), 128) for e in tok.encode_batch(texts[b:b + 128])]
    sum_tok = enc([e["summary"] for e in train[:3000]])
    lens = np.array([len(x) for x in sum_tok])
    max_sum = cfg.max_sum or int(np.clip(np.percentile(lens, 75), 128, 520))
    max_src = cfg.max_src or (ctx - max_sum - len(sep_ids) - 8)
    assert max_src + max_sum + len(sep_ids) + 2 <= ctx, f"max_src {max_src} + max_sum {max_sum} + marker exceeds the context {ctx}"
    log(f"[data] reference summaries: median {int(np.median(lens))} / p75 {int(np.percentile(lens, 75))} / p95 {int(np.percentile(lens, 95))} tokens "
        f"-> max_sum {max_sum}, max_src {max_src}, context {ctx}")
    selector = Selector()
    if cfg.select == "learned":
        sub = train[:cfg.selector_docs]
        selector.fit([e["doc"] for e in sub], [e["summary"] for e in sub])
    budget_words = int(max_src / TOK_PER_WORD)

    def prep(exs, labelled):
        src_txt = [select_text(e["doc"], budget_words, cfg.select, selector) for e in exs]
        src_ids = enc(src_txt)
        sum_ids = enc([e["summary"] for e in exs]) if labelled else [None] * len(exs)
        out = []
        for e, s, su in zip(exs, src_ids, sum_ids):
            x = Ex()
            x.id, x.src = e["id"], np.asarray(s[:max_src], dtype=np.int32)
            x.sum = np.asarray(su[:max_sum], dtype=np.int32) if su is not None else None
            x.ref = e.get("summary")
            x.head = " ".join(e["doc"].split()[:120])                                  # fallback text if a generation comes out empty
            out.append(x)
        return out

    data = {"train": prep(train, True), "dev": prep(dev, True), "test": prep(test, False), "max_sum": max_sum, "max_src": max_src,
            "sep": sep_ids, "idf": selector.idf, "default_idf": selector.default_idf}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(data, f, protocol=4)
    log(f"[data] prepared {len(data['train'])} train / {len(data['dev'])} dev / {len(data['test'])} test examples in {time.time() - t0:.0f}s")
    return data


def make_seq(ex, sep, eos):
    pre = [eos] + ex.src.tolist() + sep
    return pre, ex.sum.tolist() + [eos]


def collate6(exs, sep, eos, dev):
    seqs, starts = [], []
    for e in exs:
        pre, tgt = make_seq(e, sep, eos)
        seqs.append(pre + tgt)
        starts.append(len(pre))
    T = max(len(s) for s in seqs)
    ids = np.zeros((len(exs), T), dtype=np.int64)
    mask = np.zeros((len(exs), T), dtype=bool)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        ids[i, :len(s)] = s
        mask[i, st:len(s)] = True                         # positions of the summary tokens (+ final <eos>) = prediction targets
    return torch.from_numpy(ids).to(dev), torch.from_numpy(mask).to(dev)


# ============================================================================================
# 3. Model wrapper, loss, generation
# ============================================================================================
class SummNet(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm


def lm_hidden(lm, ids):
    """Causal hidden states (B, T, d); right padding is harmless because attention is causal."""
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).unsqueeze(0)
    x = lm.wte(ids)
    if lm.wpe is not None:
        x = x + lm.wpe(pos)
    rope = None
    if lm.rope_cos is not None:
        rope = (lm.rope_cos[pos].unsqueeze(1), lm.rope_sin[pos].unsqueeze(1))
    for blk in lm.blocks:
        x = blk(x, rope, None, None, None)
    return lm.norm_f(x)


def summary_logits(lm, ids, mask):
    h = lm_hidden(lm, ids)
    sel = mask[:, 1:]
    return lm.lm_head(h[:, :-1][sel]), ids[:, 1:][sel]


def kd_loss_tok(s_logits, t_logits, T):
    sl = F.log_softmax(s_logits.float() / T, dim=-1)
    tp = F.softmax(t_logits.float() / T, dim=-1)
    return (tp * (torch.log(tp.clamp_min(1e-9)) - sl)).sum(-1).mean() * (T * T)


@torch.no_grad()
def generate(model, prompts, dec, cfg: SCfg, dev, amp_dtype, eos, max_new):
    """prompts: list of token-id lists (eos + source + marker). Returns generated token-id lists (without the closing <eos>)."""
    B, L = len(prompts), max(len(p) for p in prompts)
    inp = torch.zeros((B, L), dtype=torch.long)
    km = torch.zeros((B, L), dtype=torch.bool)
    for b, p in enumerate(prompts):
        inp[b, L - len(p):] = torch.tensor(p, dtype=torch.long)
        km[b, L - len(p):] = True
    inp, km = inp.to(dev), km.to(dev)
    pos = (km.long().cumsum(1) - 1).clamp(min=0)
    last_pos = pos[:, -1:]
    cache = model.new_cache()
    temp, top_p, top_k = float(dec.get("temperature", 0.0)), float(dec.get("top_p", 1.0)), int(dec.get("top_k", 0))
    rep, nr = float(dec.get("rep", 1.0)), int(dec.get("nr", 0))
    min_new = int(dec.get("min_new", 0))
    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    gen = torch.zeros((B, 0), dtype=torch.long, device=dev)
    seqs, maps, done = [[] for _ in range(B)], [dict() for _ in range(B)], [False] * B
    ctxm = torch.autocast("cuda", dtype=amp_dtype) if dev == "cuda" else contextlib.nullcontext()
    with ctxm:
        logits, _ = model(inp, pos=pos, kmask=km, cache=cache, last_only=True)
        for t in range(max_new):
            lg = logits[:, -1].float().clone()
            if t < min_new:
                lg[:, eos] = -float("inf")
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
                    sl = sl.masked_fill((pr.cumsum(-1) - pr) > top_p, -float("inf"))
                    lg = torch.full_like(lg, -float("inf")).scatter(1, si, sl)
                nxt = torch.multinomial(lg.softmax(-1), 1, generator=g)[:, 0]
            nl = nxt.tolist()
            for b in range(B):
                if done[b]:
                    continue
                if nl[b] == eos:
                    done[b] = True
                    continue
                if nr >= 2 and len(seqs[b]) >= nr - 1:
                    maps[b].setdefault(tuple(seqs[b][-(nr - 1):]), []).append(nl[b])
                seqs[b].append(nl[b])
            if all(done):
                break
            gen = torch.cat([gen, nxt[:, None]], dim=1)
            km = torch.cat([km, torch.ones((B, 1), dtype=torch.bool, device=dev)], dim=1)
            last_pos = last_pos + 1
            logits, _ = model(nxt[:, None], pos=last_pos, kmask=km, cache=cache, last_only=True)
    return seqs


def gen_texts(model, tok, exs, sep, dec, cfg, dev, amp_dtype, eos, max_new):
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].src))
    out, bs, i = [None] * len(exs), cfg.gen_batch, 0
    while i < len(order):
        idx = order[i:i + bs]
        prompts = [[eos] + exs[j].src.tolist() + sep for j in idx]
        try:
            seqs = generate(model, prompts, dec, cfg, dev, amp_dtype, eos, max_new)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                gc.collect()
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                log(f"[gen] OOM -> batch size {bs}")
                continue
            raise
        for j, s in zip(idx, seqs):
            out[j] = postprocess(" ".join(tok.decode(s, skip_special_tokens=True).split()), dec.get("postproc", True)) or exs[j].head
        i += len(idx)
    return out


def postprocess(text: str, enabled: bool = True) -> str:
    """Redundancy reduction: drop sentences that are exact repeats of an earlier sentence."""
    if not enabled:
        return text
    seen, keep = set(), []
    for s in split_sents(text, glue=False):                  # no gluing here: short repeated sentences must stay separate
        k = " ".join(words_of(s))
        if k and k in seen:
            continue
        seen.add(k)
        keep.append(s)
    return " ".join(keep)


# ============================================================================================
# 4. Metrics (same libraries / definitions as the official evaluator)
# ============================================================================================
def _rouge_tokens():
    try:
        from rouge_score import tokenizers, rouge_scorer
        return rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True), tokenizers.DefaultTokenizer(use_stemmer=True)
    except ImportError:
        return None, None


def rouge_scores(preds, refs):
    scorer, tk = _rouge_tokens()
    r1 = rl = 0.0
    for pr, rf in zip(preds, refs):
        if scorer is not None:
            r1 += scorer.score(rf, pr)["rouge1"].fmeasure
            pt, rt = tk.tokenize(pr), tk.tokenize(rf)
        else:                                                                # approximation without stemming when rouge-score is missing
            pt, rt = words_of(pr), words_of(rf)
            c = sum((Counter(pt) & Counter(rt)).values())
            p_, r_ = c / max(1, len(pt)), c / max(1, len(rt))
            r1 += 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        if not pt or not rt:
            rl += 1.0 if (not pt and not rt) else 0.0
        else:
            z = P1.lcs_length(pt, rt)
            pp, rr = z / len(pt), z / len(rt)
            rl += 0.0 if pp + rr == 0 else 2 * pp * rr / (pp + rr)
    n = max(1, len(preds))
    return r1 / n, rl / n


def score_summ(preds, refs, cfg: SCfg, dev):
    r1, rl = rouge_scores(preds, refs)
    out = {"rouge1": r1, "rougeL": rl}
    if cfg.use_bertscore:
        try:
            out["bertscore"] = P1.bert_f1(preds, refs, dev)
        except Exception as e:
            log(f"[score] BERTScore unavailable ({type(e).__name__}: {e}); ranking by ROUGE-L instead")
            cfg.use_bertscore = False
    return out


def rep_rate(texts):
    """Share of word trigrams that are repeats inside the same summary (redundancy)."""
    rep = tot = 0
    for t in texts:
        w = words_of(t)
        tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
        tot += len(tri)
        rep += len(tri) - len(set(tri))
    return rep / max(1, tot)


# ============================================================================================
# 5. Training / evaluation driver
# ============================================================================================
def load_run6(cfg: SCfg, run_name: str, dev):
    from tokenizers import Tokenizer
    rd = Path(cfg.work_dir) / "sm_runs" / run_name
    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    saved = SCfg()
    apply_overrides(saved, {k: v for k, v in ck["cfg"].items() if k in {f.name for f in fields(SCfg)}})
    saved.work_dir = cfg.work_dir
    if cfg.local_data_dir:
        saved.local_data_dir = cfg.local_data_dir
    net = SummNet(P1.LM(ck["mc"]))
    J.apply_method4(net, saved)
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    return {"rd": rd, "ck": ck, "saved": saved, "net": net, "tok": Tokenizer.from_file(str(rd / "tokenizer.json"))}


@torch.no_grad()
def dev_loss(net, exs, sep, eos, cfg, dev, amp):
    net.eval()
    tot, cnt = 0.0, 0
    for b in range(0, len(exs), cfg.micro_bs):
        ids, mask = collate6(exs[b:b + cfg.micro_bs], sep, eos, dev)
        with amp():
            lg, tg = summary_logits(net.lm, ids, mask)
        tot += float(F.cross_entropy(lg.float(), tg, reduction="sum"))
        cnt += int(tg.numel())
    return tot / max(1, cnt)


def decode_sweep(net, tok, data, cfg: SCfg, dev, amp_dtype, eos):
    """Choose the decoding setup, then the generation length, on dev documents by BERTScore F1 (ROUGE-L if BERTScore is unavailable)."""
    exs = data["dev"][:cfg.sweep_n]
    refs = [e.ref for e in exs]
    max_sum, sep = data["max_sum"], data["sep"]
    key = lambda s: s.get("bertscore", s["rougeL"])
    table = []

    def trial(dec, max_new):
        t0 = time.time()
        preds = gen_texts(net.lm, tok, exs, sep, {**dec, "min_new": int(cfg.min_new_frac * max_new)}, cfg, dev, amp_dtype, eos, max_new)
        sc = score_summ(preds, refs, cfg, dev)
        row = {"decode": dec["name"], "max_new": max_new, **sc, "avg_words": float(np.mean([len(p.split()) for p in preds])),
               "rep_rate": rep_rate(preds), "sec": round(time.time() - t0, 1)}
        table.append(row)
        log(f"[sweep] {dec['name']:<22} max_new {max_new:>3} | " + " ".join(f"{k} {v:.4f}" for k, v in sc.items()) +
            f" | words {row['avg_words']:.0f} (ref {np.mean([len(r.split()) for r in refs]):.0f}) rep {row['rep_rate']:.3f} | {row['sec']}s")
        return row

    if cfg.decode:
        best = {"decode": cfg.decode, "max_new": max_sum}
    else:
        rows = [trial(d, max_sum) for d in cfg.sweep_grid]
        b = max(rows, key=key)
        best = {"decode": next(d for d in cfg.sweep_grid if d["name"] == b["decode"]), "max_new": max_sum}
        for sc_ in cfg.len_scales:                                          # generation length control
            if sc_ >= 1.0:
                continue
            r = trial(best["decode"], max(16, int(sc_ * max_sum)))
            if key(r) > key(b):
                b, best["max_new"] = r, r["max_new"]
        log(f"[sweep] BEST: {best['decode']['name']} with max_new {best['max_new']}")
    P1.jdump(table, run_dir6(cfg) / "sweep.json")
    return best, table


def run_train6(cfg: SCfg):
    rd = run_dir6(cfg)
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
            raise SystemExit('method "kd" needs "teacher_run": the name of a finished summarisation run')
        lm, mc = N.make_student(lm, mc, cfg)
    eos = tok.token_to_id(P1.EOS)
    data = build_data(cfg, tok, mc["ctx"])
    tr, dv, sep = data["train"], data["dev"], data["sep"]
    teacher, teacher_params = None, 0
    if cfg.method == "kd":
        trd = Path(cfg.work_dir) / "sm_runs" / cfg.teacher_run
        if not (trd / "best.pt").exists():
            raise SystemExit(f"teacher run not found: {trd}/best.pt (train it first)")
        if (trd / "tokenizer.json").read_bytes() != (rd / "tokenizer.json").read_bytes():
            raise SystemExit("teacher and student use different tokenizers")
        tr_ = load_run6(cfg, cfg.teacher_run, dev)
        teacher = tr_["net"]
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_params = sum(p.numel() for p in teacher.parameters())
        log(f"[kd] teacher {cfg.teacher_run}: arch={tr_['ck']['mc']['arch']} method={tr_['saved'].method} params={teacher_params / 1e6:.1f}M")

    net = SummNet(lm)
    J.apply_method4(net, cfg)
    net.to(dev)
    n_total = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"[model] total params {n_total / 1e6:.1f}M | trainable {n_train / 1e6:.3f}M | method={cfg.method} | max_src {data['max_src']} max_sum {data['max_sum']}")

    peft = cfg.method in ("lora", "lorma", "adapter", "bitfit")
    base_lr, wd0 = (cfg.peft_lr, 0.0) if peft else (cfg.lr, cfg.weight_decay)
    nb = mc["n_layer"]

    def lr_scale(name):
        if peft or cfg.llrd >= 1.0:
            return 1.0
        m_ = re.match(r"lm\.blocks\.(\d+)\.", name)
        if m_:
            return cfg.llrd ** (nb - 1 - int(m_.group(1)))
        return cfg.llrd ** nb if name.startswith(("lm.wte", "lm.wpe", "lm.lm_head")) else 1.0

    buckets = {}
    for n_, p_ in net.named_parameters():
        if p_.requires_grad:
            buckets.setdefault((round(base_lr * lr_scale(n_), 12), wd0 if p_.dim() >= 2 else 0.0), []).append(p_)
    opt = torch.optim.AdamW([{"params": ps, "lr": b_, "base": b_, "weight_decay": wd} for (b_, wd), ps in buckets.items()], betas=(0.9, 0.98), eps=1e-8)
    scaler = P1.make_scaler(dev == "cuda" and amp_dtype == torch.float16)
    ema = J.EMA(net, cfg.ema_decay) if cfg.ema_decay > 0 else None

    rng = np.random.default_rng(cfg.seed)
    dev_sub = dv[:cfg.dev_loss_docs]
    steps_per_epoch = max(1, math.ceil(len(tr) / (cfg.micro_bs * cfg.accum)))
    total_steps = cfg.epochs * steps_per_epoch
    budget = cfg.time_budget_min * 60.0
    log(f"[train] {len(tr)} examples, {steps_per_epoch} steps/epoch x {cfg.epochs} epochs ({cfg.micro_bs * cfg.accum} sequences per step); time budget {cfg.time_budget_min} min")
    best, step, bad, t_start, stop = {"loss": 1e9}, 0, 0, time.time(), False
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logf = open(rd / "log.jsonl", "a")

    def do_eval(tag, loss_val):
        nonlocal best, bad
        with (ema.swapped() if ema is not None else contextlib.nullcontext()):
            dl = dev_loss(net, dev_sub, sep, eos, cfg, dev, amp)
            log(f"[{tag}] train loss {loss_val:.4f} | dev loss {dl:.4f} (perplexity {math.exp(min(dl, 20)):.1f}) | {time.time() - t_start:.0f}s")
            logf.write(json.dumps({"tag": tag, "step": step, "train_loss": loss_val, "dev_loss": dl}) + "\n")
            logf.flush()
            if dl < best["loss"]:
                best, bad = {"loss": dl, "epoch": tag}, 0
                torch.save({"net": net.state_dict(), "mc": mc, "cfg": jd}, rd / "best.pt")
            else:
                bad += 1
        net.train()

    for ep in range(cfg.epochs):
        net.train()
        order = rng.permutation(len(tr))
        run_loss, n_b, early = 0.0, 0, False
        for g0 in range(0, len(order), cfg.micro_bs * cfg.accum):
            prog = max(step / total_steps, (time.time() - t_start) / budget if budget > 0 else 0.0)
            if prog >= 1.0:
                stop = True
                break
            for g in opt.param_groups:
                g["lr"] = J.lr_prog(g["base"], prog, cfg.warmup_frac)
            group = order[g0:g0 + cfg.micro_bs * cfg.accum]
            n_tok = 0
            for m0 in range(0, len(group), cfg.micro_bs):
                ids, mask = collate6([tr[i] for i in group[m0:m0 + cfg.micro_bs]], sep, eos, dev)
                with amp():
                    lg, tg = summary_logits(net.lm, ids, mask)
                    if teacher is not None:
                        with torch.no_grad():
                            tl, _ = summary_logits(teacher.lm, ids, mask)
                loss = F.cross_entropy(lg.float(), tg)
                if teacher is not None:
                    loss = cfg.kd_alpha * loss + (1.0 - cfg.kd_alpha) * kd_loss_tok(lg, tl, cfg.kd_temp)
                scaler.scale(loss * (len(ids) / len(group))).backward()
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
            log(f"[train] early stopping (best: {best['epoch']})")
            break
        if stop:
            log("[train] time budget reached")
            break
    logf.close()
    train_s = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0.0

    ck = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"])
    net.eval()
    best_dec, table = decode_sweep(net, tok, data, cfg, dev, amp_dtype, eos)
    dec = {**best_dec["decode"], "min_new": int(cfg.min_new_frac * best_dec["max_new"])}
    exs = dv[:cfg.final_dev_n]
    t0 = time.time()
    preds = gen_texts(net.lm, tok, exs, sep, dec, cfg, dev, amp_dtype, eos, best_dec["max_new"])
    gen_s = len(exs) / max(1e-9, time.time() - t0)
    refs = [e.ref for e in exs]
    sc = score_summ(preds, refs, cfg, dev)
    raw = score_summ(gen_texts(net.lm, tok, exs[:cfg.sweep_n], sep, {**dec, "postproc": False}, cfg, dev, amp_dtype, eos, best_dec["max_new"]),
                     refs[:cfg.sweep_n], cfg, dev)
    log(f"[dev] {len(exs)} documents with '{dec['name']}' (max_new {best_dec['max_new']}): " + " ".join(f"{k} {v:.4f}" for k, v in sc.items()) +
        f" | without redundancy removal (first {cfg.sweep_n}): " + " ".join(f"{k} {v:.4f}" for k, v in raw.items()))
    log(f"[dev] summary length {np.mean([len(p.split()) for p in preds]):.0f} words vs reference {np.mean([len(r.split()) for r in refs]):.0f} | repeated trigrams {rep_rate(preds):.3f}")
    for i in range(min(2, len(exs))):
        log(f"[dev] example {exs[i].id}\n   REF: {refs[i][:350]}\n   GEN: {preds[i][:350]}")
    ck["decode"], ck["max_new"] = dec, best_dec["max_new"]
    torch.save(ck, rd / "best.pt")
    P1.jdump({e.id: p for e, p in zip(exs, preds)}, rd / "dev_preds.json")
    P1.jdump({e.id: e.ref for e in exs}, rd / "dev_refs.json")
    variant = cfg.lorma_variant if cfg.method == "lorma" else ""
    P1.jdump({"best_epoch": best["epoch"], "dev_loss": best["loss"], "scores": sc, "scores_no_postproc": raw, "decode": dec, "max_new": best_dec["max_new"],
              "params_total": n_total, "params_trainable": n_train, "train_seconds": train_s, "peak_gpu_mb": peak, "gen_docs_per_s": gen_s,
              "p1_arch": mc["arch"], "method": cfg.method, "teacher_run": cfg.teacher_run, "teacher_params": teacher_params,
              "avg_words": float(np.mean([len(p.split()) for p in preds])), "ref_words": float(np.mean([len(r.split()) for r in refs])),
              "rep_rate": rep_rate(preds), "max_src": data["max_src"], "max_sum": data["max_sum"]}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": mc["arch"], "method": cfg.method + (f"-{variant}" if variant else ""), "select": cfg.select,
                  "params_total_M": round(n_total / 1e6, 2), "params_trainable_M": round(n_train / 1e6, 3),
                  "dev_bertscore": round(sc.get("bertscore", float("nan")), 4), "dev_rouge1": round(sc["rouge1"], 4), "dev_rougeL": round(sc["rougeL"], 4),
                  "dev_loss": round(best["loss"], 4), "decode": dec["name"], "max_new": best_dec["max_new"],
                  "avg_words": round(float(np.mean([len(p.split()) for p in preds])), 1), "ref_words": round(float(np.mean([len(r.split()) for r in refs])), 1),
                  "rep_rate": round(rep_rate(preds), 4), "train_s": round(train_s), "peak_gpu_mb": round(peak), "gen_docs_s": round(gen_s, 2),
                  "teacher_run": cfg.teacher_run})
    if cfg.predict_after:
        run_predict6(cfg, net=net, tok=tok, data=data, dec=dec, max_new=best_dec["max_new"])


def write_submission6(cfg: SCfg, rd: Path, ids, texts):
    out = [{"id": i, "summary": t if t.strip() else "The appeal is dismissed."} for i, t in zip(ids, texts)]
    assert len(out) == len(set(ids)) and all(o["summary"].strip() for o in out)
    rd.mkdir(parents=True, exist_ok=True)
    pj = rd / "prediction.json"
    P1.jdump(out, pj)
    zp = rd / "submission.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    hist = Path(cfg.work_dir) / "sm_submissions"
    hist.mkdir(parents=True, exist_ok=True)
    shutil.copy(zp, hist / f"{cfg.run}_{time.strftime('%m%d_%H%M%S')}.zip")
    log(f"[predict] wrote {pj} ({len(out)} summaries, {np.mean([len(o['summary'].split()) for o in out]):.0f} words on average) and {zp}")
    return str(zp)


def run_predict6(cfg: SCfg, net=None, tok=None, data=None, dec=None, max_new=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype, _ = P1.amp_setup(dev)
    rd = run_dir6(cfg)
    if net is None:
        r = load_run6(cfg, cfg.run, dev)
        net, tok, saved = r["net"], r["tok"], r["saved"]
        dec, max_new = r["ck"]["decode"], r["ck"]["max_new"]
        data = build_data(saved, tok, r["ck"]["mc"]["ctx"])
    eos = tok.token_to_id(P1.EOS)
    te = data["test"]
    t0 = time.time()
    texts = gen_texts(net.lm, tok, te, data["sep"], dec, cfg, dev, amp_dtype, eos, max_new)
    log(f"[predict] generated {len(te)} test summaries in {time.time() - t0:.0f}s with '{dec['name']}' (max_new {max_new})")
    P1.jdump({e.id: t for e, t in zip(te, texts)}, rd / "test_preds.json")
    return write_submission6(cfg, rd, [e.id for e in te], texts)


# ============================================================================================
# 6. Consensus (minimum-Bayes-risk style) over several runs
# ============================================================================================
def _f1_counts(a, b):
    if not a or not b:
        return 0.0
    c = sum((a & b).values())
    p_, r_ = c / sum(b.values()), c / sum(a.values())
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0


def consensus(cands):
    """cands: list of summaries of ONE document from different runs -> index of the summary most similar (unigram + bigram F1) to the others."""
    feats = []
    for t in cands:
        w = words_of(t)
        feats.append((Counter(w), Counter(zip(w, w[1:]))))
    best, best_s = 0, -1.0
    for i in range(len(cands)):
        s = np.mean([(_f1_counts(feats[i][0], feats[j][0]) + _f1_counts(feats[i][1], feats[j][1])) / 2 for j in range(len(cands)) if j != i]) if len(cands) > 1 else 0.0
        if s > best_s:
            best, best_s = i, s
    return best


def run_mbr6(cfg: SCfg):
    root = Path(cfg.work_dir) / "sm_runs"
    if cfg.ens_runs:
        names = list(cfg.ens_runs)
    elif cfg.ens_auto:
        names = [f.parent.name for f in sorted(root.glob("*/test_preds.json")) if (f.parent / "dev_preds.json").exists()
                 and (f.parent.name.startswith(cfg.ens_prefix) if cfg.ens_prefix else not f.parent.name.startswith(("selftest", "check")))]
    else:
        names = []
    if len(names) < 3:
        raise SystemExit('consensus needs >= 3 finished runs: set "ens_runs": [...] or "ens_auto": true')
    dev_p = {n: P1.jload(root / n / "dev_preds.json") for n in names}
    te_p = {n: P1.jload(root / n / "test_preds.json") for n in names}
    refs = P1.jload(root / names[0] / "dev_refs.json")
    ids_dev = [i for i in refs if all(i in dev_p[n] for n in names)]
    ids_te = list(te_p[names[0]])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    score = lambda texts: score_summ(texts, [refs[i] for i in ids_dev], cfg, dev)
    key = lambda s: s.get("bertscore", s["rougeL"])
    single = {n: score([dev_p[n][i] for i in ids_dev]) for n in names}
    for n, s in sorted(single.items(), key=lambda kv: -key(kv[1])):
        log(f"[mbr] single run {n:<28} " + " ".join(f"{k} {v:.4f}" for k, v in s.items()))
    order = sorted(names, key=lambda n: -key(single[n]))
    best_k, best_s = None, None
    for k in range(3, min(len(order), cfg.ens_max) + 1):
        mem = order[:k]
        picks = [mem[consensus([dev_p[n][i] for n in mem])] for i in ids_dev]
        s = score([dev_p[p][i] for p, i in zip(picks, ids_dev)])
        log(f"[mbr] consensus of the top {k} runs {mem}: " + " ".join(f"{a} {b:.4f}" for a, b in s.items()))
        if best_s is None or key(s) > key(best_s):
            best_k, best_s = k, s
    members = order[:best_k]
    log(f"[mbr] members: {members}  (best single run: {key(single[order[0]]):.4f} -> consensus {key(best_s):.4f})")
    texts = [te_p[members[consensus([te_p[n][i] for n in members])]][i] for i in ids_te]
    rd = run_dir6(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    P1.jdump({"members": members, "scores": best_s, "single": single}, rd / "dev_report.json")
    log_csv(cfg, {"run": cfg.run, "p1_arch": "consensus", "method": "consensus", "dev_bertscore": round(best_s.get("bertscore", float("nan")), 4),
                  "dev_rouge1": round(best_s["rouge1"], 4), "dev_rougeL": round(best_s["rougeL"], 4), "teacher_run": "+".join(members)})
    return write_submission6(cfg, rd, ids_te, texts)


# ============================================================================================
# 7. Diagnostics
# ============================================================================================
def diagnose6(cfg: SCfg):
    tok, _, mc = N.load_p1(cfg)
    allx = read_split(cfg, cfg.train_file, True, cfg.max_train_docs)
    test = read_split(cfg, cfg.test_file, False)
    enc = lambda texts: [len(e.ids) for b in range(0, len(texts), 128) for e in tok.encode_batch(texts[b:b + 128])]
    sub = allx[:1500]
    dl, sl = np.array(enc([e["doc"] for e in sub[:600]])), np.array(enc([e["summary"] for e in sub]))
    tl = np.array(enc([e["doc"] for e in test[:300]]))
    q = lambda a: f"median {int(np.median(a))} p75 {int(np.percentile(a, 75))} p95 {int(np.percentile(a, 95))} max {int(a.max())}"
    log(f"[check] {len(allx)} training pairs, {len(test)} test documents; context of your Phase-1 model: {mc['ctx']} tokens")
    log(f"[check] document tokens: {q(dl)} | summary tokens: {q(sl)} | test document tokens: {q(tl)}")
    log(f"[check] compression: summaries are {100 * np.median(sl) / np.median(dl):.1f}% of a document; a document is {np.median(dl) / mc['ctx']:.1f}x your context window")
    log("[check] NOTE: BERTScore truncates candidate and reference at ~512 RoBERTa tokens (~350-400 words); text beyond that only counts for ROUGE.")
    max_sum = cfg.max_sum or int(np.clip(np.percentile(sl, 75), 128, 520))
    max_src = cfg.max_src or (mc["ctx"] - max_sum - len(tok.encode(SEP_TEXT).ids) - 8)
    log(f"[check] the model will read max_src={max_src} source tokens and write up to max_sum={max_sum} summary tokens")
    budget = int(max_src / TOK_PER_WORD)
    fit = sub[:1500]
    selector = Selector().fit([e["doc"] for e in fit[:cfg.selector_docs]], [e["summary"] for e in fit[:cfg.selector_docs]])
    held = allx[1500:1800] if len(allx) > 1800 else allx[-200:]
    for mode in ("lead", "head_tail", "uniform", "learned"):
        cov = [ref_coverage(select_text(e["doc"], budget, mode, selector), e["summary"], selector.idf, selector.default_idf) for e in held]
        log(f"[check] information kept after compression, select='{mode}': {100 * np.mean(cov):.1f}% of the reference's (idf-weighted) words survive in the {budget}-word source")
    log("[check] pick the mode with the highest value as cfg 'select' (learned is the default)")


# ============================================================================================
# 8. Self-test
# ============================================================================================
def _synthetic(tmp, rng):
    topics = ["tax", "contract", "land", "murder", "bail", "dowry", "company", "labour"]
    filler = "the learned counsel for the appellant submitted that the impugned order was passed without hearing and the court considered the evidence on record".split()

    def doc():
        t = rng.choice(topics)
        key = [f"The dispute concerns {t} and the court examined section {rng.randint(1, 99)} of the {t} act in detail.",
               f"The appeal in the matter of {t} is {'allowed' if rng.random() < 0.5 else 'dismissed'} with costs."]
        sents = []
        for _ in range(rng.randint(15, 60)):
            sents.append(" ".join(rng.choice(filler) for _ in range(rng.randint(8, 20))).capitalize() + ".")
        sents.insert(rng.randint(0, len(sents)), key[0])
        sents.append(key[1])
        return " ".join(sents), " ".join(key)

    os.makedirs(tmp, exist_ok=True)
    texts = []
    for split, n in (("train", 60), ("test", 10)):
        recs = []
        for i in range(n):
            d, s = doc()
            texts += [d, s]
            r = {"id": f"{split}_{i}", "document": d}
            if split == "train":
                r["summary"] = s
            recs.append(r)
        json.dump(recs, open(os.path.join(tmp, f"{split}.json"), "w"))
    return texts


def selftest() -> bool:
    import tempfile
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    ok = True
    rng = random.Random(0)
    tmp = tempfile.mkdtemp(prefix="cs781_sm_selftest_")
    dd = os.path.join(tmp, "data")
    texts = _synthetic(dd, rng)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([" ".join(t.split()) for t in texts] + [SEP_TEXT] * 5, trainers.BpeTrainer(
        vocab_size=600, special_tokens=[P1.EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    V = P1.pad64(tok.get_vocab_size())
    eos = tok.token_to_id(P1.EOS)

    log("=== 1. sentences, selection, text helpers, consensus ===")
    doc = " ".join([f"Sentence number {i} is about the matter." for i in range(40)] + ["The appeal is dismissed."])
    ss = split_sents(doc)
    g1 = len(ss) == 41 and ss[-1] == "The appeal is dismissed."
    sel = {m: select_text(doc, 60, m, None) for m in ("lead", "head_tail", "uniform", "learned")}
    g2 = all(len(v.split()) <= 70 and v for v in sel.values()) and sel["lead"].startswith("Sentence number 0") and "dismissed" in sel["head_tail"]
    pp = postprocess("The court held. The court held. It was allowed.")
    g3 = pp == "The court held. It was allowed."
    cands = ["the appeal is dismissed with costs", "the appeal is dismissed with costs here", "completely unrelated text about weather"]
    g4 = consensus(cands) in (0, 1) and rep_rate(["a b c a b c a b c"]) > 0
    r1, rl = rouge_scores(["the court dismissed the appeal"], ["the court dismissed the appeal"])
    g5 = abs(r1 - 1.0) < 1e-9 and abs(rl - 1.0) < 1e-9
    ok &= all([g1, g2, g3, g4, g5])
    log(f"  sentence splitting {'PASS' if g1 else 'FAIL'} | selection modes within budget {'PASS' if g2 else 'FAIL'} | redundancy removal {'PASS' if g3 else 'FAIL'} | "
        f"consensus / repetition rate {'PASS' if g4 else 'FAIL'} | ROUGE of identical texts = 1 {'PASS' if g5 else 'FAIL'}")

    log("=== 2. generation with KV cache / left padding == step-by-step full forward (every architecture) ===")
    for arch in P1.ARCH_PRESETS:
        try:
            torch.manual_seed(0)
            lm = P1.LM(N._tiny_mc(arch, V)).eval()
            cfgg = SCfg()
            prompts = [[eos] + list(range(5, 17)), [eos] + list(range(20, 25))]
            got = generate(lm, prompts, {"temperature": 0.0}, cfgg, "cpu", None, eos, 6)
            ref = []
            with torch.no_grad():
                for p in prompts:
                    seq, out = list(p), []
                    for _ in range(6):
                        t = int(lm(torch.tensor([seq]))[0][0, -1].argmax())
                        if t == eos:
                            break
                        out.append(t)
                        seq.append(t)
                    ref.append(out)
            good = got == ref
            ok &= good
            log(f"  {arch:<7} generated {got[0][:5]}...  {'PASS' if good else 'FAIL'}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  {arch:<7} FAIL: {type(e).__name__}: {e}")
            ok = False

    log("=== 3. adapters at identity / trainable subsets, summary-only loss ===")
    p1x = os.path.join(tmp, "p1_x")
    os.makedirs(p1x, exist_ok=True)
    tok.save(os.path.join(p1x, "tokenizer.json"))
    cfg0 = apply_overrides(SCfg(), {"local_data_dir": dd, "work_dir": os.path.join(tmp, "work0"), "p1_dir": p1x, "max_src": 40, "max_sum": 24,
                                    "dev_docs": 8, "selector_docs": 40})
    data0 = build_data(cfg0, tok, 128)
    data1 = build_data(cfg0, tok, 128)                                                  # second call from the cache
    g6 = len(data0["train"]) == 52 and len(data0["dev"]) == 8 and len(data0["test"]) == 10 and len(data1["train"]) == 52 and \
        all(len(e.src) <= 40 and len(e.sum) <= 24 for e in data0["train"])
    ids, mask = collate6(data0["train"][:2], data0["sep"], eos, "cpu")
    g7 = all(int(mask[b].sum()) == len(data0["train"][b].sum) + 1 for b in range(2)) and not bool(mask[:, 0].any())
    ok &= g6 and g7
    log(f"  data preparation + cache {'PASS' if g6 else 'FAIL'} | loss mask = summary tokens + <eos> only {'PASS' if g7 else 'FAIL'}")
    for arch in P1.ARCH_PRESETS:
        for meth, var in (("adapter", "plus"), ("lora", "plus"), ("lorma", "plus"), ("lorma", "pi"), ("bitfit", "plus"), ("partial", "plus")):
            tag = f"{arch}/{meth}" + (f"-{var}" if meth == "lorma" else "")
            try:
                torch.manual_seed(0)
                c = apply_overrides(SCfg(), {"method": meth, "lorma_variant": var, "peft_r": 4, "peft_alpha": 8.0, "peft_dropout": 0.0, "adapter_dim": 8, "freeze_layers": 1})
                net = SummNet(P1.LM(N._tiny_mc(arch, V))).eval()
                with torch.no_grad():
                    ref = lm_hidden(net.lm, ids)
                J.apply_method4(net, c)
                net.eval()
                with torch.no_grad():
                    out = lm_hidden(net.lm, ids)
                diff = (ref - out).abs().max().item()
                n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
                n_all = sum(p.numel() for p in net.parameters())
                net.train()
                lg, tg = summary_logits(net.lm, ids, mask)
                F.cross_entropy(lg.float(), tg).backward()
                grads = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.parameters() if p.requires_grad)
                good = diff < 1e-4 and n_tr < n_all and grads
                ok &= good
                log(f"  {tag:<18} |adapted - base| = {diff:.1e}  trainable {n_tr:,}/{n_all:,}  {'PASS' if good else 'FAIL'}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"  {tag:<18} FAIL: {type(e).__name__}: {e}")
                ok = False

    log("=== 4. end-to-end: train -> decoding sweep -> test summaries, every architecture / method; consensus ===")
    p1dirs = {}
    for arch in P1.ARCH_PRESETS:
        p1 = os.path.join(tmp, f"p1_{arch}")
        os.makedirs(p1, exist_ok=True)
        torch.save({"model": P1.LM(N._tiny_mc(arch, V)).state_dict(), "mc": N._tiny_mc(arch, V), "state": {"step": 0}}, os.path.join(p1, "ckpt_best.pt"))
        tok.save(os.path.join(p1, "tokenizer.json"))
        p1dirs[arch] = p1
    base = {"work_dir": os.path.join(tmp, "work"), "local_data_dir": dd, "max_src": 40, "max_sum": 24, "dev_docs": 8, "selector_docs": 40, "epochs": 2,
            "micro_bs": 4, "accum": 1, "lr": 1e-3, "peft_lr": 3e-3, "peft_r": 4, "peft_alpha": 8.0, "adapter_dim": 8, "ema_decay": 0.9, "time_budget_min": 5,
            "eval_every_steps": 0, "dev_loss_docs": 8, "gen_batch": 4, "sweep_n": 4, "final_dev_n": 6, "use_bertscore": False, "len_scales": [0.6],
            "sweep_grid": [{"name": "greedy", "temperature": 0.0, "rep": 1.0}, {"name": "greedy_rp1.2_nr3", "temperature": 0.0, "rep": 1.2, "nr": 3}]}
    cases = [(f"full_{a}", a, {"method": "full"}) for a in P1.ARCH_PRESETS] + [
        ("lorma_plus", "gpt2", {"method": "lorma", "lorma_variant": "plus"}), ("lorma_pi", "retnet", {"method": "lorma", "lorma_variant": "pi"}),
        ("lora", "llama", {"method": "lora"}), ("adapter", "moe", {"method": "adapter"}), ("bitfit", "gpt2", {"method": "bitfit"}),
        ("partial", "llama", {"method": "partial", "freeze_layers": 1}), ("lead", "llama", {"method": "full", "select": "lead"}),
        ("kd", "gpt2", {"method": "kd", "teacher_run": "selftest_full_llama", "student_layers": 1})]
    for tag, arch, extra in cases:
        try:
            cfg = apply_overrides(SCfg(), {**base, **extra, "run": f"selftest_{tag}", "p1_dir": p1dirs[arch]})
            run_train6(cfg)
            with zipfile.ZipFile(run_dir6(cfg) / "submission.zip") as z:
                assert z.namelist() == ["prediction.json"]
                pr = json.loads(z.read("prediction.json"))
            assert isinstance(pr, list) and [x["id"] for x in pr] == [f"test_{i}" for i in range(10)]
            assert all(set(x) == {"id", "summary"} and isinstance(x["summary"], str) and x["summary"].strip() for x in pr)
            rep = P1.jload(run_dir6(cfg) / "dev_report.json")
            if extra["method"] in ("lora", "lorma", "adapter", "bitfit", "partial"):
                assert rep["params_trainable"] < rep["params_total"]
            if extra["method"] == "kd":
                assert rep["params_total"] < rep["teacher_params"]
            log(f"  e2e {tag:<12} PASS  (dev ROUGE-L {rep['scores']['rougeL']:.3f}, {rep['avg_words']:.0f} words, trainable {rep['params_trainable']:,}/{rep['params_total']:,})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  e2e {tag:<12} FAIL: {type(e).__name__}: {e}")
            ok = False
    try:
        cfg = apply_overrides(SCfg(), {**base, "run": "selftest_mbr", "ens_auto": True, "ens_prefix": "selftest_full_"})
        zp = run_mbr6(cfg)
        with zipfile.ZipFile(zp) as z:
            pr = json.loads(z.read("prediction.json"))
        assert len(pr) == 10 and all(x["summary"].strip() for x in pr)
        log("  consensus stage PASS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"  consensus stage FAIL: {type(e).__name__}: {e}")
        ok = False
    shutil.rmtree(tmp, ignore_errors=True)
    log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


# ============================================================================================
# 9. CLI
# ============================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["selftest", "check", "train", "predict", "mbr"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--cfg", default="", help="JSON string or JSON file with config overrides")
    args = ap.parse_args(argv)
    if args.stage == "selftest":
        return 0 if selftest() else 1
    cfg = SCfg()
    ov = {}
    if args.cfg:
        ov = json.loads(args.cfg) if args.cfg.strip().startswith("{") else P1.jload(args.cfg)
    if args.run:
        ov["run"] = args.run
    apply_overrides(cfg, ov)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    {"check": diagnose6, "train": run_train6, "predict": run_predict6, "mbr": run_mbr6}[args.stage](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
