# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from verl import DataProto
import random
# from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from functools import partial
from verl.utils.reward_score import prime_math

# ---------------------------------------------------------
# Whitening: identical to SOTA-old
# ---------------------------------------------------------
def amplify_reward(r, alpha=1.8):
    return np.exp(alpha * r)
# def exp_norm(r, alpha=2, floor=0.05):
def exp_norm(r, alpha=1.8):
    r = np.clip(r, 0.0, 1.0)
    return np.exp(alpha * r)
    # y = (np.exp(alpha * r) - 1.0) / (np.exp(alpha) - 1.0)  # in [0,1]
    # return floor + (1.0 - floor) * y
def clip_01(r):
    r = np.clip(r, 0.0, 1.0)
    return r
def _token_whiten(H, eps=1e-8):
    T, d = H.shape
    if T <= 1:
        return H - H.mean(dim=0, keepdim=True)

    Hc = H - H.mean(dim=0, keepdim=True)
    Ct = (Hc @ Hc.T) / float(d)

    eigvals, eigvecs = torch.linalg.eigh(Ct)
    eigvals = eigvals.clamp_min(eps)

    E = eigvals.rsqrt().unsqueeze(0)
    W = eigvecs * E
    Ct_inv_sqrt = W @ eigvecs.T

    return Ct_inv_sqrt @ Hc

def token_center(H):
    return H - H.mean(dim=0, keepdim=True)

# ---------------------------------------------------------
# SVD: identical to SOTA-old
# ---------------------------------------------------------
def randomized_svd_topr(H, r, oversample=8, n_iter=2):
    T, d = H.shape
    k = r + oversample

    Omega = torch.randn(d, k, device=H.device, dtype=H.dtype)
    Y = H @ Omega

    for _ in range(n_iter):
        Y = H @ (H.T @ Y)
        Y, _ = torch.linalg.qr(Y, mode='reduced')

    Q, _ = torch.linalg.qr(Y, mode='reduced')
    B = Q.T @ H

    _, _, Vh = torch.linalg.svd(B, full_matrices=False)
    V_r = Vh[:r].T   # [d, r]

    return V_r


# =========================================================
#   CoT-conditioned SOTA-old (per-sample correctness manifold)
# =========================================================
def subspace_energy_overlap_per_sample(
    golden_hidden,       # [S, D] ( golden final answer hidden)
    pred_hidden,         # [S, D] ( predicted final answer)
    golden_mask,         # [S]
    pred_mask,           # [S]
    r=32,
    k=None,
    eps=1e-8,
):


    # --------------------------------------------------
    # 1. extract golden answer tokens for sample b
    # --------------------------------------------------
    # H_all_gold = golden_hidden        # [S, D]
    H_pred = pred_hidden[pred_mask]             # [Tp, D]
    if H_pred.size(0) == 0:
        return 0
    H_gold = golden_hidden[golden_mask]       # [Tg, D]
    if H_gold.size(0) == 0:
        return 0

    # (optionally last-k tokens)
    if k and H_gold.size(0) > k:
        H_gold = H_gold[-k:]

    # whiten golden
    H_gold_w = _token_whiten(H_gold)
    # correctness subspace (per-sample)
    V_gold = randomized_svd_topr(H_gold_w, r)   # [D, r]

    # golden energy
    gold_proj = H_gold_w @ V_gold               # [Tg, r]
    gold_energy = gold_proj.pow(2).sum().item()

    if gold_energy < eps:
        return 0

    # --------------------------------------------------
    # 2. extract predicted answer tokens for sample b
    # --------------------------------------------------


    if k and H_pred.size(0) > k:
        H_pred = H_pred[-k:]

    # whiten predicted
    H_pred_w = _token_whiten(H_pred)
    # projection energy
    pred_proj = H_pred_w @ V_gold
    pred_energy = pred_proj.pow(2).sum().item()

    score = pred_energy / (gold_energy + eps)
    return score
def subspace_energy_overlap_topk(
    golden_hidden,       # [S, D] ( golden final answer hidden)
    pred_hidden,         # [S, D] ( predicted final answer)
    golden_mask,         # [S]
    pred_mask,           # [S]
    r=32,
    k=5,
    eps=1e-8,
):

    # H_all_gold = golden_hidden        # [S, D]
    H_pred = pred_hidden[pred_mask]             # [Tp, D]
    if H_pred.size(0) == 0:
        return 0
    
    H_gold = golden_hidden[golden_mask]       # [Tg, D]
    if H_gold.size(0) == 0:
        return 0

    # whiten golden
    H_gold_w = _token_whiten(H_gold)
    # correctness subspace (per-sample)
    V_gold = randomized_svd_topr(H_gold_w, r)   # [D, r]
    # golden energy
    gold_proj = H_gold_w @ V_gold               # [Tg, r]
    gold_energy = gold_proj.pow(2).sum(dim=1)
    gold_energy = gold_energy.topk(min(k,H_gold.size(0))).values.mean().item()
    if gold_energy < eps:
        return 0
    # --------------------------------------------------
    # 2. extract predicted answer tokens for sample b
    # --------------------------------------------------
    # whiten predicted
    H_pred_w = _token_whiten(H_pred)
    # projection energy
    pred_proj = H_pred_w @ V_gold
    pred_energy = pred_proj.pow(2).sum(dim=1)
    pred_energy = pred_energy.topk(min(k,H_pred.size(0))).values.mean().item()
    score = pred_energy / (gold_energy + eps)
    return score

def _token_norm(H, eps=1e-8):
    return H / (H.norm(dim=-1, keepdim=True) + eps)
def subspace_energy_overlap_topk_white(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # token normalization (length-agnostic)
    try:
        H_gold_n = _token_whiten(H_gold, eps=eps)  # [Tg, D]
        H_pred_n = _token_whiten(H_pred, eps=eps)  # [Tp, D]
    except:
        return 0.0

    try:
        _, _, Vh = torch.linalg.svd(H_gold_n, full_matrices=False)
    except:
        return 0.0

    V_gold = Vh.T  # [D, r_eff]

    # golden projection energy
    gold_proj = H_gold_n @ V_gold                      # [Tg, r_eff]
    gold_energy = gold_proj.pow(2).sum(dim=1)          # [Tg]
    gold_energy = gold_energy.topk(min(k, Tg)).values.mean().item()

    if gold_energy < eps:
        return 0.0

    # predicted projection energy
    pred_proj = H_pred_n @ V_gold                      # [Tp, r_eff]
    pred_energy = pred_proj.pow(2).sum(dim=1)          # [Tp]
    pred_energy = pred_energy.topk(min(k, Tp)).values.mean().item()

    score = pred_energy / (gold_energy + eps)
    return score

def subspace_energy_overlap_topk_white_v2(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # token normalization (length-agnostic)
    try:
        H_gold_n = _token_whiten(H_gold, eps=eps)  # [Tg, D]
        H_pred_n = _token_whiten(H_pred, eps=eps)  # [Tp, D]
    except:
        return 0.0

    try:
        _, _, Vh = torch.linalg.svd(H_gold_n, full_matrices=False)
    except:
        return 0.0

    V_gold = Vh.T  # [D, r_eff]

    # golden projection energy
    gold_proj = H_gold_n @ V_gold                      # [Tg, r_eff]
    gold_energy = gold_proj.pow(2).sum(dim=1)          # [Tg]
    gold_energy = gold_energy.topk(min(k, Tg)).values.mean().item()

    if gold_energy < eps:
        return 0.0

    # predicted projection energy
    pred_proj = H_pred_n @ V_gold                      # [Tp, r_eff]
    pred_energy = pred_proj.pow(2).sum(dim=1)          # [Tp]
    pred_energy = pred_energy.topk(min(k, Tp)).values.mean().item()

    score = pred_energy / (gold_energy + eps)
    if score>1:
        score=0
    return score
def subspace_energy_overlap_topk_white_v3(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # token normalization (length-agnostic)
    try:
        H_gold_n = _token_whiten(H_gold, eps=eps)  # [Tg, D]
        H_pred_n = _token_whiten(H_pred, eps=eps)  # [Tp, D]
    except:
        return 0.0

    try:
        _, _, Vh = torch.linalg.svd(H_gold_n, full_matrices=False)
    except:
        return 0.0

    V_gold = Vh.T  # [D, r_eff]

    # golden projection energy
    gold_proj = H_gold_n @ V_gold                      # [Tg, r_eff]
    gold_energy = gold_proj.pow(2).sum(dim=1)          # [Tg]
    gold_energy = gold_energy.topk(min(k, Tg)).values.mean().item()

    if gold_energy < eps:
        return 0.0

    # predicted projection energy
    pred_proj = H_pred_n @ V_gold                      # [Tp, r_eff]
    pred_energy = pred_proj.pow(2).sum(dim=1)          # [Tp]
    pred_energy = pred_energy.topk(min(k, Tp)).values.mean().item()

    score = pred_energy / (gold_energy + eps)
    if score>1.5:
        score=0
    if score>1:
        score=1
    return score
def subspace_energy_overlap_topk_white_norm(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # token normalization (length-agnostic)
    if Tg==1 or Tp==1:
        H_gold_n = _token_norm(H_gold, eps=eps)  # [Tg, D]
        H_pred_n = _token_norm(H_pred, eps=eps)  # [Tp, D]
    else:
        H_gold_n = _token_whiten(H_gold, eps=eps)  # [Tg, D]
        H_pred_n = _token_whiten(H_pred, eps=eps)  # [Tp, D]


    # effective rank (cannot exceed min(Tg, D))
    # r_eff = min(r, Tg, D)
    # if r_eff <= 0:
        # return 0.0

    # deterministic SVD on golden normalized states
    # H_gold_n = U S Vh, where Vh: [min(Tg,D), D]
    try:
        _, _, Vh = torch.linalg.svd(H_gold_n, full_matrices=False)
    except:
        return 0.0

    V_gold = Vh.T  # [D, r_eff]

    # golden projection energy
    gold_proj = H_gold_n @ V_gold                      # [Tg, r_eff]
    gold_energy = gold_proj.pow(2).sum(dim=1)          # [Tg]
    gold_energy = gold_energy.topk(min(k, Tg)).values.mean().item()

    if gold_energy < eps:
        return 0.0

    # predicted projection energy
    pred_proj = H_pred_n @ V_gold                      # [Tp, r_eff]
    pred_energy = pred_proj.pow(2).sum(dim=1)          # [Tp]
    pred_energy = pred_energy.topk(min(k, Tp)).values.mean().item()

    score = pred_energy / (gold_energy + eps)
    return score
def subspace_energy_overlap_topk_norm(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # token normalization (length-agnostic)
    H_gold_n = _token_norm(H_gold, eps=eps)  # [Tg, D]
    H_pred_n = _token_norm(H_pred, eps=eps)  # [Tp, D]

    # effective rank (cannot exceed min(Tg, D))
    # r_eff = min(r, Tg, D)
    # if r_eff <= 0:
        # return 0.0

    # deterministic SVD on golden normalized states
    # H_gold_n = U S Vh, where Vh: [min(Tg,D), D]
    try:
        _, _, Vh = torch.linalg.svd(H_gold_n, full_matrices=False)
    except:
        return 0.0

    V_gold = Vh.T  # [D, r_eff]

    # golden projection energy
    gold_proj = H_gold_n @ V_gold                      # [Tg, r_eff]
    gold_energy = gold_proj.pow(2).sum(dim=1)          # [Tg]
    gold_energy = gold_energy.topk(min(k, Tg)).values.mean().item()

    if gold_energy < eps:
        return 0.0

    # predicted projection energy
    pred_proj = H_pred_n @ V_gold                      # [Tp, r_eff]
    pred_energy = pred_proj.pow(2).sum(dim=1)          # [Tp]
    pred_energy = pred_energy.topk(min(k, Tp)).values.mean().item()

    score = pred_energy / (gold_energy + eps)
    return score
def subspace_energy_overlap_topk_nocenter(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # token normalization (length-agnostic)
    H_gold_n = H_gold
    H_pred_n = H_pred

    # effective rank (cannot exceed min(Tg, D))
    # r_eff = min(r, Tg, D)
    # if r_eff <= 0:
        # return 0.0

    # deterministic SVD on golden normalized states
    # H_gold_n = U S Vh, where Vh: [min(Tg,D), D]
    try:
        _, _, Vh = torch.linalg.svd(H_gold_n, full_matrices=False)
    except:
        return 0.0

    V_gold = Vh.T  # [D, r_eff]

    # golden projection energy
    gold_proj = H_gold_n @ V_gold                      # [Tg, r_eff]
    gold_energy = gold_proj.pow(2).sum(dim=1)          # [Tg]
    gold_energy = gold_energy.topk(min(k, Tg)).values.mean().item()

    if gold_energy < eps:
        return 0.0

    # predicted projection energy
    pred_proj = H_pred_n @ V_gold                      # [Tp, r_eff]
    pred_energy = pred_proj.pow(2).sum(dim=1)          # [Tp]
    pred_energy = pred_energy.topk(min(k, Tp)).values.mean().item()

    score = pred_energy / (gold_energy + eps)
    return score
def subspace_energy_overlap_nocenter_topk(
    golden_hidden,       # [S, D] ( golden final answer hidden)
    pred_hidden,         # [S, D] ( predicted final answer)
    golden_mask,         # [S]
    pred_mask,           # [S]
    r=32,
    k=5,
    eps=1e-8,
):
    H_pred = pred_hidden[pred_mask]             # [Tp, D]
    if H_pred.size(0) == 0:
        return 0
    H_gold = golden_hidden[golden_mask]       # [Tg, D]
    if H_gold.size(0) == 0:
        return 0
    # correctness subspace (per-sample)
    V_gold = randomized_svd_topr(H_gold, r)   # [D, r]
    # golden energy
    gold_proj = H_gold @ V_gold               # [Tg, r]
    gold_energy = gold_proj.pow(2).sum(dim=1)
    gold_energy = gold_energy.topk(min(k,H_gold.size(0))).values.mean().item()
    if gold_energy < eps:
        return 0
    # --------------------------------------------------
    # 2. extract predicted answer tokens for sample b
    # --------------------------------------------------
    # projection energy
    pred_proj = H_pred @ V_gold
    pred_energy = pred_proj.pow(2).sum(dim=1)
    pred_energy = pred_energy.topk(min(k,H_pred.size(0))).values.mean().item()
    score = pred_energy / (gold_energy + eps)
    return score
def subspace_energy_overlap_tokencenter_topk(
    golden_hidden,       # [S, D] ( golden final answer hidden)
    pred_hidden,         # [S, D] ( predicted final answer)
    golden_mask,         # [S]
    pred_mask,           # [S]
    r=32,
    k=5,
    eps=1e-8,
):
    """
    For each sample b:
      - extract golden final answer tokens
      - build per-sample correctness subspace V_b
      - compute golden_energy_b
      - extract predicted final answer tokens
      - compute predicted_energy_b
      - score_b = predicted_energy_b / golden_energy_b

    return: list of scores (length B)
    """
    # H_all_gold = golden_hidden        # [S, D]
    H_pred = pred_hidden[pred_mask]             # [Tp, D]
    if H_pred.size(0) == 0:
        return 0
    H_gold = golden_hidden[golden_mask]       # [Tg, D]
    if H_gold.size(0) == 0:
        return 0
    # whiten golden
    H_gold_w = token_center(H_gold)
    # correctness subspace (per-sample)
    V_gold = randomized_svd_topr(H_gold_w, r)   # [D, r]
    # golden energy
    gold_proj = H_gold_w @ V_gold               # [Tg, r]
    gold_energy = gold_proj.pow(2).sum(dim=1)
    gold_energy = gold_energy.topk(min(k,H_gold.size(0))).values.mean().item()
    if gold_energy < eps:
        return 0
    # --------------------------------------------------
    # 2. extract predicted answer tokens for sample b
    # --------------------------------------------------
    # whiten predicted
    H_pred_w = token_center(H_pred)
    # projection energy
    pred_proj = H_pred_w @ V_gold
    pred_energy = pred_proj.pow(2).sum(dim=1)
    pred_energy = pred_energy.topk(min(k,H_pred.size(0))).values.mean().item()
    score = pred_energy / (gold_energy + eps)
    return score
def mean_cosine_score(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    eps=1e-8,
):
    # extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    # span mean (no centering, no token norm)
    v_pred = H_pred.mean(dim=0)   # [D]
    v_gold = H_gold.mean(dim=0)   # [D]

    # cosine similarity
    denom = (v_pred.norm() * v_gold.norm()).clamp_min(eps)
    score = (v_pred @ v_gold) / denom
    score =(score.item()+1)/2

    return score
def subspace_energy_overlap_goldencenter_topk(
    golden_hidden,       # [S, D] ( golden final answer hidden)
    pred_hidden,         # [S, D] ( predicted final answer)
    golden_mask,         # [S]
    pred_mask,           # [S]
    r=32,
    k=5,
    eps=1e-8,
):
    """
    For each sample b:
      - extract golden final answer tokens
      - build per-sample correctness subspace V_b
      - compute golden_energy_b
      - extract predicted final answer tokens
      - compute predicted_energy_b
      - score_b = predicted_energy_b / golden_energy_b

    return: list of scores (length B)
    """
    # H_all_gold = golden_hidden        # [S, D]
    H_pred = pred_hidden[pred_mask]             # [Tp, D]
    if H_pred.size(0) == 0:
        return 0
    H_gold = golden_hidden[golden_mask]       # [Tg, D]
    if H_gold.size(0) == 0:
        return 0
    # whiten golden
    # correctness subspace (per-sample)
    V_gold = randomized_svd_topr(H_gold, r)   # [D, r]
    # golden energy
    gold_proj = H_gold @ V_gold               # [Tg, r]
    gold_energy = gold_proj.pow(2).sum(dim=1)
    gold_energy = gold_energy.topk(min(k,H_gold.size(0))).values.mean().item()
    if gold_energy < eps:
        return 0
    # --------------------------------------------------
    # 2. extract predicted answer tokens for sample b
    # --------------------------------------------------
    # whiten predicted
    # projection energy
    pred_proj = H_pred @ V_gold
    pred_energy = pred_proj.pow(2).sum(dim=1)
    pred_energy = pred_energy.topk(min(k,H_pred.size(0))).values.mean().item()
    score = pred_energy / (gold_energy + eps)
    return score
def subspace_energy_overlap_tokencenter_mean(
    golden_hidden,       # [S, D] ( golden final answer hidden)
    pred_hidden,         # [S, D] ( predicted final answer)
    golden_mask,         # [S]
    pred_mask,           # [S]
    r=32,
    eps=1e-8,
):
    """
    For each sample b:
      - extract golden final answer tokens
      - build per-sample correctness subspace V_b
      - compute golden_energy_b
      - extract predicted final answer tokens
      - compute predicted_energy_b
      - score_b = predicted_energy_b / golden_energy_b

    return: list of scores (length B)
    """
    # H_all_gold = golden_hidden        # [S, D]
    H_pred = pred_hidden[pred_mask]             # [Tp, D]
    if H_pred.size(0) == 0:
        return 0
    H_gold = golden_hidden[golden_mask]       # [Tg, D]
    if H_gold.size(0) == 0:
        return 0
    # whiten golden
    H_gold_w = token_center(H_gold)
    # correctness subspace (per-sample)
    V_gold = randomized_svd_topr(H_gold_w, r)   # [D, r]
    # golden energy
    gold_proj = H_gold_w @ V_gold               # [Tg, r]
    gold_energy = gold_proj.pow(2).mean().item()
    # gold_energy = gold_energy.topk(min(k,H_gold.size(0))).values.mean().item()
    if gold_energy < eps:
        return 0
    # --------------------------------------------------
    # 2. extract predicted answer tokens for sample b
    # --------------------------------------------------
    # whiten predicted
    H_pred_w = token_center(H_pred)
    # projection energy
    pred_proj = H_pred_w @ V_gold
    pred_energy = pred_proj.pow(2).mean().item()
    score = pred_energy / (gold_energy + eps)
    return score
def subspace_sim(
    golden_hidden,   # [Lg, D]
    pred_hidden,     # [Lp, D]
    golden_mask,     # [Lg]
    pred_mask,       # [Lp]
    r: int = 32,
    k: int = None,
):
    # ---- 1. extract tokens ----
    Hg = golden_hidden[golden_mask]   # [Tg, D]
    Hp = pred_hidden[pred_mask]       # [Tp, D]

    if Hg.size(0) < 2 or Hp.size(0) < 2:
        return 0.0

    if k is not None and Hg.size(0) > k:
        Hg = Hg[-k:]
    if k is not None and Hp.size(0) > k:
        Hp = Hp[-k:]

    # ---- 3. SVD → subspaces ----
    # Hg = U S Vh → Vg = Vh.T
    _, _, Vh_g = torch.linalg.svd(Hg, full_matrices=False)
    _, _, Vh_p = torch.linalg.svd(Hp, full_matrices=False)

    Vg = Vh_g.T
    Vp = Vh_p.T

    max_r = min(r, Vg.size(1), Vp.size(1))
    if max_r == 0:
        return 0.0

    Vg = Vg[:, :max_r]
    Vp = Vp[:, :max_r]

    # ---- 4. Grassmann similarity (cos²) ----
    M = Vg.T @ Vp                      # [r, r]
    sim = (M.pow(2).sum() / max_r).item()  # ∈ [0, 1]

    return sim

def subspace_sim_r_weighted_golden_nodiv(
    golden_hidden,   # [B, Lg, D]
    pred_hidden,     # [B, Lp, D]
    golden_mask,     # [B, Lg]
    pred_mask,       # [B, Lp]
    r: int = 32,
    k: int = None,
    beta: float = 1.0,     # 权重幂次: 0.5 或 1.0 常用
    eps: float = 1e-8,
):
    """
    Weighted Grassmann similarity:
      sim_w = (1/r) * || diag(w) * (Vg^T Vp) ||_F^2 / ||w||_2^2
    where w = (Sg[:max_r])**beta

    - bounded, no energy ratio, harder to hack
    - divide by fixed r to avoid short-span inflation
    """
    Hg = golden_hidden[golden_mask]   # [Tg, D]
    Hp = pred_hidden[pred_mask]       # [Tp, D]

    if Hg.size(0) ==0 or Hp.size(0) ==0 :
        return 0
    if k is not None and Hg.size(0) > k:
        Hg = Hg[-k:]
    if k is not None and Hp.size(0) > k:
        Hp = Hp[-k:]
    # token-center
    Hg = Hg - Hg.mean(dim=0, keepdim=True)
    Hp = Hp - Hp.mean(dim=0, keepdim=True)
    # SVD
    # Hg = Ug Sg Vg^T
    # Hp = Up Sp Vp^T
    _, Sg, Vh_g = torch.linalg.svd(Hg, full_matrices=False)
    _, Sp, Vh_p = torch.linalg.svd(Hp, full_matrices=False)
    Vg = Vh_g.T
    Vp = Vh_p.T
    max_r = min(r, Vg.size(1), Vp.size(1), Sg.numel(), Sp.numel())
    if max_r == 0:
        return 0
    Vg_r = Vg[:, :max_r]
    Vp_r = Vp[:, :max_r]
    # overlap matrix
    M = Vg_r.T @ Vp_r                         # [max_r, max_r]

    # golden singular-value weights (non-negative)
    w = (Sg[:max_r].clamp_min(eps) ** beta)   # [max_r]
    w2 = (w * w)                              # [max_r]

    # weighted Frobenius: sum_{i,j} (w_i^2) * M_{ij}^2
    # implement as (w[:,None] * M)^2 then sum
    weighted_sum = (w[:, None] * M).pow(2).sum()

    # normalize by ||w||^2 so weights don't change scale across samples
    sim = weighted_sum / (w2.sum() + eps)
    return sim.item()
def subspace_energy_overlap_per_sample(
    golden_hidden,       # [S, D] ( golden final answer hidden)
    pred_hidden,         # [S, D] ( predicted final answer)
    golden_mask,         # [S]
    pred_mask,           # [S]
    r=32,
    k=None,
    eps=1e-8,
):
    """
    For each sample b:
      - extract golden final answer tokens
      - build per-sample correctness subspace V_b
      - compute golden_energy_b
      - extract predicted final answer tokens
      - compute predicted_energy_b
      - score_b = predicted_energy_b / golden_energy_b

    return: list of scores (length B)
    """

    # --------------------------------------------------
    # 1. extract golden answer tokens for sample b
    # --------------------------------------------------
    # H_all_gold = golden_hidden        # [S, D]
    H_pred = pred_hidden[pred_mask]             # [Tp, D]
    if H_pred.size(0) == 0:
        return 0
    H_gold = golden_hidden[golden_mask]       # [Tg, D]
    if H_gold.size(0) == 0:
        return 0

    # (optionally last-k tokens)
    if k and H_gold.size(0) > k:
        H_gold = H_gold[-k:]

    # whiten golden
    H_gold_w = _token_whiten(H_gold)
    # correctness subspace (per-sample)
    V_gold = randomized_svd_topr(H_gold_w, r)   # [D, r]

    # golden energy
    gold_proj = H_gold_w @ V_gold               # [Tg, r]
    gold_energy = gold_proj.pow(2).sum().item()

    if gold_energy < eps:
        return 0

    # --------------------------------------------------
    # 2. extract predicted answer tokens for sample b
    # --------------------------------------------------


    if k and H_pred.size(0) > k:
        H_pred = H_pred[-k:]

    # whiten predicted
    H_pred_w = _token_whiten(H_pred)
    # projection energy
    pred_proj = H_pred_w @ V_gold
    pred_energy = pred_proj.pow(2).sum().item()

    score = pred_energy / (gold_energy + eps)
    return score
def subspace_energy_overlap_golden_anchored_topk(
    golden_hidden, pred_hidden, golden_mask, pred_mask,
    r=32, k=8, m_dir=None, eps=1e-8,
):
    H_gold = golden_hidden[golden_mask]
    H_pred = pred_hidden[pred_mask]

    if H_gold.size(0) ==0 or H_pred.size(0) ==0 :
        return 0

    H_gold_w = _token_whiten(H_gold)
    # Only use Vh (no need for Sg)
    Vh_g = torch.linalg.svd(H_gold_w, full_matrices=False).Vh
    V_gold_full = Vh_g.T  # [D, rank]
    r_eff = min(r, V_gold_full.size(1))
    if r_eff == 0:
        return 0
    V_gold = V_gold_full[:, :r_eff]  # [D, r_eff]
    # gold directional energies
    gold_dir_energy_tok = (H_gold_w @ V_gold).pow(2)  # [Tg, r_eff]
    k_g = min(k, gold_dir_energy_tok.size(0))
    gold_dir_energy = gold_dir_energy_tok.topk(k_g, dim=0).values.mean(dim=0)  # [r_eff]
    # select top-m directions by gold energy (optional)
    if m_dir is not None and m_dir < r_eff:
        idx = torch.topk(gold_dir_energy, m_dir).indices
        gold_dir_energy = gold_dir_energy[idx]
        V_gold = V_gold[:, idx]
        r_eff = m_dir
    gold_energy = gold_dir_energy.sum().item()
    if gold_energy < eps:
        return 0
    # pred
    H_pred_w = _token_whiten(H_pred)
    pred_dir_energy_tok = (H_pred_w @ V_gold).pow(2)  # [Tp, r_eff]
    k_p = min(k, pred_dir_energy_tok.size(0))
    pred_dir_energy = pred_dir_energy_tok.topk(k_p, dim=0).values.mean(dim=0)

    pred_energy = pred_dir_energy.sum().item()
    score = pred_energy / (gold_energy + eps)
    return score
def subspace_energy_overlap_topk_symmetric(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # Extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # Token normalization (length-agnostic)
    try:
        H_gold = _token_whiten(H_gold, eps=eps)  # [Tg, D]
        H_pred = _token_whiten(H_pred, eps=eps)  # [Tp, D]
    except:
        return 0.0

    try:
        _, _, Vh_gold = torch.linalg.svd(H_gold, full_matrices=False)
        _, _, Vh_pred = torch.linalg.svd(H_pred, full_matrices=False)
    except:
        return 0.0

    V_gold = Vh_gold.T  # [D, r_eff_gold]
    V_pred = Vh_pred.T  # [D, r_eff_pred]

    # Golden projection energy onto pred subspace
    gold_proj_pred = H_gold @ V_pred  # [Tg, r_eff_pred]
    gold_energy_pred = gold_proj_pred.pow(2).sum(dim=1)  # [Tg]
    gold_energy_pred = gold_energy_pred.topk(min(k, Tg)).values.mean().item()

    if gold_energy_pred < eps:
        return 0.0

    # Predicted projection energy onto gold subspace
    pred_proj_gold = H_pred @ V_gold  # [Tp, r_eff_gold]
    pred_energy_gold = pred_proj_gold.pow(2).sum(dim=1)  # [Tp]
    pred_energy_gold = pred_energy_gold.topk(min(k, Tp)).values.mean().item()

    if pred_energy_gold < eps:
        return 0.0

    # Symmetric score: energy ratio in both directions
    pred_gold_ratio = np.clip(pred_energy_gold / (gold_energy_pred + eps),0,1)
    gold_pred_ratio = np.clip(gold_energy_pred / (pred_energy_gold + eps),0,1)
    score = (pred_gold_ratio + gold_pred_ratio) / 2
    return score

def subspace_energy_overlap_topk_symmetric_norm(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # Extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # Token normalization (length-agnostic)
    try:
        H_gold = _token_norm(H_gold, eps=eps)  # [Tg, D]
        H_pred = _token_norm(H_pred, eps=eps)  # [Tp, D]
    except:
        return 0.0

    try:
        _, _, Vh_gold = torch.linalg.svd(H_gold, full_matrices=False)
        _, _, Vh_pred = torch.linalg.svd(H_pred, full_matrices=False)
    except:
        return 0.0

    V_gold = Vh_gold.T  # [D, r_eff_gold]
    V_pred = Vh_pred.T  # [D, r_eff_pred]

    # Golden projection energy onto pred subspace
    gold_proj_pred = H_gold @ V_pred  # [Tg, r_eff_pred]
    gold_energy_pred = gold_proj_pred.pow(2).sum(dim=1)  # [Tg]
    gold_energy_pred = gold_energy_pred.topk(min(k, Tg)).values.mean().item()

    if gold_energy_pred < eps:
        return 0.0

    # Predicted projection energy onto gold subspace
    pred_proj_gold = H_pred @ V_gold  # [Tp, r_eff_gold]
    pred_energy_gold = pred_proj_gold.pow(2).sum(dim=1)  # [Tp]
    pred_energy_gold = pred_energy_gold.topk(min(k, Tp)).values.mean().item()

    if pred_energy_gold < eps:
        return 0.0

    # Symmetric score: energy ratio in both directions
    pred_gold_ratio = np.clip(pred_energy_gold / (gold_energy_pred + eps),0,1)
    gold_pred_ratio = np.clip(gold_energy_pred / (pred_energy_gold + eps),0,1)
    score = (pred_gold_ratio + gold_pred_ratio) / 2
    return score

def subspace_energy_overlap_topk_gold_to_pred(
    golden_hidden,       # [S, D] golden final answer hidden
    pred_hidden,         # [S, D] predicted final answer hidden
    golden_mask,         # [S] boolean mask for golden answer tokens
    pred_mask,           # [S] boolean mask for predicted answer tokens
    k=5,
    eps=1e-8,
):
    # Extract spans
    H_pred = pred_hidden[pred_mask]    # [Tp, D]
    if H_pred.size(0) == 0:
        return 0.0

    H_gold = golden_hidden[golden_mask]  # [Tg, D]
    if H_gold.size(0) == 0:
        return 0.0

    Tg, D = H_gold.shape
    Tp = H_pred.size(0)

    # Token normalization (length-agnostic)
    H_gold = _token_whiten(H_gold, eps=eps)  # [Tg, D]
    H_pred = _token_whiten(H_pred, eps=eps)  # [Tp, D]

    try:
        _, _, Vh_pred = torch.linalg.svd(H_pred, full_matrices=False)
    except:
        return 0.0

    V_pred = Vh_pred.T  # [D, r_eff_pred]

    # Golden projection energy onto pred subspace
    gold_proj_pred = H_gold @ V_pred  # [Tg, r_eff_pred]
    gold_energy_pred = gold_proj_pred.pow(2).sum(dim=1)  # [Tg]
    gold_energy_pred = gold_energy_pred.topk(min(k, Tg)).values.mean().item()

    if gold_energy_pred < eps:
        return 0.0

    # Final score: energy ratio from golden to pred
    return gold_energy_pred

def sigmoid_k(x, k=6):
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x (float): Input value in the range [0, 1].
        k (float): Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        float: Mapped value in the range [0, 1].
    """
    if k == 0: # vanilla sigmoid
        return 1 / (1 + np.exp(-x))

    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)
    
    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))
    
    # Scale the output to [0, 1]
    return sigmoid

def threshold_t_sigmoid_k(x, t, k=6):
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x (float): Input value in the range [0, 1].
        k (float): Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        float: Mapped value in the range [0, 1].
    """
    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)
    
    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))

    result = 0 if sigmoid < t else sigmoid
    
    return result

def subspace_energy_overlap_goldencenter_topk(
    golden_hidden,       # [S, D] ( golden final answer hidden)
    pred_hidden,         # [S, D] ( predicted final answer)
    golden_mask,         # [S]
    pred_mask,           # [S]
    r=32,
    k=5,
    eps=1e-8,
):
    """
    For each sample b:
      - extract golden final answer tokens
      - build per-sample correctness subspace V_b
      - compute golden_energy_b
      - extract predicted final answer tokens
      - compute predicted_energy_b
      - score_b = predicted_energy_b / golden_energy_b

    return: list of scores (length B)
    """
    # H_all_gold = golden_hidden        # [S, D]
    H_pred = pred_hidden[pred_mask]             # [Tp, D]
    if H_pred.size(0) == 0:
        return 0
    H_gold = golden_hidden[golden_mask]       # [Tg, D]
    if H_gold.size(0) == 0:
        return 0
    # whiten golden
    # H_gold_w = token_center(H_gold)
    golden_mean=H_gold.mean(dim=0,keepdim=True)
    H_gold_w=H_gold-golden_mean
    # correctness subspace (per-sample)
    V_gold = randomized_svd_topr(H_gold_w, r)   # [D, r]
    # golden energy
    gold_proj = H_gold_w @ V_gold               # [Tg, r]
    gold_energy = gold_proj.pow(2).sum(dim=1)
    gold_energy = gold_energy.topk(min(k,H_gold.size(0))).values.mean().item()
    if gold_energy < eps:
        return 0
    # --------------------------------------------------
    # 2. extract predicted answer tokens for sample b
    # --------------------------------------------------
    # whiten predicted
    # H_pred_w = token_center(H_pred)
    H_pred_w = H_pred - golden_mean
    # projection energy
    pred_proj = H_pred_w @ V_gold
    pred_energy = pred_proj.pow(2).sum(dim=1)
    pred_energy = pred_energy.topk(min(k,H_pred.size(0))).values.mean().item()
    score = pred_energy / (gold_energy + eps)
    return score
def threshold_t_sigmoidv2_k(x, t, k=6):
    # concave curve
    if x < t:
        result = 0
    else:
        x = x - t
        x = x * k
        result = 1 / (1 + np.exp(-x))
    return result


def threshold_t_sigmoidv2fixed_k(x, t, k=6):
    if x < t:
        result = 0
    else:
        x = (x- t) * k
        result = 1 / (1 + np.exp(-x))  * ((1 - t) / 0.5) - ((1-t) - t) 
    
    return result


def threshold_t_sigmoidv3_k(x, t, k=6):
    # convex curve
    if x < t:
        result = 0
    else:
        x = (x - 1) * k
        result = 1 / (1 + np.exp(-x)) + 0.5
    return result


def leaky_relu_like(score, threshold, alpha=0.01):
    """
    Maps a score from [0, 1] to [0, 1] using a Leaky ReLU-like function.

    Parameters:
    - score: The input score in the range [0, 1].
    - threshold: The threshold below which the score is scaled.
    - alpha: The slope for scores below the threshold (default is 0.01).

    Returns:
    - The transformed score in the range [0, 1].
    """
    if score < threshold:
        return alpha * score
    else:
        return score


def threshold_t_tanh_k(score, t, k=6):
    # Apply tanh transformation with a configurable scaling factor
    transformed_score = (np.tanh(score * k - k / 2) + 1) / 2
    
    # Threshold values smaller than 0.05 to 0
    if transformed_score < t:
        transformed_score = 0
    
    return transformed_score


def format_reward(predict_str: str, format_mode='R1') -> float:
    if format_mode not in ['R1','R1_nothink']:
        return 0
    def _validate_tags(input_string):
        if format_mode == 'R1':
            tags = ['<think>', '</think>', '<answer>', '</answer>']
        elif format_mode == 'R1_nothink':
            tags = ['<answer>', '</answer>']
        else:
            raise ValueError(f"Unsupported format mode: {format_mode}")
        for tag in tags:
            if input_string.count(tag) != 1:
                return 0.0
        return 1.0

    if _validate_tags(predict_str) == 0.0:
        return 0.0
    if format_mode == 'R1':
        pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>.*', re.DOTALL)
    elif format_mode == 'R1_nothink':
        pattern = re.compile(r'.*<answer>.*</answer>.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)

    return 1.0 if match_result else 0.0


class HiddenRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score_name=None, 
                 shaping_function_name=None, discrete_function_name=None, 
                 format_coefficient=0.1, save_results_dir=None, reward_type='pr',
                 gt_tokens_one_more=False, gt_tokens_one_more_adjusted=False,
                 format_mode='R1') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # assert compute_score is None
        self.compute_score_name = compute_score_name
        print(f"{shaping_function_name=}")
        if shaping_function_name == 'identity':
            self.shaping_function = lambda x:x
        elif shaping_function_name == 'exp':
            self.shaping_function = partial(amplify_reward, alpha=1.8)
        elif shaping_function_name == 'exp_norm':
            self.shaping_function = partial(exp_norm, alpha=2)
        elif shaping_function_name == 'clip_01':
            self.shaping_function = partial(clip_01)
        elif shaping_function_name == 'one_minus':
            self.shaping_function = lambda x: 1 - x
        elif shaping_function_name == 'random':
            self.shaping_function = lambda x: random.random()
        elif shaping_function_name.startswith('threshold'):
            threshold = float(shaping_function_name.split('_')[-1])
            self.shaping_function = lambda x: 0 if x < threshold else x
        elif shaping_function_name.startswith('sigmoid_'):
            print(f"Selecting sigmoid_k function.")
            k = float(shaping_function_name.split('_')[-1])
            self.shaping_function = partial(sigmoid_k, k=k)
        elif shaping_function_name.startswith('leaky_'):
            # e.g., leaky_0.05
            print(f"Using leaky-relu like function")
            threshold = float(shaping_function_name.split('_')[1])
            self.shaping_function = partial(leaky_relu_like, threshold=threshold)
        elif shaping_function_name.startswith('comp'): # compound
            # comp_threshold_0.3_sigmoid_6
            threshold = float(shaping_function_name.split('_')[2])
            k = float(shaping_function_name.split('_')[4])
            if 'sigmoidv2fixed' in shaping_function_name:
                print(f"Using sigmoid v2fixed")
                self.shaping_function = partial(threshold_t_sigmoidv2fixed_k, t=threshold, k=k)
            elif 'sigmoidv3' in shaping_function_name:
                print(f"Using sigmoid v3")
                self.shaping_function = partial(threshold_t_sigmoidv3_k, t=threshold, k=k)
            elif 'sigmoidv2' in shaping_function_name:
                print(f"Using sigmoid v2")
                self.shaping_function = partial(threshold_t_sigmoidv2_k, t=threshold, k=k)
            elif 'sigmoid' in shaping_function_name:
                print(f"Using sigmoid v1")
                self.shaping_function = partial(threshold_t_sigmoid_k, t=threshold, k=k)
            elif 'tanh' in shaping_function_name:
                self.shaping_function = partial(threshold_t_tanh_k, t=threshold, k=k)
            else:
                raise ValueError
        else:
            print(f"{shaping_function_name=}")
            raise NotImplementedError(f"{shaping_function_name=}")
        self.discrete_function_name = discrete_function_name
        self.format_coefficient = format_coefficient
        self.reward_type = reward_type
        self.format_mode = format_mode
        self.gt_tokens_one_more = gt_tokens_one_more
        self.gt_tokens_one_more_adjusted = gt_tokens_one_more_adjusted

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        hidden_score_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        scoreA_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        extracted_answer_list = ["do not log"]*len(data)
        # prompt_ids = data[0].batch['prompts']
        prompt_length = data[0].batch['prompts'].shape[-1]
        golden_hidden=data.batch['golden_hidden_states'][0].to("cuda")
        pred_hidden=data.batch['actor_hidden_states'][0].to("cuda")
        golden_mask=data.batch['golden_answer_mask'].to("cuda")
        pred_mask=data.batch['actor_answer_mask'].to("cuda")
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] # len(response_ids): 1024
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum() # 329
            valid_response_ids = response_ids[:valid_response_length] 

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # sequences_str = self.tokenizer.decode(sequences) # <|im_start|>system\n..the answer is: \boxed{10, 30, 40}<|im_end|>

            # prompt_str = self.tokenizer.decode(valid_prompt_ids) # '<|im_start|>system\nA conversation between ... answer here </answer>.<|im_end|>\n<|im_start|>user\nLet the parabola ... and $X_{M}$.<|im_end|>\n<|im_start|>assistant\n'
            predict_str = self.tokenizer.decode(valid_response_ids) # To determine the relationship ... relative to each other and the parabola.<|im_end|>

            format_score = format_reward(predict_str=predict_str, format_mode=self.format_mode)

            # hidden_score= subspace_energy_overlap_per_sample(
            #     golden_hidden=data_item.batch['golden_hidden_states'][0],
            #     pred_hidden=data_item.batch['actor_hidden_states'][0],
            #     golden_mask=data_item.batch['golden_answer_mask'],
            #     pred_mask=data_item.batch['actor_answer_mask']
            # )
            hidden_score= subspace_energy_overlap_per_sample(
                golden_hidden=golden_hidden[i],
                pred_hidden=pred_hidden[i],
                golden_mask=golden_mask[i],
                pred_mask=pred_mask[i]
            )
            hidden_score = self.shaping_function(hidden_score)

            if self.format_coefficient == -1:
                score = hidden_score if format_score == 1 else -1
            else:
                score = (1 - self.format_coefficient) * hidden_score + self.format_coefficient * format_score
            reward_tensor[i, valid_response_length - 1] = score
            format_reward_tensor[i, valid_response_length - 1] = format_score
            hidden_score_tensor[i, valid_response_length - 1] = hidden_score


        return reward_tensor, hidden_score_tensor, scoreA_tensor, format_reward_tensor,extracted_answer_list 


    @staticmethod
    def map_to_bins(score: float, num_bins: int) -> float:
        """
        Maps a score in [0,1] to the nearest discrete value based on num_bins.
        
        :param score: A float between 0 and 1.
        :param num_bins: The number of discrete bins.
        :return: The mapped discrete value.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        
        step = 1 / num_bins
        return round(score / step) * step
