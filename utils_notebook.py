import torch
import numpy as np
import scipy
from typing import Dict, Optional, List
# from tqdm import tqdm


def probs_decrease(probs: np.array) -> np.array:
    L = len(probs)
    diffs = []
    for i in range(L):
        for j in range(i + 1, L):
            diffs.append(probs[j] - probs[i])
    return np.array(diffs)


def modal_probs_decreasing(
    _preds: Dict[int, torch.Tensor],
    _probs: torch.Tensor,
    layer: Optional[int] = None,
    verbose: bool = False,
    N: int = 10000,
    diffs_type: str = "consecutive",
    thresholds: List[float] = [-0.01, -0.05, -0.1, -0.2, -0.5],
    return_ids: bool = False
) -> Dict[float, float]:
    """
    nr. of decreasing modal probability vectors in anytime-prediction regime
    function can also be used for ground truth probabilities, set layer=None
    """
    nr_non_decreasing = {threshold: 0 for threshold in thresholds}
    diffs = {threshold: [] for threshold in thresholds}
    for i in range(N):
        if layer is None:
            c = _preds[i]
        else:
            c = _preds[layer - 1][i]
        probs_i = _probs[:, i, c].cpu().numpy()
        if diffs_type == "consecutive":
            diffs_i = np.diff(probs_i)
        elif diffs_type == "all":
            diffs_i = probs_decrease(probs_i)
        else:
            raise ValueError()
        # diffs.append(diffs_i.min())
        for threshold in nr_non_decreasing.keys():
            if np.all(diffs_i > threshold):
                nr_non_decreasing[threshold] += 1
            else:
                diffs[threshold].append(i)
                if verbose:
                    print(i, probs_i)
    # print(nr_non_decreasing)
    # print(np.mean(diffs))
    nr_decreasing = {
        -1.0 * k: ((N - v) / N) * 100 for k, v in nr_non_decreasing.items()
    }
    if return_ids:
        return nr_decreasing, diffs
    else:
        return nr_decreasing
    

def f_probs_ovr_poe_logits_weighted_generalized(logits, threshold=0.0, weights=None):
    L, N, C = logits.shape[0], logits.shape[1], logits.shape[2]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.0
    if weights is not None:
        assert logits.shape[0] == weights.shape[0]
        for l in range(L):
            probs[l, :, :] = probs[l, :, :] ** weights[l]
    probs = np.cumprod(probs, axis=0)
    # normalize
    for l in range(L):
        for n in range(N):
            sum_l_n = probs[l, n, :].sum()
            if sum_l_n > 0.:
                probs[l, n, :] = probs[l, n, :] / sum_l_n
            else:
                # probs[l, n, :] = (1 / C) * torch.ones(C)
                # probs[l, n, :] = torch.zeros(C)
                probs[l, n, :] = torch.softmax(logits[l, n, :], dim=0)
                # probs[l, n, :] = (logits[:l + 1, n, :] > 0).sum(axis=0) / (logits[:l + 1, n, :] > 0).sum()
    return probs


def anytime_caching(_probs: torch.Tensor, N: int, L: int) -> torch.Tensor:
    _preds = []
    _probs_stateful = []
    for n in range(N):
        preds_all, probs_stateful_all = [], []
        max_prob_all, pred_all, max_id = 0., None, 0.
        for l in range(L):
            _max_prob, _pred = _probs[l, n, :].max(), _probs[l, n, :].argmax()
            if _max_prob >= max_prob_all:
                max_prob_all = _max_prob
                pred_all = _pred
                prob_stateful_all = _probs[l, n, :]
                max_id = l
            else:
                prob_stateful_all = _probs[max_id, n, :]
            preds_all.append(pred_all)
            probs_stateful_all.append(prob_stateful_all)
        _preds.append(torch.stack(preds_all))
        _probs_stateful.append(torch.stack(probs_stateful_all))

    _preds = torch.stack(_preds)
    _probs_stateful = torch.stack(_probs_stateful)
    return _probs_stateful.permute(1, 0, 2)


def get_metrics_for_paper(logits: torch.Tensor, targets: torch.Tensor, model_name: str, thresholds: List[float] = [-0.01, -0.05, -0.1, -0.2, -0.25, -0.33, -0.5]):

    L = len(logits)
    N = len(targets)

    acc_dict, mono_modal_dict, mono_ground_truth_dict = {}, {}, {}

    probs = torch.softmax(logits, dim=2)
    preds = {i: torch.argmax(probs, dim=2)[i, :] for i in range(L)}
    acc = [(targets == preds[i]).sum() / len(targets) for i in range(L)]

    probs_pa = torch.tensor(f_probs_ovr_poe_logits_weighted_generalized(logits, weights=(np.arange(1, L + 1, 1, dtype=float) / L)))
    preds_pa = {i: torch.argmax(probs_pa, dim=2)[i, :] for i in range(L)}
    acc_pa = [(targets == preds_pa[i]).sum() / len(targets) for i in range(L)]

    probs_ca = anytime_caching(probs, N=N, L=L)
    preds_ca= {i: torch.argmax(probs_ca, dim=2)[i, :] for i in range(L)}
    acc_ca = [(targets == preds_ca[i]).sum() / len(targets) for i in range(L)]

    for _probs, _preds, _acc, _name in zip([probs, probs_pa, probs_ca], [preds, preds_pa, preds_ca], [acc, acc_pa, acc_ca], [model_name, model_name + '-PA', model_name + '-CA']):
        acc_dict[_name] = [round(float(x), 4) for x in _acc]
        mono_modal_dict[_name] = [round(x, 4) for x in modal_probs_decreasing(_preds, _probs, layer=L, N=N, thresholds=thresholds, diffs_type="all").values()]
        mono_ground_truth_dict[_name] = [round(x, 4) for x in modal_probs_decreasing(targets, _probs, layer=None, N=N, thresholds=thresholds, diffs_type="all").values()]

    return acc_dict, mono_modal_dict, mono_ground_truth_dict