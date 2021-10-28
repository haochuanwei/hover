import math
import numpy as np
import torch
from collections import defaultdict
from hover.utils.torch_helper import cross_entropy_with_probs


def loss_coteaching_directed(y_student, y_teacher, target, forget_rate):
    """
    Subroutine for loss_coteaching_graph.
    """
    num_remember = math.ceil((1 - forget_rate) * target.size(0))
    assert (
        num_remember > 0
    ), f"Expected at least one remembered target, got {num_remember}"

    loss_teacher_detail = cross_entropy_with_probs(y_teacher, target, reduction="none")
    idx_to_learn = np.argsort(loss_teacher_detail.data)[:num_remember]
    loss_student = cross_entropy_with_probs(
        y_student[idx_to_learn], target[idx_to_learn], reduction="mean"
    ).unsqueeze(0)
    return loss_student


def prediction_disagreement(pred_list, reduce=False):
    """
    Compute disagreements between predictions.
    """
    disagreement = defaultdict(dict)
    for i in range(0, len(pred_list)):
        for j in range(i, len(pred_list)):
            _disagreed = np.not_equal(pred_list[i], pred_list[j])
            if reduce:
                _disagreed = np.mean(_disagreed)
            disagreement[i][j] = _disagreed
            disagreement[j][i] = _disagreed
    return dict(disagreement)


def loss_coteaching_graph(y_list, target, tail_head_adjacency_list, forget_rate):
    """
    Co-teaching from differences.
    Generalized to graph representation where each vertex is a classifier and each edge is a source to check differences with and to learn from.
    y_list: list of logits from different classifiers.
    target: target, which is allowed to be probabilistic.
    tail_head_adjacency_list: the 'tail' classifier learns from the 'head'.
    forget_rate: the proportion of high-loss contributions to discard.
    """
    # initialize co-teaching losses
    loss_list = []
    for i in range(0, len(y_list)):
        assert tail_head_adjacency_list[i], f"Expected at least one teacher for {i}"
        _losses = []
        for j in tail_head_adjacency_list[i]:
            # fetch yi as student(tail), yj as teacher(head)
            _yi, _yj = y_list[i], y_list[j]
            _tar = target

            # add loss contribution to list
            _contribution = loss_coteaching_directed(_yi, _yj, _tar, forget_rate)
            _losses.append(_contribution)

        # concatenate and average up
        _loss = torch.mean(torch.cat(_losses))
        loss_list.append(_loss)

    return loss_list


def identity_adjacency(info_dict):
    """
    Each node points to itself.
    """
    refs = []
    acc_list = info_dict["accuracy"]
    for i in range(0, len(acc_list)):
        refs.append([i])
    return refs


def cyclic_adjacency(info_dict, acc_bar=0.5):
    """
    Nodes form a cycle.
    Triggers if accuracies are high enough.
    """
    refs = []
    acc_list = info_dict["accuracy"]
    for i in range(0, len(acc_list)):
        candidate = (i + 1) % (len(acc_list))
        if acc_list[i] > acc_bar and acc_list[candidate] > acc_bar:
            refs.append([candidate])
        else:
            refs.append([i])
    return refs


def cyclic_except_last(info_dict, acc_bar=0.5):
    """
    Cyclic except the last member.
    Triggers if accuracies are high enough.
    """
    refs = []
    acc_list = info_dict["accuracy"]
    for i in range(0, len(acc_list) - 1):
        candidate = (i + 1) % (len(acc_list) - 1)
        if acc_list[i] > acc_bar and acc_list[candidate] > acc_bar:
            refs.append([candidate])
        else:
            refs.append([i])
    refs.append([len(acc_list) - 1])
    return refs


def accuracy_priority(info_dict, acc_bar=0.5):
    """
    Every node points at the most accurate member that is not itself.
    Triggers if accuracies are high enough.
    """
    refs = []
    acc_list = info_dict["accuracy"]
    for i in range(0, len(acc_list)):
        top_candidates = sorted(
            range(len(acc_list)), key=lambda j: acc_list[j], reverse=True
        )
        candidate = top_candidates[0] if top_candidates[0] != i else top_candidates[1]
        if acc_list[i] > acc_bar and acc_list[candidate] > acc_bar:
            refs.append([candidate])
        else:
            refs.append([i])
    return refs


def disagreement_priority(info_dict, acc_bar=0.5):
    """
    Everyone node points at the most different member that is not itself.
    Triggers if accuracies are high enough.
    """
    refs = []
    acc_list = info_dict["accuracy"]
    disagree_dict = info_dict["disagreement_rate"]
    for i in range(0, len(acc_list)):
        top_candidates = sorted(
            disagree_dict[i].keys(), key=lambda j: disagree_dict[i][j], reverse=True
        )
        candidate = top_candidates[0] if top_candidates[0] != i else top_candidates[1]
        if acc_list[i] > acc_bar and acc_list[candidate] > acc_bar:
            refs.append([candidate])
        else:
            refs.append([i])
    return refs