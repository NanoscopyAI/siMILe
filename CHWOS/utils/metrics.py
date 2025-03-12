import numpy as np  # type: ignore


def _get_tp_fp_tn_fn(pos_removed: np.array, total_P: np.array, neg_removed: np.array, total_N: np.array):
    TP = pos_removed
    FP = neg_removed
    TN = total_N - neg_removed
    FN = total_P - pos_removed

    return np.array(TP), np.array(FP), np.array(TN), np.array(FN)


def get_accuracy(pos_removed: np.array, total_P: np.array, neg_removed: np.array, total_N: np.array):
    TP, _, TN, _ = _get_tp_fp_tn_fn(**locals())

    P_acc = TP.sum() / total_P.sum()
    N_acc = TN.sum() / total_N.sum()
    PN_acc = (P_acc + N_acc) / 2
    PN_weighted_acc = (TP.sum() + TN.sum()) / (total_P.sum() + total_N.sum())

    return_dict = {
        "P_acc": P_acc,
        "N_acc": N_acc,
        "PN_acc": PN_acc,
        "PN_weighted_acc": PN_weighted_acc,
    }
    if len(total_P) + len(total_N) > 2:
        Tclass = np.concatenate((TP, TN))
        Tclass_total = np.concatenate((total_P, total_N))
        class_acc = Tclass.sum() / Tclass_total.sum()
        class_weighted_acc = (Tclass / Tclass_total).sum() / len(Tclass_total)
        return_class_dict = {
            "class_acc": class_acc,
            "class_weighted_acc": class_weighted_acc,
        }
        return_dict = {**return_dict, **return_class_dict}

    return return_dict


def get_precision(pos_removed: np.array, total_P: np.array, neg_removed: np.array, total_N: np.array):
    TP, FP, _, _ = _get_tp_fp_tn_fn(**locals())

    if TP.sum() + FP.sum() == 0:
        precision = 0.0
    else:
        precision = TP.sum() / (TP.sum() + FP.sum())

    return_dict = {"precision": precision}
    if len(total_P) > 1:
        a = TP
        b = TP + FP
        weighted_precision = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
        weighted_precision = weighted_precision.sum() / len(TP)

        return_dict["class_weighted_precision"] = weighted_precision
    return return_dict


def get_recall(pos_removed: np.array, total_P: np.array, neg_removed: np.array, total_N: np.array):
    TP, _, _, FN = _get_tp_fp_tn_fn(**locals())

    recall = TP.sum() / (TP.sum() + FN.sum())

    return_dict = {"recall": recall}
    if len(total_P) > 1:
        weighted_recall = (TP / (TP + FN)).sum() / len(TP)
        return_dict["class_weighted_precision"] = weighted_recall
    return return_dict


def get_f1(pos_removed: np.array, total_P: np.array, neg_removed: np.array, total_N: np.array):
    kwargs = {**locals()}
    precision_dict = get_precision(**kwargs)
    recall_dict = get_recall(**kwargs)

    precision = precision_dict["precision"]
    recall = recall_dict["recall"]
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    result_dict = {"f1": f1}

    if len(total_P) > 1:
        weighted_precision = precision_dict["class_weighted_precision"]
        weighted_recall = recall_dict["class_weighted_precision"]
        if weighted_precision + weighted_recall == 0:
            weighted_f1 = 0.0
        else:
            weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)

        result_dict["weighted_f1"] = weighted_f1

    return result_dict
