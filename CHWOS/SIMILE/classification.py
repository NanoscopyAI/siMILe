import numpy as np
from sklearn.cluster import KMeans

from CHWOS.utils.bags import generate_train_test_validation_bags
from CHWOS.utils.log import get_logger
from CHWOS.utils.timer import timer_decorator

logger = get_logger(__name__)


@timer_decorator
def classify_symc(dataset, splittag, val_prediction_dict, cutoffs=None):
    # Aggregate prediction values
    agg_pred_vals = []
    for k, v in val_prediction_dict.items():
        if len(v) > 1:
            agg_pred_vals.append(np.mean(v[1:]))
    agg_pred_vals = np.array(agg_pred_vals)

    if agg_pred_vals.size == 0:
        return False

    # Get kmeans centers and cutoffs
    if not cutoffs:
        kmeans_centers = sorted(
            KMeans(
                n_clusters=3,
                init="random",
                random_state=dataset.exp_config.seed,
                n_init=1000,
            )
            .fit(agg_pred_vals.reshape(-1, 1))
            .cluster_centers_
        )
        uppercutoff = ((kmeans_centers[2] - kmeans_centers[1]) / 2)[0]
        lowercutoff = ((kmeans_centers[0] - kmeans_centers[1]) / 2)[0]
        logger.debug(f"Found: upper cutoff: {uppercutoff}, lower cutoff: {lowercutoff}")
    else:
        uppercutoff = cutoffs["uppercutoff"]
        lowercutoff = cutoffs["lowercutoff"]
        logger.debug(f"Using: upper cutoff: {uppercutoff}, lower cutoff: {lowercutoff}")

    # Classify
    A_TAG, B_TAG = dataset.A_TAG, dataset.B_TAG
    for k, v in val_prediction_dict.items():
        if len(v) > 1:
            pred_val = np.mean(v[1:])
            if pred_val >= uppercutoff and dataset.inst_to_weak_lbl[splittag][k] == A_TAG:  # Allow common across both?
                dataset.classified_instances[splittag][k] = A_TAG
            elif pred_val <= lowercutoff and dataset.inst_to_weak_lbl[splittag][k] == B_TAG:
                dataset.classified_instances[splittag][k] = B_TAG
    return {"uppercutoff": uppercutoff, "lowercutoff": lowercutoff}


@timer_decorator
def classify_miles(dataset, splittag, val_prediction_dict):
    # Classify
    classify_tag = dataset.single_classify_tag
    logger.debug(f"Using MILES classify on {classify_tag}")
    for k, v in val_prediction_dict.items():
        if len(v) > 1:
            pred_val = v[-1]
            if pred_val == 1 and dataset.inst_to_weak_lbl[splittag][k] == classify_tag:
                dataset.classified_instances[splittag][k] = classify_tag


@timer_decorator
def cut(dataset, splittag, cut_sym=True):
    A_TAG, B_TAG = dataset.A_TAG, dataset.B_TAG
    classify_tag = dataset.single_classify_tag  # If single sided classify, use this, otherwise it is the same as A_TAG

    pre_cut_shape = dataset.featuresSplit[splittag][classify_tag].shape
    logger.debug("Pre cut {} feature shape: {}: {}".format(splittag, classify_tag, pre_cut_shape))
    dataset.featuresSplit[splittag][classify_tag] = np.array(
        [
            i
            for i in dataset.featuresSplit[splittag][classify_tag]
            if tuple(i) not in dataset.classified_instances[splittag]
        ]
    )

    if cut_sym:
        logger.debug(
            "Pre cut {} feature shape: {}: {}".format(splittag, B_TAG, dataset.featuresSplit[splittag][B_TAG].shape)
        )
        dataset.featuresSplit[splittag][B_TAG] = np.array(
            [
                i
                for i in dataset.featuresSplit[splittag][B_TAG]
                if tuple(i) not in dataset.classified_instances[splittag]
            ]
        )
        logger.debug(
            "Post cut {}: New feature shape: {}: {} {}: {}".format(
                splittag,
                A_TAG,
                dataset.featuresSplit[splittag][A_TAG].shape,
                B_TAG,
                dataset.featuresSplit[splittag][B_TAG].shape,
            )
        )
    else:
        post_cut_shape = dataset.featuresSplit[splittag][classify_tag].shape
        logger.debug("Post cut {} feature shape: {}: {}".format(splittag, classify_tag, post_cut_shape))
        if pre_cut_shape == post_cut_shape:
            logger.debug("No instances classified for {}".format(splittag))
            return False

    # Recreate bags and labels post cut
    bags, labels = generate_train_test_validation_bags(
        dataset.featuresSplit[splittag],
        A_TAG=A_TAG,
        B_TAG=B_TAG,
        split=[1, 0, 0],
        bagsize=dataset.exp_config.bagsize,
        mil_labels=dataset.exp_config.mil_labels,
    )
    dataset.set_bags_and_labels(splittag, bags, labels)
    return True


@timer_decorator
def classify_and_cut(prediction_dict, dataset, splittag, iteration, found_cutoffs=None):
    AE = dataset.exp_config.AE
    SYM_C = dataset.exp_config.SYM_C

    was_classified = True
    cutoffs = {}
    if SYM_C:
        cutoffs = classify_symc(dataset, splittag, prediction_dict, cutoffs=found_cutoffs)
        if not cutoffs:
            was_classified = False
    else:
        classify_miles(dataset, splittag, prediction_dict)

    was_cut = False
    if AE and was_classified:
        was_cut = cut(dataset, splittag, cut_sym=SYM_C)

    # Allow for any dataset specific post classify and cut function
    if "classify_and_cut" in dataset.post_step:
        dataset.post_step["classify_and_cut"](iteration, splittag)

    return was_cut, cutoffs
