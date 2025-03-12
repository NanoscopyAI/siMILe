from CHWOS.SIMILE.model import SimileIterModel
from CHWOS.utils.log import get_logger

logger = get_logger(__name__)


def train_SIMILE(dataset, ss):
    exp_config = dataset.exp_config

    logger.debug("RUN PARAMS:")
    logger.debug(f"AE: {exp_config.AE}")
    logger.debug(f"SYM_C: {exp_config.SYM_C}")
    logger.debug(f"C: {exp_config.C}")
    logger.debug(f"minacc: {exp_config.MIN_ACC}")
    if exp_config.MIN_ACC_VALID > 0:
        logger.debug(f"minacc: {exp_config.MIN_ACC_VALID}")
    logger.debug(f"sigma: {exp_config.sigma}")
    logger.debug(f"bagsize: {exp_config.bagsize}")
    if exp_config.fold_split_idx > 0:
        logger.debug(f"foldindex: {exp_config.fold_split_idx}")
    if exp_config.MAX_ITER > 0:
        logger.debug(f"max_iter: {exp_config.MAX_ITER}")

    itercounter = 1
    dataset.itercounter = itercounter
    while itercounter <= exp_config.MAX_ITER if exp_config.MAX_ITER > 0 else True:
        if exp_config.MAX_ITER != 1 and exp_config.AE:
            logger.info(f"On iteration: {itercounter}")

        iter_model = SimileIterModel(itercounter)
        iter_model.set_dataset(dataset)
        iter_model.train()
        bag_accuracys = iter_model.predict_bags()

        if not bag_accuracys_good(bag_accuracys, exp_config):
            return False

        iter_model.predict_bag_instance_scores()
        cut_results = iter_model.classify_and_cut()

        dataset.update_results(
            iter_model.trainer,
            bag_accuracys["train"],
            bag_accuracys["valid"],
            iter_model.train_kmean_cutoffs,
            iter_model.train_kmean_cutoffs,
            iter_model.ip["train"],
            iter_model.ip["valid"],
            save=True,
        )

        if not cut_results_good(cut_results, exp_config, iter_model):
            return False

        if exp_config.save_models:
            iter_model.save()

        itercounter += 1
        dataset.itercounter = itercounter

    return True


def bag_accuracys_good(bag_accuracys, exp_config):
    if not (bag_accuracys["train"]["avg"] >= exp_config.MIN_ACC):
        logger.debug("Insufficient train performance, exiting...")
        return False

    if not (bag_accuracys["valid"]["avg"] >= exp_config.MIN_ACC_VALID):
        logger.debug("Insufficient valid performance, exiting...")
        return False

    return True


def cut_results_good(cut_results, exp_config, iter_model):
    if not cut_results["valid"]["was_cut"]:
        if exp_config.AE:
            logger.debug("No changes to instance classification")
        else:
            logger.debug("No adversarial erasing, finishing up...")
            if exp_config.save_models:
                iter_model.save()
        return False
    else:
        return True
