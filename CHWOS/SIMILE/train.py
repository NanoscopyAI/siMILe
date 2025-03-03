import os
#sys.path.append("/local-scratch/localhome/cdh13/Documents/CHWOS")
import numpy as np

from CHWOS.SIMILE.SIMILE import train_SIMILE
from CHWOS.SIMILE.run_saved_model import SavedModel
from CHWOS.utils.parse_configs import get_config
from CHWOS.utils.timer import timer_decorator
from CHWOS.utils.dataset import get_dataset
from CHWOS.utils.misc import set_random_and_get_sequence
from CHWOS.utils.log import get_logger
from CHWOS.utils.folds import cross_validation_split

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

logger = get_logger(__name__)

@timer_decorator    
def _single_sided_runs(dataset, ss, oneside=False):
    dataset.exp_config.MAX_ITER = 1
    single_sided_iterations = 0
    while(True):
        if dataset.exp_config.AE:
            logger.info(f'SINGLE SIDED ITERATIONS: {single_sided_iterations}')
        logger.info('#'*10 + 'TRAINING FIRST LABEL' + '#'*10)
        A_run = train_SIMILE(dataset, ss)

        if not oneside:
            logger.info('#'*20 + 'TRAINING SECOND LABEL' + '#'*20)
            dataset.single_classify_tag = dataset.B_TAG
            dataset.exp_config.mil_labels = [1, 0]
            dataset.create_and_set_split_bags_and_labels(gave_splits=True)
            B_run = train_SIMILE(dataset, ss)
        else:
            B_run = False

        if not any([A_run, B_run]):
            logger.info('Done training, exiting')
            break
        
        if not oneside:
            dataset.single_classify_tag = dataset.A_TAG
            dataset.exp_config.mil_labels = [0, 1]
            dataset.create_and_set_split_bags_and_labels(gave_splits=True)
        single_sided_iterations += 1
    
    logger.debug('FINAL VALID RESULTS:')
    #dataset.print_removed_class_count('valid')
    #dataset.calculate_performance_metrics(print_perf=True)


@timer_decorator
def _double_sided_run(dataset, ss):
    logger.info('#'*20 + 'TRAINING' + '#'*20)
    train_SIMILE(dataset, ss)
    
    logger.debug('FINAL VALID RESULTS:')
    #dataset.print_removed_class_count('valid')
    #dataset.calculate_performance_metrics(print_perf=True)


@timer_decorator
def run(config_path=None, exp_config=None, dataset=None, ss=None):
    if dataset is None:
        if not config_path and not exp_config:
            raise ValueError('Either config_path or exp_config must be provided')

        if not exp_config:
            exp_config = get_config(config_path)
        
        if not ss:    
            ss = set_random_and_get_sequence(exp_config.seed)
            
        dataset = get_dataset(exp_config)
    else:
        exp_config = dataset.exp_config
        if not ss:
            ss = set_random_and_get_sequence(exp_config.seed)
    
    if exp_config.SYM_C:
        _double_sided_run(dataset, ss)
    else:
        oneside = False
        if exp_config.DATA_NAME == 'PC3PTRF':
            oneside = True
        _single_sided_runs(dataset, ss, oneside=oneside)


    return dataset


def run_from_save(exp_config, ss=None):
    if exp_config.run_all:
        dataset = get_dataset(exp_config, setup=False)
        dataset.process_data()
        Af = np.copy(dataset.features[dataset.A_TAG])
        Bf = np.copy(dataset.features[dataset.B_TAG])
        
        dataset.featuresSplit = {'test':{}, 'train':{}, 'valid':{}}
        dataset.featuresSplit['train'][dataset.A_TAG] = Af
        dataset.featuresSplit['train'][dataset.B_TAG] = Bf
        dataset.featuresSplit['valid'][dataset.A_TAG] = Af
        dataset.featuresSplit['valid'][dataset.B_TAG] = Bf
        dataset.create_and_set_split_bags_and_labels(gave_splits=True)
    else:
        dataset = get_dataset(exp_config)
        
    sm = SavedModel(exp_config.run_saved, dataset)
    try:
        sm.run()
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return sm


@timer_decorator
def CV(exp_config, features=None, n_folds=5, ss=None):
    
    if features is None:
        dataset = get_dataset(exp_config, setup=False)
        dataset.process_data()
        Af = np.copy(dataset.features[dataset.A_TAG])
        Bf = np.copy(dataset.features[dataset.B_TAG])
        del dataset
    else:
        Af, Bf = features
    
    for i, (A_train, B_train, A_test, B_test) in enumerate(cross_validation_split(Af, Bf, n_folds, seed=exp_config.seed)):
        logger.info(f'Fold = {i+1}')
        exp_config.fold_split_idx = i + 1
        dataset = get_dataset(exp_config, setup=False)

        dataset.process_data()
        dataset.featuresSplit = {'test':{}, 'train':{}, 'valid':{}}
        dataset.featuresSplit['train'][dataset.A_TAG] = A_train
        dataset.featuresSplit['train'][dataset.B_TAG] = B_train
        dataset.featuresSplit['valid'][dataset.A_TAG] = A_test
        dataset.featuresSplit['valid'][dataset.B_TAG] = B_test
        dataset.create_and_set_split_bags_and_labels(gave_splits=True)
        run(dataset=dataset, ss=ss)


    return dataset


def nested_CV(exp_config, n_folds=5, ss=None):
    dataset = get_dataset(exp_config, setup=False)
    dataset.process_data()
    Af = np.copy(dataset.features[dataset.A_TAG])
    Bf = np.copy(dataset.features[dataset.B_TAG])
    del dataset
    
    folds = list(cross_validation_split(Af, Bf, n_folds, seed=exp_config.seed))
    fold_n = exp_config.args.NCV_fold
    A_train, B_train, A_test, B_test = folds[fold_n - 1]
    logger.info(f'Outer Fold = {fold_n}')
    
    if not exp_config.use_ncv_test:
        CV(exp_config, features=(A_train, B_train), n_folds=5, ss=ss)
    else:
        logger.info('Running against outer fold test set')
        dataset = get_dataset(exp_config, setup=False)
        dataset.process_data()
        dataset.featuresSplit = {'test':{}, 'train':{}, 'valid':{}}
        dataset.featuresSplit['train'][dataset.A_TAG] = A_train
        dataset.featuresSplit['train'][dataset.B_TAG] = B_train
        dataset.featuresSplit['valid'][dataset.A_TAG] = A_test
        dataset.featuresSplit['valid'][dataset.B_TAG] = B_test
        dataset.create_and_set_split_bags_and_labels(gave_splits=True)
        run(dataset=dataset, ss=ss)