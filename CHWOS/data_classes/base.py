import copy
import json
import numpy as np # type: ignore

from CHWOS.utils.save import save_data_numpy, get_savefile_name
from CHWOS.utils.bags import get_split_bags_and_labels
from CHWOS.utils import metrics as metrics
from CHWOS.SIMILE.prediction import get_prediction_dict

from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

class BASE_dataset():
    def __init__(self, exp_config, setup=True):
        self.exp_config = exp_config
        self.A_TAG = exp_config.A_TAG
        self.B_TAG = exp_config.B_TAG
        self.TAGS = [self.A_TAG, self.B_TAG]
        self.single_classify_tag = self.A_TAG
        self.features = {self.A_TAG: {}, self.B_TAG: {}}
        self.classified_instances = {'train': {}, 'valid': {}, 'test': {}}
        
        self.save = True
        
        logger.info(f'Dataset A: {self.A_TAG} vs B: {self.B_TAG}')
        
        self.load_feature_data()
        if setup:
            logger.debug('Running setup')
            self.randomize_data()
            self.process_data()
            self.set_feats_by_split()
            self.results['featureSplit'] = copy.deepcopy(self.featuresSplit)
        else:
            logger.debug('Skipping setup')
            
        self.set_post_step_functions()
        self.set_save_string()
        
    def load_feature_data(self):
        pass
    
    def randomize_data(self):
        pass
    
    def process_data(self):
        self.set_result_dict()
        
    def set_post_step_functions(self):
        self.post_step = {}

    def set_feats_by_split(self):
        self.featuresSplit = {'test':{}, 'train':{}, 'valid':{}}
        self.featuresSplit['train'][self.A_TAG] = np.array(copy.deepcopy(self.features[self.A_TAG]))
        self.featuresSplit['train'][self.B_TAG] = np.array(copy.deepcopy(self.features[self.B_TAG]))
        self.create_and_set_split_bags_and_labels(gave_splits=False)
        

    def create_and_set_split_bags_and_labels(self, gave_splits):
        self.featuresSplit, \
            self.train_bags, self.train_bag_labels, \
            self.valid_bags, self.valid_bag_labels, \
            self.test_bags, self.test_bag_labels = get_split_bags_and_labels(self.featuresSplit, self.exp_config, gave_splits=gave_splits)
        self.clean_valid_iter_predictions = get_prediction_dict(self.valid_bags, print_str='valid')
        self.clean_train_iter_predictions = get_prediction_dict(self.train_bags, print_str='train')
        self.set_inst_to_weak_lbl()
        
    def set_bags_and_labels(self, splittag, bags, labels): #should use a dict
        if splittag == 'train':
            self.train_bags = bags
            self.train_bag_labels = labels
        elif splittag == 'valid':
            self.valid_bags = bags
            self.valid_bag_labels = labels
        elif splittag == 'test':
            self.test_bags = bags
            self.test_bag_labels = labels
        else:
            logger.error('Invalid splittag')
            return
        logger.debug(f'New {splittag} shape {bags.shape}, {len(labels)}')

    def set_inst_to_weak_lbl(self):
        self.inst_to_weak_lbl = {}
        for split in self.featuresSplit:
            self.inst_to_weak_lbl[split] = {}
            for milclass in self.featuresSplit[split]:
                for inst in self.featuresSplit[split][milclass]:
                    self.inst_to_weak_lbl[split][tuple(inst)] = milclass

    def set_result_dict(self):
        results = {'valid': {}, 'test': {}, 'train': {}}
        for split in ['valid', 'test', 'train']:
            results[split]['ip'] = []
            results[split]['acc'] = []
            results[split]['cutoffs'] = []

        #results['trainer'] = []
        results['C'] = self.exp_config.C
        results['bs'] = self.exp_config.bagsize
        results['sigma'] = self.exp_config.sigma
        #results['valid_bags'] = np.copy(self.valid_bags)
        #results['train_bags'] = np.copy(self.train_bags)
        #results['features'] = np.copy(self.features)
        results['all_iter_predictions'] = []
        self.results = results

    def update_results(self, trainer, train_acc, valid_acc, train_cutoffs, valid_cutoffs, train_ip, valid_ip, save=True):
        #self.results['trainer'].append(trainer)
        self.results['predictions'] = self.classified_instances
        self.results['all_iter_predictions'].append(copy.deepcopy(self.classified_instances))
        
        self.results['train']['acc'].append(train_acc)
        #self.results['train']['ip'].append(train_ip)
        #self.results['train']['cutoffs'].append(train_cutoffs)

        self.results['valid']['acc'].append(valid_acc)
        self.results['valid']['ip'].append(valid_ip)
        self.results['valid']['cutoffs'].append(valid_cutoffs)

        if save and self.save:
            self.save_results()

    def set_save_string(self, include_date=True):
        save_name_dict = {
            'SIMILE': self.exp_config.DATA_NAME,
            'A': self.exp_config.A_TAG,
            'B': self.exp_config.B_TAG,
            'AE': self.exp_config.AE,
            'SC': self.exp_config.SYM_C,
            'sig': self.exp_config.sigma,
            'c': self.exp_config.C,
            'bs': self.exp_config.bagsize,
            'fld': self.exp_config.fold_split_idx,
            'NCV_fold': self.exp_config.NCV_fold,
            'minacc': self.exp_config.MIN_ACC,
            'minacc_valid': self.exp_config.MIN_ACC_VALID
        }
        if self.exp_config.AE == 1:
            del save_name_dict['AE']
        if self.exp_config.SYM_C == 1:
            del save_name_dict['SC']
        if self.exp_config.fold_split_idx == 0:
            del save_name_dict['fld']
        if self.exp_config.NCV_fold == 0:
            del save_name_dict['NCV_fold']

        self.save_string = get_savefile_name(save_name_dict, include_date=include_date)

    def save_results(self, iteration=None):            
        save_data_numpy(self.results, self.exp_config.RESULTS_LOCATION, self.save_string, iteration=iteration)