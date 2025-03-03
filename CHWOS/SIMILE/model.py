import os
import copy 
import numpy as np # type: ignore
import joblib

from functools import partial

from CHWOS.SIMILE.classification import classify_and_cut
from CHWOS.SIMILE.svm import train
from CHWOS.SIMILE.inference import predict_bag, predict_instances, compute_all_accuracy
from CHWOS.utils.log import get_logger


logger = get_logger(__name__)

#############################################################################
##
##  Would like to split classify and cut, also fold mil library into this, 
##  and seperate classes for AE, SYMC on off etc
##
###############################################################################

#########################################################################################
##
##  NOT INCLUDED ARE RE-PREDICTIONS FOR VALID, SUCH AS FOR THE DEPRECATED HELA DATASET 
##
##        all_accs = [valid_acc]
##        ex_bags, ex_labels = copy.deepcopy(dataset.valid_bags), copy.deepcopy(dataset.valid_bag_labels)
##        for val_pred_i in range(exp_config.valid_pred_iters):
##           ex_bags, ex_labels = reform_bags_and_labels(ex_bags, ex_labels)
##            ex_pred_lbls_valid_bags = predict_bag(trainer, ex_bags)
##            ex_valid_acc = compute_all_accuracy_partial(ex_pred_lbls_valid_bags, ex_labels)
##           logger.info(f'Testing valid iter: {val_pred_i}, acc: {ex_valid_acc}')
##            all_accs.append(ex_valid_acc)
##
##       for k in list(all_accs[0].keys())
##        valid_acc[k] = np.mean([a[k] for a in all_accs])
#########################################################################################
        
#Class for a given iteration of Simile
class SimileIterModel:
    def __init__(self, iteration):
        self.iteration = iteration
        self.dataset = None
        self.trainer = None
        self.train_kmean_cutoffs = None
        self.ip = None
        
    def set_dataset(self, dataset):
        self.dataset = dataset
        self.exp_config = dataset.exp_config
        self.model_save_folder = os.path.join(self.exp_config.MODEL_LOCATION, self.dataset.save_string)
        
        self.predict_instance_partial = partial(predict_instances, miles_predict=not self.exp_config.SYM_C)
        self.compute_all_accuracy_partial = partial(compute_all_accuracy, A_TAG=dataset.A_TAG, B_TAG=dataset.B_TAG, mil_labels=self.exp_config.mil_labels)

    def train(self):
        assert self.dataset != None, 'No dataset has been set for iteration model'
        
        self.trainer = train(sigma2=self.exp_config.sigma, train_bags=self.dataset.train_bags, \
                            train_bag_labels=self.dataset.train_bag_labels, \
                            bagsize=self.exp_config.bagsize, C=self.exp_config.C)
    
    def predict_bags(self, train=True, valid=True, test=False):
        #Currently valid is used with test based on configuration, should be fixed
        assert self.trainer != None, 'Iteration model not trained'
        
        splitbags = [self.dataset.train_bags, self.dataset.valid_bags, self.dataset.test_bags]
        splitlbls = [self.dataset.train_bag_labels, self.dataset.valid_bag_labels, self.dataset.test_bag_labels]
        results = {}
        
        for split_str, split_flag, bags, lbls in zip(['train', 'valid', 'test'], [train, valid, test], splitbags, splitlbls):
            if not split_flag:
                continue
            logger.info(f"Predicting {split_str} bags...")
            
            pred_lbls = predict_bag(self.trainer, bags)
            results[split_str] = self.compute_all_accuracy_partial(pred_lbls, lbls)
            
            logger.info(f'{split_str} accuracy: {results[split_str]}')
            
        return results

    def predict_bag_instance_scores(self, train=True, valid=True):
        empty_instance_pred_dict = {
            'train': copy.deepcopy(self.dataset.clean_train_iter_predictions),
            'valid': copy.deepcopy(self.dataset.clean_valid_iter_predictions)
            }
            
        assert self.trainer != None, 'Iteration model not trained'
        
        splitbags = [self.dataset.train_bags, self.dataset.valid_bags]
        results = {}
        
        for split_str, split_flag, bags in zip(['train', 'valid'], [train, valid], splitbags):
            if not split_flag:
                continue
            logger.info(f"Predicting {split_str} instance scores...")
            
            results[split_str] = self.predict_instance_partial(self.trainer, bags, empty_instance_pred_dict[split_str])
            self.ip = results
            
        return results
    
    def classify_and_cut(self, train=True, valid=True):
        assert self.ip != None, 'Instance scores have not been predicted'
        
        results = {}
        for split_str, split_flag in zip(['train', 'valid'], [train, valid]):
            if not split_flag:
                continue
            logger.info(f"Classifying and cutting {split_str}...")
            
            assert not (split_str != 'train' and self.train_kmean_cutoffs == None), 'train cutoffs have not been found'
                
            
            was_cut, cutoffs = classify_and_cut(prediction_dict=self.ip[split_str], dataset=self.dataset, 
                                                splittag=split_str, iteration=self.iteration, 
                                                found_cutoffs=self.train_kmean_cutoffs)
            results[split_str] = {'was_cut': was_cut, 'cutoffs': cutoffs}
            
            if split_str == 'train' and self.train_kmean_cutoffs == None:
                self.train_kmean_cutoffs = cutoffs
                
        return results
    
    def save(self, compress=3):        
        path = self.model_save_folder
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except:
            pass
        
        file_name = f'trainer_{self.iteration}.joblib'
        full_save_path = os.path.join(path, file_name)
        logger.info(f'Saving model to {full_save_path}')
        
        tmp_dataset = self.dataset
        self.dataset = None
        joblib.dump(self, full_save_path, compress=compress)  
        self.dataset = tmp_dataset