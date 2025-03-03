import pandas as pd
import numpy as np
from CHWOS.utils.log import get_logger


logger = get_logger(__name__)
class CavPTRF_exp:
    
    def __init__(self):
        
        self.cav_features_coloc_loc = "/home/hallgric/SIMILE/CHWOS_experiments/Cav_PTRF_Simile_colocalize/compute_coloc/result_dfs/3d/colocal_all_iter_results.pkl"
        self.cav_features_coloc = pd.read_pickle(self.cav_features_coloc_loc)
        
        self.iteration_to_coloc = {}
        
    def experiment_report(self, dataset, iteration):
        split = 'valid'

        classified_instances_float32 = {tuple(np.array(key, dtype=np.float32)): value for key, value in dataset.classified_instances[split].items()}
        self.cav_features_coloc['predicted'] = self.cav_features_coloc.apply(lambda row: 1 if tuple(row[:30]) in classified_instances_float32 else 0, axis=1)
        
        pred = self.cav_features_coloc.query('predicted == 1')
        pred_and_coloc = self.cav_features_coloc.query('predicted == 1 & ptrf_overlaps != -1')
        pred_and_no_coloc = self.cav_features_coloc.query('predicted == 1 & ptrf_overlaps == -1')
                
        coloc_dict = {'total_blobs': self.cav_features_coloc.shape[0],
                      'total_predicted': pred.shape[0],
                      'total_predicted_and_coloc': pred_and_coloc.shape[0],
                      'total_predicted_and_no_coloc': pred_and_no_coloc.shape[0]}
                
        logger.info(f"Total blob count: {coloc_dict['total_blobs']}")
        logger.info(f"Number predicted: {coloc_dict['total_predicted']}")
        if coloc_dict['total_predicted'] != 0:
            logger.info(f"Number predicted and colocalizes: {coloc_dict['total_predicted_and_coloc']}, frac: {round(coloc_dict['total_predicted_and_coloc']/coloc_dict['total_predicted'], 3)}")
            logger.info(f"Number predicted and does not colocalize: {coloc_dict['total_predicted_and_no_coloc']}, frac: {round(coloc_dict['total_predicted_and_no_coloc']/coloc_dict['total_predicted'], 3)}")
            
            
        
        self.iteration_to_coloc[iteration] = coloc_dict
        
        if coloc_dict['total_predicted'] != 0:
            if coloc_dict['total_predicted_and_coloc']/coloc_dict['total_predicted'] <= 0.55:
                return 'bad'