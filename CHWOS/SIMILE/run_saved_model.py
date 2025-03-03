import os
import re
import glob
import joblib
from CHWOS.utils.log import get_logger

logger = get_logger(__name__)

class SavedModel:
    
    def __init__(self, model_path, dataset):
        self.model_path = model_path
        logger.info(f'Model path: {self.model_path}')

        self.dataset = dataset
        self.model_files = self.get_models()
        
    def get_models(self):
        logger.info(f'222 Model path: {self.model_path}')
        all_files = glob.glob(os.path.join(self.model_path, '*.joblib'))
        logger.info(f'Found saved models: {all_files}')
        pattern = r'trainer_(\d+)\.joblib'
        all_files = sorted(all_files, key=lambda x: int(re.search(pattern, x).group(1)))
        logger.info(f'Found saved models: {len(all_files)}')
        return all_files
    
    def run(self):
        logger.info('Running saved model on validation')
                
        for i, file in enumerate(self.model_files):
            logger.info(f'Running {i+1}/{len(self.model_files)}, file: {file}')
            
            iter_model = joblib.load(file)
            self.dataset.bagsize = iter_model.exp_config.bagsize
            iter_model.set_dataset(self.dataset)

            bag_accuracys = iter_model.predict_bags(train=False)       
            iter_model.predict_bag_instance_scores(train=False)
            iter_model.classify_and_cut(train=False)    
            
            self.dataset.update_results(iter_model.trainer, 
                                            None, bag_accuracys['valid'], 
                                            None, iter_model.train_kmean_cutoffs, 
                                            None, iter_model.ip['valid'], save=True)
            self.dataset.save_results(iteration=i+1)
            
            if 0 < self.dataset.exp_config.MAX_ITER <= i+1:
                logger.info(f'Max iterations of {self.dataset.exp_config.MAX_ITER} reached')
                break