import json
import numpy as np # type: ignore
from collections import defaultdict
from CHWOS.utils import metrics as metrics
from CHWOS.data_classes.Mock import Mock_dataset

from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

class PC3PTRF_dataset(Mock_dataset):
    def __init__(self, exp_config, setup=True):
        super().__init__(exp_config, all_blobclasses=[1,2,3,4], setup=setup)

    def set_feats_by_split(self):
        self.featuresSplit = {'test':{}, 'train':{}, 'valid':{}}
        self.set_split_by_config_cells()
        self.create_and_set_split_bags_and_labels(gave_splits=True)


    def set_split_by_config_cells(self):
        parser = self.exp_config.parserEXP
        valid_cells = json.loads(parser['DATA'].get('valid_cells', []))
        test_cells = json.loads(parser['DATA'].get('test_cells', []))
        usetest = parser.getboolean('DATA', 'use_test')
        logger.info(f'Use test: {usetest}')
        
        if valid_cells == [] and test_cells == []:
            logger.info('No valid or test cells specified')
            return
        
        assert len(set(valid_cells).intersection(set(test_cells))) == 0, "The lists have common elements"
        logger.info(f'Valid cells: {valid_cells}')
        logger.info(f'Test cells: {test_cells}')
        
        for t in self.TAGS:
            files = self.meta[t]['file']
            valid_mask = np.array([any(s in filename for s in valid_cells) for filename in files])
            test_mask = np.array([any(s in filename for s in test_cells) for filename in files])
            
            assert sum(valid_mask & test_mask) == 0, "A file is both in valid and test"
            
            train_mask = ~(valid_mask | test_mask)
                
            for m, s in zip([valid_mask, test_mask, train_mask], ['valid', 'test', 'train']):
                logger.info(f'{t}, {s} mask len: {sum(m)}')

            self.featuresSplit['train'][t] = self.features[t][train_mask]                 
            if not usetest:
                self.featuresSplit['valid'][t] = self.features[t][valid_mask]
            else:
                self.featuresSplit['valid'][t] = self.features[t][test_mask]
            
                
    def set_reporting_variables(self):
        self.removed_classes_count = {}
        blobclass_dict = {i:0 for i in self.all_blobclasses}
        self.removed_classes_count['train'] = {self.A_TAG: {**blobclass_dict}, self.B_TAG: {**blobclass_dict}}
        self.removed_classes_count['valid'] = {self.A_TAG: {**blobclass_dict}, self.B_TAG: {**blobclass_dict}}
    
        self.feat_to_split = {}

        for t in self.TAGS:
            for emb in self.featuresSplit['train'][t]:
                self.feat_to_split[tuple(emb)] = 'train'
            for emb in self.featuresSplit['valid'][t]:
                self.feat_to_split[tuple(emb)] = 'valid'

        self.class_count = {i: {self.A_TAG: defaultdict(int), self.B_TAG: defaultdict(int)} for i in ['valid', 'train']}
        self.inst_to_class = {self.A_TAG: {}, self.B_TAG: {}}
        for t in self.TAGS:
            for i, emb in enumerate(self.features[t]):
                _split = self.feat_to_split.get(tuple(emb), False)
                if _split:
                    self.class_count[_split][t][self.meta[t]['classes'][i]] += 1
                    self.inst_to_class[t][tuple(emb)] = self.meta[t]['classes'][i]
                    
        logger.debug(f'Class count: {self.class_count}')

        for classstr in list(self.removed_classes_count['train'].keys()):
            for blobclass in list(self.removed_classes_count['train'][classstr].keys()):
                self.removed_classes_count['train'][classstr][f'Total:{blobclass}'] = self.class_count['train'][classstr][blobclass]
        for classstr in list(self.removed_classes_count['valid'].keys()):
            for blobclass in list(self.removed_classes_count['valid'][classstr].keys()):
                self.removed_classes_count['valid'][classstr][f'Total:{blobclass}'] = self.class_count['valid'][classstr][blobclass]

        parser = self.exp_config.parserEXP
        usesb = parser.getboolean('DATA', 'use_s1b')
        
        logger.debug(f'Use sb: {usesb}')
        
        if not usesb:
            self.lbl_to_pos_neg = {self.A_TAG: 
                                        {'pos': [4], 'neg': [1, 2, 3], 'all': [1, 2, 3, 4]}, 
                                self.B_TAG: 
                                        {'pos': [], 'neg': [1, 2], 'all': [1, 2]}
                                }
        else:
            self.lbl_to_pos_neg = {self.A_TAG: 
                                        {'pos': [4, 2], 'neg': [1, 3], 'all': [1, 2, 3, 4]}, 
                                self.B_TAG: 
                                        {'pos': [], 'neg': [1, 2], 'all': [1, 2]}
                                }

        logger.debug(f"lbl blobclasses to mil class: {self.lbl_to_pos_neg}")

        self.print_removed_class_count('train')
        self.print_removed_class_count('valid')
        self.report_vars_set = True