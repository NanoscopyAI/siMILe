from CHWOS.data_classes.dstorm_sim import dstorm_sim_dataset

from CHWOS.utils.log import get_logger
logger = get_logger(__name__)


class hela_dynasore_pitstop_dataset(dstorm_sim_dataset):
    
    def set_reporting_variables(self):
        self.removed_classes_count = {}
        self.feat_to_split = {}
        self.class_count_by_split = {}
        
        for split in ['train', 'valid']:
            self.removed_classes_count[split] = {}
            self.class_count_by_split[split] = {}
            
            for t in self.TAGS:
                class_dict = {i:0 for i in self.all_classes[t]}
                self.removed_classes_count[split][t] = {**class_dict}
                self.class_count_by_split[split][t] = {**class_dict}
                
                for f in self.featuresSplit[split][t]:
                    self.feat_to_split[tuple(f)] = split
                    self.class_count_by_split[split][t][self.class_record[t][tuple(f)]] += 1
                    
        self.tag_to_common_distinct_class = {}
        for t in self.TAGS:
            self.tag_to_common_distinct_class[t] = {'distinct': [2], 'common': [1,3], 'all': [1,2,3]}
        logger.debug(f'Tag classes: {self.tag_to_common_distinct_class}')