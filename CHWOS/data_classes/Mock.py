import copy
import json
import numpy as np # type: ignore
from collections import defaultdict
from CHWOS.data_classes.base import BASE_dataset
from CHWOS.utils import metrics as metrics

from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

class Mock_dataset(BASE_dataset):
    def __init__(self, exp_config, all_blobclasses=[1,2,3], setup=True):
        self.processed = False
        self.all_blobclasses = all_blobclasses
        self.report_vars_set = False
        
        super().__init__(exp_config, setup=setup)   
                     
    def set_post_step_functions(self):
        super().set_post_step_functions()
        self.post_step['classify_and_cut'] = self.experiment_report

    def load_feature_data(self):
        self.meta = {self.A_TAG: {}, self.B_TAG: {}}
        self.features[self.A_TAG], self.meta[self.A_TAG] = np.load(self.exp_config.DATALOCS[self.exp_config.A_TAG], allow_pickle=True)
        self.features[self.B_TAG], self.meta[self.B_TAG] = np.load(self.exp_config.DATALOCS[self.exp_config.B_TAG], allow_pickle=True)

        self.features[self.A_TAG] = self.features[self.A_TAG][self.exp_config.A_TAG]
        self.features[self.B_TAG] = self.features[self.B_TAG][self.exp_config.B_TAG]
        self.features[self.A_TAG] = self.features[self.A_TAG][:, :30]
        self.features[self.B_TAG] = self.features[self.B_TAG][:, :30]
        
        self.meta[self.A_TAG] = self.meta[self.A_TAG][self.exp_config.A_TAG]
        self.meta[self.B_TAG] = self.meta[self.B_TAG][self.exp_config.B_TAG]
        
        logger.debug(f"A feature original shape: {self.features[self.A_TAG].shape}")
        logger.debug(f"B feature original shape: {self.features[self.B_TAG].shape}")
        
    def randomize_data(self):
        if not self.processed:
            self.process_data()
        for lbl in self.TAGS:
            _temp = list(zip(self.features[lbl], self.meta[lbl]['classes']))
            np.random.shuffle(_temp)
            self.features[lbl], self.meta[lbl]['classes'] = zip(*_temp)
            self.features[lbl] = np.array(self.features[lbl])
            self.meta[lbl]['classes'] = list(self.meta[lbl]['classes'])
            
    def process_data(self):
        if self.processed:
            return
        super().process_data()
        self.processed = True
        
    def create_and_set_split_bags_and_labels(self, gave_splits):
        super().create_and_set_split_bags_and_labels(gave_splits)
        if not self.report_vars_set:
            self.set_reporting_variables()
        
    def set_result_dict(self):
        super().set_result_dict()
        self.results['meta'] = np.copy(self.meta)
        self.results['metrics'] = []
        
    def set_reporting_variables(self):
        self.removed_classes_count = {}
        blobclass_dict = {i:0 for i in self.all_blobclasses}
        self.removed_classes_count['train'] = {self.A_TAG: {**blobclass_dict}, self.B_TAG: {**blobclass_dict}}
        self.removed_classes_count['valid'] = {self.A_TAG: {**blobclass_dict}, self.B_TAG: {**blobclass_dict}}
    
        self.feat_to_split = {}

        for milclass in self.TAGS:
            #for emb in featuresSplit['test'][milclass]:
            #    feat_to_split[tuple(emb)] = 'test'
            for emb in self.featuresSplit['train'][milclass]:
                self.feat_to_split[tuple(emb)] = 'train'
            for emb in self.featuresSplit['valid'][milclass]:
                self.feat_to_split[tuple(emb)] = 'valid'

        self.class_count = {'valid': {self.A_TAG: defaultdict(int), self.B_TAG: defaultdict(int)}, 'train': {self.A_TAG: defaultdict(int), self.B_TAG: defaultdict(int)}}
        self.inst_to_class = {self.A_TAG: {}, self.B_TAG: {}}
        for milclass in self.TAGS:
            for i, emb in enumerate(self.features[milclass]):
                _split = self.feat_to_split.get(tuple(emb), False)
                if _split:
                    self.class_count[_split][milclass][self.meta[milclass]['classes'][i]] += 1
                    self.inst_to_class[milclass][tuple(emb)] = self.meta[milclass]['classes'][i]
                    
        logger.debug(f'Class count: {self.class_count}')

        for classstr in list(self.removed_classes_count['train'].keys()):
            for blobclass in list(self.removed_classes_count['train'][classstr].keys()):
                self.removed_classes_count['train'][classstr][f'Total:{blobclass}'] = self.class_count['train'][classstr][blobclass]
        for classstr in list(self.removed_classes_count['valid'].keys()):
            for blobclass in list(self.removed_classes_count['valid'][classstr].keys()):
                self.removed_classes_count['valid'][classstr][f'Total:{blobclass}'] = self.class_count['valid'][classstr][blobclass]

        self.lbl_to_pos_neg = {}
        for lbl in self.removed_classes_count['valid'].keys():
            self.lbl_to_pos_neg[lbl] = {'pos': [], 'neg': [], 'all': []}
            for blobclass in self.removed_classes_count['valid'][lbl].keys():
                if type(blobclass) == str:
                    continue

                if self.removed_classes_count['valid'][lbl][f'Total:{blobclass}'] == 0:
                    continue

                self.lbl_to_pos_neg[lbl]['all'].append(blobclass)

        lbl, lbl2 = self.removed_classes_count['valid'].keys()
        for lbl, lbl2 in [(lbl, lbl2), (lbl2, lbl)]:
            for blobclass in self.lbl_to_pos_neg[lbl]['all']:
                self.lbl_to_pos_neg[lbl]['neg' if blobclass in self.lbl_to_pos_neg[lbl2]['all'] else 'pos'].append(blobclass)
        logger.debug(f'lbl blobclasses to mil class: {self.class_count}')

        self.print_removed_class_count('train')
        self.print_removed_class_count('valid')
        self.report_vars_set = True
        
    def print_removed_class_count(self, splittag):
        printstr = f'{splittag.upper()}: REMOVED CLASS COUNT: \n'
        for lbl in self.removed_classes_count[splittag]:
            printstr += f"{lbl}: \n"
            for i in self.removed_classes_count[splittag][lbl]:
                if type(i) == int:
                    printstr += f"    Class {i}: {self.removed_classes_count[splittag][lbl][i]} / {self.removed_classes_count[splittag][lbl][f'Total:{i}']}\n"
        logger.debug(printstr)
        
    def calculate_performance_metrics(self, print_perf=True):
        self.performance_metrics = {}
        splittag = 'valid'
        for lbl in self.removed_classes_count[splittag]:
            self.performance_metrics[lbl] = {}

            if self.lbl_to_pos_neg[lbl]['pos'] == []:
                if print_perf:
                    logger.debug(f'No metrics for: {lbl}')
                continue
            ###fix this since one side does not use the same classes and it does not have positive classes
            ####just override make it quick
            pos_classes = [i for i in self.lbl_to_pos_neg[lbl]['pos'] if type(i) == int]
            pos_removed = np.array([self.removed_classes_count[splittag][lbl][i] for i in pos_classes])
            total_pos = np.array([self.removed_classes_count[splittag][lbl][f'Total:{i}'] for i in pos_classes])
            
            neg_classes = [i for i in self.lbl_to_pos_neg[lbl]['neg'] if type(i) == int]
            neg_removed = np.array([self.removed_classes_count[splittag][lbl][i] for i in neg_classes])
            total_neg = np.array([self.removed_classes_count[splittag][lbl][f'Total:{i}'] for i in neg_classes])

            metric_dict = {}
            metric_args = (pos_removed, total_pos, neg_removed, total_neg)
            for metric_func in [metrics.get_accuracy, metrics.get_precision, metrics.get_recall, metrics.get_f1]:
                metric_dict.update(metric_func(*metric_args))

            self.performance_metrics[lbl] = metric_dict
        self.results['performance_metrics'] = self.performance_metrics

        if print_perf:
            self.print_performance_metrics()


    def print_performance_metrics(self):
        logger.debug("Performance Metrics: ")
        logger.debug(json.dumps(self.performance_metrics, indent=4))

    def experiment_report(self, iteration, splittag):
        blobclass_dict = {i:0 for i in self.all_blobclasses}
        removed_classes_count_total = {self.A_TAG: {**blobclass_dict}, self.B_TAG: {**blobclass_dict}}
        for inst, lbl in self.classified_instances[splittag].items():
            removed_classes_count_total[lbl][self.inst_to_class[lbl][inst]] += 1

        for lbl in removed_classes_count_total:
            for blobclass in removed_classes_count_total[lbl]:
                self.removed_classes_count[splittag][lbl][blobclass] = removed_classes_count_total[lbl][blobclass]
        
        self.print_removed_class_count(splittag)
        if splittag == 'valid':
            self.calculate_performance_metrics()
            self.save_metrics_to_result()
            
    def save_metrics_to_result(self):
        A_metrics = self.performance_metrics[self.A_TAG]
        B_metrics = self.performance_metrics[self.B_TAG]

        combined_metrics = {}
        for metric in A_metrics:
            metric_average = A_metrics[metric]
            if metric in B_metrics:
                metric_average = (A_metrics[metric] + B_metrics[metric]) / 2
            combined_metrics[metric] = metric_average
            
        self.results['metrics'].append({'A_metric': A_metrics, 
                                        'B_metric': B_metrics, 
                                        'combined_metrics': combined_metrics,
                                        'valid_removed': copy.deepcopy(self.removed_classes_count['valid'])})