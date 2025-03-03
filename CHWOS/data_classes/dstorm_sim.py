import copy
import json
import glob
import pandas as pd
import math

import numpy as np # type: ignore
from collections import defaultdict
from CHWOS.data_classes.CSV import CSV_dataset
from CHWOS.utils import metrics as metrics

from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

#similiar to MOCK, setup to see the results as it runs. Similiar code but using csv 
#and improved for clarity
class dstorm_sim_dataset(CSV_dataset):
    def __init__(self, exp_config, setup=True):
        self.report_vars_set = False
        super().__init__(exp_config, setup=setup)
        
    
    def process_data(self):
        super().process_data()
        for t in self.TAGS:
            self.meta[t]['class'] = self.meta[t]['class'].str.replace('c', '').astype(int)
        #self.results['meta'] = np.copy(self.meta)
        self.results['metrics'] = []
        self.setup_class_record()
    
    def create_and_set_split_bags_and_labels(self, gave_splits):
        super().create_and_set_split_bags_and_labels(gave_splits)
        if not self.report_vars_set:
            self.set_reporting_variables()
        
    def set_post_step_functions(self):
        super().set_post_step_functions()
        self.post_step['classify_and_cut'] = self.experiment_report
        
    def setup_class_record(self):
        self.class_record = {}
        self.all_classes = {}
        self.class_count = {}
        for t in self.TAGS:
            self.class_record[t] = {}
            for i, f in enumerate(self.features[t]):
                self.class_record[t][tuple(f)] = self.meta[t]['class'].iloc[i]
        
            self.all_classes[t], self.class_count[t] = np.unique(list(self.class_record[t].values()), return_counts=True)
        
        logger.debug(f'All classes: {self.all_classes}')
        logger.debug(f'Class count: {self.class_count}')
        
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
            self.tag_to_common_distinct_class[t] = {'distinct': [], 'common': [], 'all': []}
            for c in self.all_classes[t]:

                if self.class_count_by_split['valid'][t][c] == 0:
                    continue

                self.tag_to_common_distinct_class[t]['all'].append(c)

        for t, t2 in [(self.A_TAG, self.B_TAG), (self.B_TAG, self.A_TAG)]:
            for c in self.tag_to_common_distinct_class[t]['all']:
                self.tag_to_common_distinct_class[t]['common' if c in self.tag_to_common_distinct_class[t2]['all'] else 'distinct'].append(c)
        logger.debug(f'Tag classes: {self.tag_to_common_distinct_class}')
        self.report_vars_set = True
        
    def print_removed_class_count(self, split):
        printstr = f'{split.upper()}: REMOVED CLASS COUNT: \n'
        for t in self.removed_classes_count[split]:
            printstr += f"{t}: \n"
            for c in self.removed_classes_count[split][t]:
                printstr += f"    Class {c}: {self.removed_classes_count[split][t][c]} / {self.class_count_by_split[split][t][c]}\n"
        logger.debug(printstr)
        
    def experiment_report(self, iteration, split):
        removed_classes_count_total = {self.A_TAG: {i:0 for i in self.all_classes[self.A_TAG]}, self.B_TAG: {i:0 for i in self.all_classes[self.B_TAG]}}
        for inst, t in self.classified_instances[split].items():
            removed_classes_count_total[t][self.class_record[t][inst]] += 1

        for t in removed_classes_count_total:
            for c in removed_classes_count_total[t]:
                self.removed_classes_count[split][t][c] = removed_classes_count_total[t][c]
        
        self.print_removed_class_count(split)
        
        if split == 'valid':
            self.calculate_performance_metrics()
            self.save_metrics_to_result()
            
    
    def ji(self, EXP=0.1, CONF=0.99):
        bagsize_per_class = []
        for tag in self.removed_classes_count['train'].keys():
            total_removed = 0
            total = 0
            for cls, count in self.removed_classes_count['train'][tag].items():
                total_removed += count
            for cls, count in self.class_count_by_split['train'][tag].items():
                total += count
            
            FRACTION_REMOVED = total_removed/total
        
            PROP_REMAINING = (EXP * (1-FRACTION_REMOVED))/(1-EXP*FRACTION_REMOVED)
            N = math.log(1 - CONF) / math.log(1 - PROP_REMAINING)
            bagsize_per_class.append(N)
            
            logger.info(f'{tag}| N: {N} FRACTION_REMOVED: {FRACTION_REMOVED} removed: {total_removed}, total: {total}')        

        logger.info(f'Using bagsize: {min(bagsize_per_class)}')
        return min(bagsize_per_class)
        
    def calculate_performance_metrics(self, print_perf=True):
        self.performance_metrics = {}
        split = 'valid'
        for t in self.removed_classes_count[split]:
            self.performance_metrics[t] = {}

            if self.tag_to_common_distinct_class[t]['distinct'] == []:
                if print_perf:
                    logger.debug(f'No metrics for: {t}')
                continue
            ###fix this since one side does not use the same classes and it does not have positive classes
            ####just override make it quick
            distinct_classes = [i for i in self.tag_to_common_distinct_class[t]['distinct']]
            distinct_removed = np.array([self.removed_classes_count[split][t][i] for i in distinct_classes])
            total_distinct = np.array([self.class_count_by_split[split][t][i] for i in distinct_classes])
            
            common_classes = [i for i in self.tag_to_common_distinct_class[t]['common']]
            common_removed = np.array([self.removed_classes_count[split][t][i] for i in common_classes])
            total_common = np.array([self.class_count_by_split[split][t][i] for i in common_classes])

            metric_dict = {}
            metric_args = (distinct_removed, total_distinct, common_removed, total_common)
            for metric_func in [metrics.get_accuracy, metrics.get_precision, metrics.get_recall, metrics.get_f1]:
                metric_dict.update(metric_func(*metric_args))

            self.performance_metrics[t] = metric_dict
        #self.results['performance_metrics'] = self.performance_metrics

        if print_perf:
            self.print_performance_metrics()
            
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
        
    def print_performance_metrics(self):
        logger.debug("Performance Metrics: ")
        logger.debug(json.dumps(self.performance_metrics, indent=4))