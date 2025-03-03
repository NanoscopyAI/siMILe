import copy
import json
import numpy as np # type: ignore
from collections import defaultdict
from CHWOS.data_classes.base import BASE_dataset
from CHWOS.utils import metrics as metrics

from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

class Cav1Mutant_dataset(BASE_dataset):
    def load_feature_data(self):
        self.meta = {self.A_TAG: {}, self.B_TAG: {}}
        self.features[self.A_TAG], self.meta[self.A_TAG] = np.load(self.exp_config.DATALOCS[self.exp_config.A_TAG], allow_pickle=True)
        self.features[self.B_TAG], self.meta[self.B_TAG] = np.load(self.exp_config.DATALOCS[self.exp_config.B_TAG], allow_pickle=True)

        self.features[self.A_TAG] = self.features[self.A_TAG][self.exp_config.A_TAG]
        self.features[self.B_TAG] = self.features[self.B_TAG][self.exp_config.B_TAG]
        self.meta[self.A_TAG] = self.meta[self.A_TAG][self.exp_config.A_TAG]
        self.meta[self.B_TAG] = self.meta[self.B_TAG][self.exp_config.B_TAG]
        for lbl in [self.A_TAG, self.B_TAG]:
            _temp = list(zip(self.features[lbl], self.meta[lbl]['classes']))
            np.random.shuffle(_temp)
            self.features[lbl], self.meta[lbl]['classes'] = zip(*_temp)
            self.features[lbl] = np.array(self.features[lbl])
            self.meta[lbl]['classes'] = list(self.meta[lbl]['classes'])

        
        logger.debug(f"A feature original shape: {self.features[self.A_TAG].shape}")
        logger.debug(f"B feature original shape: {self.features[self.B_TAG].shape}")

    def set_result_dict(self):
        super().set_result_dict()
        self.results['meta'] = np.copy(self.meta)