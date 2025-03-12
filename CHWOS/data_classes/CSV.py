import glob
import json

import numpy as np  # type: ignore
import pandas as pd

from CHWOS.data_classes.base import BASE_dataset
from CHWOS.utils import metrics as metrics
from CHWOS.utils.log import get_logger

logger = get_logger(__name__)


class CSV_dataset(BASE_dataset):
    def load_feature_data(self):
        for t in self.TAGS:
            self.features[t] = self.combine_csv_in_folder_to_dataframe(self.exp_config.DATALOCS[t])
            self.features[t] = self.features[t].dropna()

    def randomize_data(self):
        for t in self.TAGS:
            self.features[t] = self.features[t].sample(frac=1)

    def process_data(self):
        self.meta = {t: {} for t in self.TAGS}
        for t in self.TAGS:
            self.features[t], self.meta[t] = self.split_numeric_from_df(self.features[t])
            self.features[t] = self.features[t].to_numpy()

        parser = self.exp_config.parserEXP
        used_feats = json.loads(parser["DATA"].get("used_feats", "[]"))
        all_feature_names = json.loads(parser["DATA"].get("all_feature_names", "[]"))
        if used_feats != [] and all_feature_names != []:
            logger.info(f"Using features: {used_feats}")
            used_feats_idcs = [all_feature_names.index(i) for i in used_feats]
            for t in self.TAGS:
                self.features[t] = self.features[t][:, used_feats_idcs]

        super().process_data()

    def set_result_dict(self):
        super().set_result_dict()
        # self.results['meta'] = np.copy(self.meta)

    def combine_csv_in_folder_to_dataframe(self, location):
        if "*" in location:
            all_filenames = glob.glob(location, recursive=True)
        else:
            all_filenames = [location]

        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        return combined_csv

    def split_numeric_from_df(self, df):
        numeric_df = df.select_dtypes(include=[np.number])
        non_numeric_df = df.select_dtypes(exclude=[np.number])

        return numeric_df, non_numeric_df
