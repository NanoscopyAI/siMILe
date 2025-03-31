import fcntl
import json
import os
from itertools import product

import numpy as np

from CHWOS.SIMILE.train import run as start_train
from CHWOS.utils.dataset import get_dataset
from CHWOS.utils.log import get_logger

logger = get_logger(__name__)


class Sweep:
    def __init__(self, config_inst, param_path, output_path):
        self.param_path = param_path
        self.output_path = output_path
        self.config_inst = config_inst

    @staticmethod
    def create_parameter_file(w_param_path, search_range_path):
        # search range should be a config value to set and range to go over
        with open(search_range_path, "r") as f:
            param_and_range = json.load(f)

        params = param_and_range.keys()
        with open(w_param_path, "w") as f:
            header = ",".join(params)
            f.write(header + "\n")
            for items in product(*param_and_range.values()):
                param_vals = ",".join([str(i) for i in items])
                f.write(param_vals + "\n")

    def get_from_param_file(self):
        with open(self.param_path, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                lines = f.read().strip().split("\n")
                if len(lines) <= 1:
                    return None
                header = lines[0].split(",")
                last_line = lines[-1].split(",")

                f.seek(0)
                for line in lines[:-1]:
                    f.write(line + "\n")
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        result = dict(zip(header, last_line))
        return result

    def set_config_values(self, value_dict):
        for var, val in value_dict.items():
            if var == "bagsize":
                val = int(val)
            else:
                val = float(val)
            setattr(self.config_inst, var, val)

    def convert_keys(self, obj):
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if isinstance(key, np.integer):
                    key = int(key)
                elif not isinstance(key, (str, int, float, bool, type(None))):
                    raise TypeError(f"Unsupported key type: {type(key)}")
                new_obj[key] = self.convert_keys(value)
            return new_obj
        elif isinstance(obj, list):
            return [self.convert_keys(item) for item in obj]
        else:
            return obj

    def writeout_metrics(self, params, metrics):
        write_str = json.dumps(self.convert_keys(params)) + "|" + json.dumps(self.convert_keys(metrics)) + "\n"
        with open(self.output_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(write_str)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def run(self):
        iterations = 0
        while True:
            iterations += 1
            value_dict = self.get_from_param_file()
            if value_dict is None:
                break

            logger.info(f"{iterations}: Running sweep at params: {json.dumps(value_dict)}")
            self.set_config_values(value_dict)
            dataset = get_dataset(self.config_inst)
            dataset.save = False

            try:
                start_train(dataset=dataset)

                metrics = dataset.results.get("metrics", [])
                valid_accs = dataset.results["valid"].get("acc", [])
                train_accs = dataset.results["train"].get("acc", [])
                writeout_dict = {
                    "metrics": metrics,
                    "valid_accs": valid_accs,
                    "train_accs": train_accs,
                }
                print(value_dict)
                print(writeout_dict)
                self.writeout_metrics(value_dict, writeout_dict)
            except Exception as e:
                print(e)
                self.writeout_metrics(value_dict, {})
