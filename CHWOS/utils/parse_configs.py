from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path


def get_config(config_path: Namespace):
    exp_config = EXP_CONFIG(config_path)

    if exp_config.SYM_C:
        exp_config.mil_labels = [-1, 1]  # type: ignore hack
    else:
        exp_config.mil_labels = [0, 1]  # type: ignore hack

    return exp_config


class EXP_CONFIG:
    def __init__(self, args: Namespace):
        self.args = args

        self.parserEXP = ConfigParser()
        self.parserEXP.read(args.config)
        if not self.parserEXP.sections():
            raise ValueError(f"No configuration settings found in {args.config}, probably the wrong path")

        self.DATA_BASE_PATH = self.parserEXP["DATA"]["base_path"] + "/"
        # self.datatype = args.ext  # unused

        self.LIMIT_BAGS_SIZE = args.limit_bags
        self.LIMIT_BAGS = bool(self.LIMIT_BAGS_SIZE)

        self.AE = bool(args.AE)
        self.SYM_C = bool(args.SYM_C)
        self.C = args.C
        self.sigma = args.sigma
        self.bagsize = args.bagsize
        self.MIN_ACC = args.minacc
        self.MIN_ACC_VALID = args.minacc_valid
        self.fold_split_idx = args.foldindex
        self.NCV_fold = args.NCV_fold
        self.use_ncv_test = bool(args.use_ncv_test)
        self.MAX_ITER = args.max_iter
        self.seed = args.seed

        self.valid_pred_iters = args.valid_pred_iters

        self.save_models = args.save
        self.run_saved = args.run_saved
        self.run_all = args.all

        self.DATA_NAME = self.parserEXP["DATA"]["name"]
        self.A_TAG = args.A
        self.B_TAG = args.B
        self.A_path = self.parserEXP["DATA"][self.A_TAG]
        self.B_path = self.parserEXP["DATA"][self.B_TAG]

        if self.DATA_NAME == "MOCK":
            self.A_TAG = int(self.A_TAG)
            self.B_TAG = int(self.B_TAG)

        self.DATALOCS = {
            self.A_TAG: self.DATA_BASE_PATH + self.A_path,
            self.B_TAG: self.DATA_BASE_PATH + self.B_path,
        }

        self.setup_results_and_log(args)

        self.train_split = args.trainsplit
        self.valid_split = args.validsplit
        self.test_split = 1 - self.train_split - self.valid_split
        assert self.test_split >= 0, "train and valid split size too large"

    def setup_results_and_log(self, args):
        output_path = Path(args.output)
        if not output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)

        parent_folder = f"SiMILe-M_{self.A_TAG}_vs_{self.B_TAG}_output_{args.expname}/"
        parent_folder_path = output_path / parent_folder

        self.RESULTS_LOCATION = parent_folder_path / "results"
        self.MODEL_LOCATION = parent_folder_path / "model"
        self.LOG_LOCATION = parent_folder_path / "logs"

        checked_paths = [self.RESULTS_LOCATION, self.LOG_LOCATION]
        if self.save_models:
            checked_paths.append(self.MODEL_LOCATION)

        for newpath in checked_paths:
            try:
                if not newpath.exists():
                    newpath.mkdir(parents=True, exist_ok=True)
            except Exception as _:
                pass
