import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Args for running the script")

# Add the arguments
default_C = 50
default_minacc = 0.75
default_sigma = 1000
default_bagsize = 100

##
# Required arguments

parser.add_argument("--config", type=str, required=True, help="(required) The config path")
parser.add_argument(
    "-B",
    type=str,
    required=True,
    help="(required) First conditino to compare, corresponds to .ini",
)
parser.add_argument(
    "-A",
    type=str,
    required=True,
    help="(required) Second condition to compare, corresponds to .ini",
)

##
# Optional arguments

parser.add_argument(
    "--output",
    type=str,
    default="./OUTPUT",
    help="output path to store results and logs. Default to current directory",
)
parser.add_argument(
    "--expname",
    type=str,
    default="",
    help="experiment names, used for setting the output folder name. Default to empty",
)
# TODO: add support to export to different file formats (ie. csv, npy, ...); csv is default
#       currently, saved as .npy binary
# parser.add_argument(
#     "--ext",
#     type=str,
#     default="csv",
#     help="file extension for saving results. Default to csv",
# )  # unused

parser.add_argument(
    "--bagsize",
    type=int,
    default=default_bagsize,
    help=f"MIL bag size. Default: {default_bagsize}",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=default_sigma,
    help=f"MILES sigma value. Default: {default_sigma}",
)
parser.add_argument("--C", type=float, default=default_C, help=f"SVM C value. Default: {default_C}")
parser.add_argument(
    "--minacc",
    type=float,
    default=default_minacc,
    help=f"AE Min accuracy value. Default: {default_minacc}",
)
parser.add_argument(
    "--minacc_valid",
    type=float,
    default=0,
    help="AE Min accuracy value for validation set. Default: 0",
)
parser.add_argument(
    "--valid_pred_iters",
    type=float,
    default=0,
    help="Number of times to shuffle and predict validation score, default 0 for none",
)

parser.add_argument(
    "--foldindex",
    type=int,
    default=0,
    help="fold index valur (integer), starts at 1, or 0 for none",
)
parser.add_argument(
    "--limit_bags",
    type=int,
    default=0,
    help="limit number of bags. Default is 0 for no limit",
)
parser.add_argument(
    "--max_iter",
    type=int,
    default=0,
    help="max adversarial iterations. Default is 0 for no limit",
)
parser.add_argument(
    "--trainsplit",
    type=float,
    default=0.6,
    help="fraction to use for train split when not using a fold, combined with --validsplit should be <1, with rest going to test. Default 0.6",
)
parser.add_argument(
    "--validsplit",
    type=float,
    default=0.2,
    help="fraction to use for valid split when not using a fold, combined with --trainsplit should be <1, with rest going to test. Default 0.2",
)

parser.add_argument("--AE", type=int, default=1, help="Use adversarial erasing (1 or 0). Default: 1")
parser.add_argument("--SYM_C", type=int, default=1, help="Use symmetric classifier (1 or 0). Default: 1")
parser.add_argument("--seed", type=int, default=42, help="seed value (integer). Default 42")

parser.add_argument(
    "--CV",
    type=int,
    default=0,
    help="Number of cross validation folds 0 or >1. Default: 0 for not in use",
)
parser.add_argument(
    "--NCV_fold",
    type=int,
    default=0,
    help="which nested CV fold number to run, 0 for off, 1 for first fold, etc. Default 0. If --CV is not set, default if 5 folds",
)
parser.add_argument(
    "--use_ncv_test",
    type=int,
    default=0,
    help="Use nested CV outer testset, default 0 for off, only used if NCV_fold > 0",
)

parser.add_argument("--sweep_output_path", type=str, default="", help="Path to output values for sweep")
parser.add_argument("--sweep_param_path", type=str, default="", help="Path to get sweep parameters")
parser.add_argument(
    "--create_sweep_param_file",
    type=str,
    default="",
    help="Path to get sweep parameters from",
)

parser.add_argument("--save", action="store_true")
parser.add_argument("--run_saved", type=str, default="", help="Path to saved models, used when provided")

parser.add_argument("--all", action="store_true")


# eg: --config SIMILE/CHWOS/configs/CHWOS/CHWOS_1.ini --AE 1 --SYM_C 1 --C 0.5 --minacc 0.75 --sigma 2000 --bagsize 50 --foldindex 0
# mock:: --C 0.1 --minacc 0.65 --sigma 100 --bagsize 50
def get_args():
    args = parser.parse_args()
    return args
