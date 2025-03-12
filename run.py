from CHWOS.utils.args import get_args
from CHWOS.utils.log import get_logger, set_logger_path_with_config
from CHWOS.utils.misc import set_random_and_get_sequence
from CHWOS.utils.parse_configs import get_config


def run_with_cli_args(args):
    exp_config = get_config(args)
    ss = set_random_and_get_sequence(exp_config.seed)

    # import after setting log string
    set_logger_path_with_config(exp_config)
    logger = get_logger(__name__)
    logger.info(f"Seed = {exp_config.seed}")
    from CHWOS.SIMILE.train import CV, nested_CV, run, run_from_save
    from CHWOS.utils.sweep import Sweep

    if args.create_sweep_param_file != "":
        logger.info("Creating parameter file")
        Sweep.create_parameter_file(args.sweep_param_path, args.create_sweep_param_file)
    elif args.sweep_output_path != "" or args.sweep_param_path != "":
        if not (args.sweep_output_path != "" and args.sweep_param_path != ""):
            logger.error("Missing sweep output or parameter path")
        else:
            swp = Sweep(exp_config, args.sweep_param_path, args.sweep_output_path)
            logger.info(
                f"Starting sweep with param path {args.sweep_param_path} and output path {args.sweep_output_path}"
            )
            swp.run()
            logger.info("Done sweep")
    else:
        if args.NCV_fold > 0:
            if args.CV == 0:
                args.CV = 5

            nested_CV(exp_config=exp_config, n_folds=args.CV, ss=ss)
        elif args.CV > 1:
            CV(exp_config=exp_config, n_folds=args.CV, ss=ss)
        elif args.run_saved != "":
            run_from_save(exp_config=exp_config, ss=ss)
        else:
            run(exp_config=exp_config, ss=ss)
        logger.info("Done run")


if __name__ == "__main__":
    args = get_args()
    run_with_cli_args(args)
