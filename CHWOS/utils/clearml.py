import sys
import json
from clearml import Task

CLEARML_STARTED = False

def setup_clearml(exp_config, mlflow_str):
    return
    task = Task.init(task_name=f'SIMILE-{mlflow_str}', project_name=exp_config.clearml_taskname, task_type=Task.TaskTypes.training)
    task.connect_configuration({s:dict(exp_config.parserEXP.items(s)) for s in exp_config.parserEXP.sections()}, name='Experiment Config')
    rest_config = {
        'C': exp_config.C,
        'bagsize': exp_config.bagsize,
        'sigma': exp_config.sigma,
        'MIN_ACC': exp_config.MIN_ACC,
        'HIGH_SIG_ACC': exp_config.HIGH_SIG_ACC,
        'TOTAL_PROCESS_COUNT': exp_config.TOTAL_PROCESS_COUNT,

    }
    task.connect_configuration(rest_config, name='Params')

    return task


def start_clearml(config):
    return
    global CLEARML_STARTED

    if CLEARML_STARTED:
        return None
    
    mlflow_str = str(sys.argv[2])
    task = setup_clearml(config, mlflow_str)
    config.mlflow_str = mlflow_str
    CLEARML_STARTED = True

    return task


def clearml_upload(name, data):
    return
    task = Task.current_task()
    task.upload_artifact(name, json.dumps(data, indent=4))


def close_clearml():
    return
    if not CLEARML_STARTED:
        return

    task = Task.current_task()
    task.close()