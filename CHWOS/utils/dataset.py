from CHWOS.data_classes.Mock import Mock_dataset
from CHWOS.data_classes.PC3PTRF import PC3PTRF_dataset
from CHWOS.data_classes.Cav1Mutant import Cav1Mutant_dataset
from CHWOS.data_classes.CSV import CSV_dataset
from CHWOS.data_classes.dstorm_sim import dstorm_sim_dataset
from CHWOS.data_classes.hela_dynasore_pitstop import hela_dynasore_pitstop_dataset

from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

def get_dataset(config, **kwargs):
    dataset = None
    

    if config.DATA_NAME == 'MOCK':
        dataset = Mock_dataset(config, **kwargs)
    elif config.DATA_NAME == 'PC3PTRF':
        dataset = PC3PTRF_dataset(config, **kwargs)
    elif config.DATA_NAME == 'Cav1Mutant':
        dataset = Cav1Mutant_dataset(config, **kwargs)
    elif ('dstorm_sim' in config.DATA_NAME) or ('csv_wperf' in config.DATA_NAME):
        dataset = dstorm_sim_dataset(config, **kwargs)
    elif 'hela' in config.DATA_NAME:
        if 'class2' in config.DATA_NAME:
            dataset = CSV_dataset(config, **kwargs)
        else:
            dataset = hela_dynasore_pitstop_dataset(config, **kwargs)
    elif config.datatype == 'csv':
        dataset = CSV_dataset(config, **kwargs)
    else:
        raise ValueError(f'No dataset found: {config.DATA_NAME}, exiting')
    
    logger.debug(f'Dataset created: {config.DATA_NAME}, {type(dataset)}')
    return dataset