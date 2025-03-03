from CHWOS.data_classes.Mock import Mock_dataset
from CHWOS.data_classes.CSV import CSV_dataset
from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

def get_dataset(config, **kwargs):
    dataset = None
    
    if config.DATA_NAME == 'MOCK':
        dataset = Mock_dataset(config, **kwargs)
    elif config.datatype == 'csv':
        dataset = CSV_dataset(config, **kwargs)
    else:
        raise ValueError(f'No dataset found: {config.DATA_NAME}, exiting')
    
    logger.debug(f'Dataset created: {config.DATA_NAME}, {type(dataset)}')
    return dataset