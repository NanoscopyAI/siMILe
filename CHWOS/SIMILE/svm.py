from CHWOS.mil.preprocessing import StandarizerBagsList
from CHWOS.mil.models import MILES
from CHWOS.mil.trainer import Trainer
from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

def train(sigma2, train_bags, train_bag_labels, bagsize, C):
    logger.debug(f'Training with: C:{C}, sigma:{sigma2}, bagsize:{bagsize}')
    trainer = Trainer()
    model = MILES(sigma2=sigma2, C=C, svm_max_iter=100000, verbose=0)
    
    pipeline = [('scale', StandarizerBagsList())]

    trainer.prepare(model, preprocess_pipeline=pipeline)
    trainer.fit(X_train=train_bags,\
                y_train=train_bag_labels, \
                verbose=1)

    return trainer
