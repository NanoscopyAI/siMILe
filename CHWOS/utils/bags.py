import numpy as np # type: ignore
from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

    
#generate bags for train, valid, test,
#features should be {'A': [], 'B': []}
def form_bags(feature_set, bagsize): 
        num_bags =  feature_set.shape[0] // bagsize
        bags = np.array_split(feature_set, num_bags )        
        return np.array(bags, dtype=object)


def generate_train_test_validation_bags(features, A_TAG, B_TAG, split, bagsize, mil_labels, seed=None):
    if seed:
        np.random.seed(seed) 

    def form_bags_and_labels(feature_set): 
        A_n =  feature_set[A_TAG].shape[0] // bagsize
        bags = np.array_split(feature_set[A_TAG], A_n)
        
        B_n = feature_set[B_TAG].shape[0] // bagsize
        bags.extend(np.array_split(feature_set[B_TAG], B_n))
        
        bag_labels = np.hstack([np.ones(A_n)*mil_labels[1], np.ones(B_n)*mil_labels[0]])
    
        _temp = list(zip(bags, bag_labels))
        np.random.shuffle(_temp)
        bags, bag_labels = zip(*_temp)

        bags = np.array(bags, dtype=object)
        bag_labels = np.array(bag_labels)
        
        return bags, bag_labels
    
    
    pos_array = np.copy(features[A_TAG]); neg_array = np.copy(features[B_TAG])
    np.random.shuffle(pos_array); np.random.shuffle(neg_array)
    trainF = {}; validF = {}; testF = {}

    if split[0] == 1:
        trainF[A_TAG] = pos_array
        trainF[B_TAG] = neg_array
        train_bags, train_bag_labels = form_bags_and_labels(trainF)
        return train_bags, train_bag_labels

    pos_split_idx = [int(len(pos_array)*split[0]), int(len(pos_array)*(split[0]+split[1]))]
    neg_split_idx = [int(len(neg_array)*split[0]), int(len(neg_array)*(split[0]+split[1]))]

    trainF[A_TAG], validF[A_TAG], testF[A_TAG] = np.split(pos_array, pos_split_idx)
    trainF[B_TAG], validF[B_TAG], testF[B_TAG] = np.split(neg_array, neg_split_idx)

    train_bags, train_bag_labels = form_bags_and_labels(trainF)
    valid_bags, valid_bag_labels = form_bags_and_labels(validF)    
    test_bags, test_bag_labels = form_bags_and_labels(testF)

    return train_bags, train_bag_labels,\
            test_bags, test_bag_labels, \
            valid_bags, valid_bag_labels


#includes those given
def mask_bag_features(bags, idx=[]):
    bags = np.array([bag[:,idx] for bag in bags], dtype=object)
    return bags

def reform_bags_and_labels(bags, labels): 
    unq_labels = np.unique(labels)
    new_bags = []
    new_labels = []
    for lbl in unq_labels:
        _bags = bags[labels == lbl]
        _bags = shuffle_bags(_bags)
        
        #logger.info(f'@@@ {lbl} num_lbls {_bags.shape[0]}')
        #logger.info(f'@@@ {lbl} num_lbls {_bags.shape[0]} bagsize {_bags.shape}')
        _new_lbls = [lbl] * _bags.shape[0]
        
        if len(_bags.shape) == 3:
            _bags = [_bags[i, :, :] for i in range(_bags.shape[0])]
        else:
            _bags = _bags.tolist()
            
        new_bags.extend(_bags)
        new_labels.extend(list(_new_lbls))
        
    new_bags = np.array(new_bags, dtype=object)
    new_labels = np.array(new_labels)
    #logger.info(f'@@@ new_bags shape {new_bags.shape[0]},  {new_bags[0].shape}')
    #logger.info(f'@@@ new_lbls shape {new_labels.shape[0]}')
    return new_bags, new_labels
        
def shuffle_bags(bags):
    bagsize = bags[-1].shape[0]
    #logger.info(f'@@@ BAGSIZE {bagsize}')
    all_instances = np.array([inst for bag in bags for inst in bag])
    np.random.shuffle(all_instances)
    bags = form_bags(all_instances, bagsize)
    return bags


def get_split_bags_and_labels(featuresSplit, exp_config, gave_splits=False):
    mil_labels = exp_config.mil_labels
    A_TAG, B_TAG = exp_config.A_TAG, exp_config.B_TAG
    if gave_splits: #Used with folds
        train_bags, train_bag_labels = generate_train_test_validation_bags(featuresSplit['train'], A_TAG=A_TAG, B_TAG=B_TAG, split=[1,0,0], bagsize=exp_config.bagsize, mil_labels=mil_labels)
        valid_bags, valid_bag_labels = generate_train_test_validation_bags(featuresSplit['valid'], A_TAG=A_TAG, B_TAG=B_TAG, split=[1,0,0], bagsize=exp_config.bagsize, mil_labels=mil_labels)
        test_bags, test_bag_labels = np.array([]), np.array([])
    else:
        train_bags, train_bag_labels,\
        test_bags, test_bag_labels, \
        valid_bags, valid_bag_labels = generate_train_test_validation_bags(featuresSplit['train'], A_TAG=A_TAG, B_TAG=B_TAG, split=[exp_config.train_split, exp_config.test_split, exp_config.valid_split], bagsize=exp_config.bagsize, mil_labels=mil_labels)

        _a_bags = train_bags[train_bag_labels == mil_labels[1]]
        _b_bags = train_bags[train_bag_labels == mil_labels[0]]
        featuresSplit['train'][A_TAG] = np.array([inst for bag in _a_bags for inst in bag])
        featuresSplit['train'][B_TAG] = np.array([inst for bag in _b_bags for inst in bag])

        _a_bags = valid_bags[valid_bag_labels == mil_labels[1]]
        _b_bags = valid_bags[valid_bag_labels == mil_labels[0]]
        featuresSplit['valid'][A_TAG] = np.array([inst for bag in _a_bags for inst in bag])
        featuresSplit['valid'][B_TAG] = np.array([inst for bag in _b_bags for inst in bag])

        _a_bags = test_bags[test_bag_labels == mil_labels[1]]
        _b_bags = test_bags[test_bag_labels == mil_labels[0]]
        featuresSplit['test'][A_TAG] = np.array([inst for bag in _a_bags for inst in bag])
        featuresSplit['test'][B_TAG] = np.array([inst for bag in _b_bags for inst in bag])


    if exp_config.LIMIT_BAGS:
        logger.debug(f'Limiting train bag count to: {exp_config.LIMIT_BAGS_SIZE}')
        train_bags = train_bags[:exp_config.LIMIT_BAGS_SIZE]
        train_bag_labels = train_bag_labels[:exp_config.LIMIT_BAGS_SIZE]

    if train_bags.size > 0:
        logger.debug(f'train bag shape: {train_bags[-1].shape}. # bags: {len(train_bag_labels)}')
    if valid_bags.size > 0:
        logger.debug(f'valid bag shape: {valid_bags[-1].shape}. # bags: {len(valid_bag_labels)}')
    if test_bags.size > 0:
        logger.debug(f'test bag shape: {test_bags[-1].shape}. # bags: {len(test_bag_labels)}')

    return featuresSplit, train_bags, train_bag_labels, valid_bags, valid_bag_labels, test_bags, test_bag_labels