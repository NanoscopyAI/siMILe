import numpy as np 
from CHWOS.utils.log import get_logger

logger = get_logger(__name__)

def predict_bag(trainer, bags):
    predlabels = trainer.predict(bags)
    return predlabels


def compute_all_accuracy(predlabels, labels, A_TAG, B_TAG, mil_labels):
    acc = {'avg':0, B_TAG:0, A_TAG:0}

    correct_pred = predlabels == labels

    filter_A = labels == mil_labels[1]
    filter_A_len = np.sum(filter_A)
    acc[A_TAG] = np.sum(correct_pred[filter_A])/filter_A_len

    filter_B = labels == mil_labels[0]
    filter_B_len = np.sum(filter_B)
    acc[B_TAG] = np.sum(correct_pred[filter_B])/filter_B_len

    A_w = filter_A_len/len(labels)
    B_w = filter_B_len/len(labels) 
    acc['avg'] += (acc[A_TAG]*A_w + acc[B_TAG]*B_w)

    return acc


def combine_accuracy(accs, weights):
    return_acc = {k:0 for k in accs[0].keys()}
    for a, w in zip(accs, weights):
        for k in a.keys():
            return_acc[k] += a[k]*w
    return return_acc


def predict_instances(trainer, bags, iter_predictions, miles_predict=False):
    p = _predict_instances(trainer, bags, miles_predict=miles_predict)
    return aggregate_predictions([p], iter_predictions)


def aggregate_predictions(preds, iter_predictions):
    for p in preds:
        if type(p) != list and p.size != 0:
            for j in range(p.shape[0]):
                iter_predictions[tuple(p[j,:-1])].append(p[j,-1:])
    return iter_predictions


def _predict_instances(trainer, bags, miles_predict=False):
    tot = 0
    instance_predictions = []

    for i in range(len(bags)):
        bag_to_predict = bags[i]
        
        in_pred = trainer.predict_instances([bag_to_predict], use_threshold=miles_predict)

        if len(in_pred) > 0:
            ins = bag_to_predict[in_pred[:,0].astype(np.int64),:]
            ins = np.hstack((ins, in_pred[:,1][None].T))
            
            if len(instance_predictions) == 0:
                instance_predictions = ins
            else:
                instance_predictions = np.vstack((instance_predictions, ins))
        

        tot += len(in_pred)

    return instance_predictions