from CHWOS.mil.models import LinearSVC
from CHWOS.mil.models.model import Classifier
from CHWOS.mil.bag_representation.mapping import MILESMapping 
import numpy as np # type: ignore
import sys

from CHWOS.utils.log import get_logger
logger = get_logger(__name__)

class MILES(Classifier):
    """
    Mapping bags to a instance based feature space, from paper
    MILES: Multiple-Instance Learning via Embedded Instance Selection (Chen et al.)
    http://infolab.stanford.edu/~wangz/project/imsearch/SVM/PAMI06/chen.pdf
    """
    def __init__(self, sigma2=9**5, C=0.5, svm_max_iter=100000, verbose=True):
        """
        Parameters
        ----------
        sigma2 : parameter sigma^2 in line 4 of Algorithm 4.1 in MILES paper. #scaling for distance metric
        C : float, regularizer parameter of linear svm.
        """
        self.sigma2 = sigma2
        self.C = C
        self.svm_max_iter = svm_max_iter
        self.verbose = verbose
        self.first_predict_print = True
    
    def fit(self, X, y, **kwargs):
        """ 
        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        y : array-like containing all the training labels.
        """
        self.check_exceptions(X)
        # mapping bags to the instance based feature space
        
        logger.info("Mapping bags")
        self.mapping = MILESMapping(self.sigma2)
        mapped_bags = self.mapping.fit_transform(X) #maps to the similarity space in paper 2.3 

        #train the SVM
        
        self.model = LinearSVC(penalty="l1", C=self.C, dual=False, # number of instances equals number of features so dual/primal shouldnt matter
                               class_weight='balanced', max_iter=self.svm_max_iter, verbose=self.verbose)
        logger.info("Fitting bags")
        self.model.fit(mapped_bags, y, **kwargs)
        
        # get parameters from SVM
        self.coef_ = self.model.coef_[0]
        self.intercept_ = self.model.intercept_
        
        return self
        
    def predict(self, X):
        """ 
        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        """
        self.check_exceptions(X)
        # mapping bags to the instance based feature space
        mapped_bags = self.mapping.transform(X)    
        
        #predict classes
        return self.model.predict(mapped_bags)

    def predict_instances(self, bag, use_threshold=False):
        if self.first_predict_print:
            logger.debug(f'INSTANCE PREDICTION: Predicting with threshold: {use_threshold}')
            self.first_predict_print = False
        #argminIDX contains the index of the instance in the bag closest to the ith instance of iip
        mapped_bag, argminIDX = self.mapping.transform(bag, return_argmin=True) #now its mapped to similarity measures
        #argminIDX = argminIDX[0,:]
        #The rows should each be a different bag and its tranformation into the similarity space. Currently
        #just doing one bag so it doesnt matter. 

        #First value is a set of bag indices which are the most similiar to a given pool instance, second is the pool instance it was closest to
        #There are repeats if multiple instances have the same closest similarity
        inst_sim_idx, pool_sim_idx = argminIDX[0]


        mapped_bag = mapped_bag[0] 

        #Only the weights which are non-zero are important, so we only use those similarities which are used in the svm
        weight_filter = (np.abs(self.coef_) > 0)#np.finfo(float).eps #keep at >0 or introduce machine error?
        if np.sum(weight_filter) == 0:
            return []
        
        #The indicies of the similarities which are used in the svm
        used_sims_by_idx = np.array(range(len(self.coef_)))[weight_filter]

        #For each instance that is important for classification, we record the instances in the pool which are most similar to it
        #excluding those pool instances which are not used in the svm 
        inst_to_simidx = {}
        for i in range(bag[0].shape[0]): 
            if i not in inst_sim_idx:
                continue

            i_sim_idx = inst_sim_idx == i
            i_pool_sim_idx = pool_sim_idx[i_sim_idx]

            inst_to_simidx[i] = [j for j in i_pool_sim_idx if j in used_sims_by_idx] #only use the ones which are used in the svm

        #Record for every pool instance used in the svm, the number of instances with equal closest similarity
        sim_to_mk = {}
        for i in used_sims_by_idx:
            sim_to_mk[i] = np.sum(i == pool_sim_idx)
        
        
        threshold = -1*self.intercept_[0]/len(inst_to_simidx.keys()) #threshold for postive (negative) if above (below) value, section 3.2

        t = []
        
        for i in range(bag[0].shape[0]):
            if i not in inst_to_simidx:
                t.append([i,0])
                continue

            g = 0
            for sim_idx in inst_to_simidx[i]:
                wk = np.array(self.coef_)[sim_idx]
                sim = np.array(mapped_bag)[sim_idx]
                mk = sim_to_mk[sim_idx]
                g += (wk*sim / mk)

            if use_threshold:
                g = 1 if g > threshold else -1
            t.append([i,g]) #return all the values instead of classifying directly here. Let user decide
        
        

        return np.array(t)
