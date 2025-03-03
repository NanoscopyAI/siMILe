import copy 
import numpy as np # type: ignore

from multiprocessing import Pool
from functools import partial

from CHWOS.SIMILE.classification import classify_and_cut
from CHWOS.SIMILE.svm import train
from CHWOS.SIMILE.inference import predict_bag, aggregate_predictions, _predict_instances, compute_all_accuracy
from CHWOS.SIMILE.process_child import choose_trainer

from CHWOS.utils.save import shelve_data
from CHWOS.utils.timer import timer_decorator

def SIMILE(dataset, ss):
    exp_config = dataset.exp_config
    
    print('RUN PARAMS:')
    print(f"AE: {exp_config.AE}")
    print(f"SYM_C: {exp_config.SYM_C}")
    print(f"C: {exp_config.C}")
    print(f"minacc: {exp_config.MIN_ACC}")
    print(f"sigma: {exp_config.sigmas}")
    print(f"bagsize: {exp_config.bagsize}")
    print(f"foldindex: {exp_config.fold_split_idx}")

    itercounter = 0
    dataset.itercounter = itercounter
    while(itercounter < exp_config.MAX_ITER if exp_config.MAX_ITER > 0 else True):
        if exp_config.MAX_ITER != 1 and exp_config.AE:
            print('On iteration: ', itercounter)

        param_train_iter_pred_dict = [copy.deepcopy(dataset.clean_train_iter_predictions) for i in range(exp_config.TOTAL_PROCESS_COUNT)]
        param_valid_iter_pred_dict = [copy.deepcopy(dataset.clean_valid_iter_predictions) for i in range(exp_config.TOTAL_PROCESS_COUNT)]

        train_partial = partial(train, train_bags=dataset.train_bags, train_bag_labels=dataset.train_bag_labels, bagsize=exp_config.bagsize, C=exp_config.C)
        predict_instance_partial = partial(_predict_instances, miles_predict=not exp_config.SYM_C)
        compute_all_accuracy_partial = partial(compute_all_accuracy, A_TAG=dataset.A_TAG, B_TAG=dataset.B_TAG, mil_labels=exp_config.mil_labels)


        classify_and_cut_train = partial(classify_and_cut, iteration=itercounter, dataset=dataset, splittag='train')
        classify_and_cut_valid = partial(classify_and_cut, iteration=itercounter, dataset=dataset, splittag='valid')
        choose_trainer_partial = partial(choose_trainer, min_acc=exp_config.MIN_ACC, high_sig_acc=exp_config.HIGH_SIG_ACC, A_TAG=dataset.A_TAG, B_TAG=dataset.B_TAG)    

        @timer_decorator
        def train_all():
            if len(exp_config.sigmas) == 1:
                trainers = [train_partial(exp_config.sigmas[0])]
            else:
                with Pool(exp_config.MAX_CONCUR_PROCESS_COUNT) as pool:
                    trainers = pool.map(train_partial, exp_config.sigmas)
            return trainers

        @timer_decorator
        def predict_for_all_trainers(trainers):
            pool = Pool(exp_config.MAX_CONCUR_PROCESS_COUNT) 
            split_size_bag_pred_train = 1
            split_size_bag_pred_val = 1

            split_size_inst_pred_train = 1
            split_size_inst_pred_val = 1

            split_bags_train = np.array_split(dataset.train_bags, split_size_bag_pred_train)
            split_bags_valid = np.array_split(dataset.valid_bags, split_size_bag_pred_val)
            split_bag_pred_train = []
            split_bag_pred_valid = []
            for i in range(len(trainers)):
                if split_size_bag_pred_train == 1: #not really correct, should probably still be pool outside of the grdisearch
                    split_bag_pred_train.append([predict_bag(trainers[i], dataset.train_bags)])
                else:
                    split_bag_pred_train.append(pool.starmap_async(predict_bag, [(trainers[i], split_bags_train[j]) for j in range(split_size_bag_pred_train)]))
                
                if split_size_bag_pred_val == 1:
                    split_bag_pred_valid.append([predict_bag(trainers[i], dataset.valid_bags)])
                else:
                    split_bag_pred_valid.append(pool.starmap_async(predict_bag, [(trainers[i], split_bags_valid[j]) for j in range(split_size_bag_pred_val)]))
            
            
            split_bags_train = np.array_split(dataset.train_bags, split_size_inst_pred_train)
            split_bags_valid = np.array_split(dataset.valid_bags, split_size_inst_pred_val)
            split_train_inst_preds = []
            split_valid_inst_preds = []
            for i in range(len(trainers)):
                if split_size_inst_pred_train == 1:
                    split_train_inst_preds.append([predict_instance_partial(trainers[i], dataset.train_bags)])
                else:
                    split_train_inst_preds.append(pool.starmap_async(predict_instance_partial, [(trainers[i], split_bags_train[j]) for j in range(split_size_inst_pred_train)]))
                
                if split_size_inst_pred_val == 1:
                    split_valid_inst_preds.append([predict_instance_partial(trainers[i], dataset.valid_bags)])
                else:
                    split_valid_inst_preds.append(pool.starmap_async(predict_instance_partial, [(trainers[i], split_bags_valid[j]) for j in range(split_size_inst_pred_val)]))


            pool.close()
            pool.join()
            if split_size_inst_pred_val > 1:
                split_bag_pred_valid = [i.get() for i in split_bag_pred_valid]
                split_valid_inst_preds = [i.get() for i in split_valid_inst_preds]
            
            if split_size_inst_pred_train > 1:
                split_bag_pred_train = [i.get() for i in split_bag_pred_train]
                split_train_inst_preds = [i.get() for i in split_train_inst_preds]
                

            acc_train = [compute_all_accuracy_partial(np.hstack(accs), dataset.train_bag_labels) for accs in split_bag_pred_train]
            acc_valid = [compute_all_accuracy_partial(np.hstack(accs), dataset.valid_bag_labels) for accs in split_bag_pred_valid]

            inst_train_predictions = [aggregate_predictions(preds, param_train_iter_pred_dict[i]) for i, preds in enumerate(split_train_inst_preds)]
            inst_valid_predictions = [aggregate_predictions(preds, param_valid_iter_pred_dict[i]) for i, preds in enumerate(split_valid_inst_preds)]

            return acc_train, acc_valid, inst_train_predictions, inst_valid_predictions


        trainers = train_all()
        acc_train, acc_valid, inst_train_predictions, inst_valid_predictions = predict_for_all_trainers(trainers)
        chosen_trainer_result_dict, passed = choose_trainer_partial(zip(trainers, acc_train, inst_train_predictions, acc_valid, inst_valid_predictions, exp_config.sigmas))
        
        if not passed:
            print('None valid, exiting')
            return False 
        
        classify_and_cut_train(chosen_trainer_result_dict['train']['ip'])
        was_cut = classify_and_cut_valid(chosen_trainer_result_dict['valid']['ip'])


        dataset.update_results(chosen_trainer_result_dict)

        if exp_config.SHELVE:
            save_name_dict = {
                'Simile': exp_config.RESULTS_PREFIX,
                'A': exp_config.A_TAG,
                'B': exp_config.B_TAG,
            }
            shelve_data(dir(), globals(), exp_config.RESULTS_LOCATION, save_name_dict, include_date=True)
        
        if not was_cut:
            if exp_config.AE:
                print(f'No changes to instance classification on iteration {itercounter}')
            else:
                print('No adversarial erasing, finishing up...')
            return False
        

        itercounter += 1
        dataset.itercounter = itercounter

    return False