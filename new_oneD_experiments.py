import pickle
import itertools
import os
import torch
import numpy as np
import sigopt
from sklearn.model_selection import KFold
from cnn_optim_train_test import train_and_test_full_model

def train_with_found_parameters(config, dict_actors_index, actors, experiments, gpu):
    # Train / test with scrambled data, found hyperparameters
    kfold = KFold(n_splits=config['outer_k_folds'], shuffle=True)
    samples = list(itertools.chain(*[dict_actors_index[actor] for actor in actors]))
    for fold, (train_samples, test_samples) in enumerate(kfold.split(samples)):
        fold_path = f'outer-fold-{fold + 1}/'
        path = config['path'] + fold_path
        if not os.path.exists(path):
            os.makedirs(path)
        best_runs = sigopt.get_experiment(experiments[fold]).get_best_runs()
        best_accuracy = 0
        hyperparameters = {}
        for run in best_runs:
            accuracy = run.values['Average validation accuracy'].value
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                hyperparameters = run.assignments
        train_and_test_full_model(train_samples, test_samples, hyperparameters, path, config, gpu,
                                  dict_actors_index, mode=config['mode'])
        print(hyperparameters['batchsize_train'] + '&' + hyperparameters['learning_rate'] + '&' +
              hyperparameters['dropout1'] + '&' + hyperparameters['dropout2'] + '&' + hyperparameters['dropout3'])


def train_mix_actors(config, dict_actors_index, actors, experiments, gpu):
    # Train / test with found hyperparameters
    kfold = KFold(n_splits=config['outer_k_folds'], shuffle=True)
    for fold, (train_actors, test_actors) in enumerate(kfold.split(actors)):
        fold_path = f'outer-fold-{fold + 1}/'
        path = config['path'] + fold_path
        train_actors = [actors[i] for i in train_actors]
        test_actors = [actors[i] for i in test_actors]
        if not os.path.exists(path):
            os.makedirs(path)
        best_runs = sigopt.get_experiment(experiments[fold]).get_best_runs()
        best_accuracy = 0
        hyperparameters = {}
        for run in best_runs:
            accuracy = run.values['Average validation accuracy'].value
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                hyperparameters = run.assignments
        train_and_test_full_model(train_actors, test_actors, hyperparameters, path, config, gpu,
                                  dict_actors_index, mode=config['mode'])

def main():
    config = dict()
    config['batchsize_test'] = 32
    config['epochs'] = 4000
    config['num_cl'] = 6
    config['seed'] = 0
    config['inner_k_folds'] = 3
    config['outer_k_folds'] = 6
    config['datafolder'] = '../data-processing/1D_processed_data/v2/song/'
    config['path'] = 'temp_data/1D_cnn/speaker_independent/v3/song/'
    config['model_name'] = '1D-si-song-v3' # '1D-si-speech-v3' '1D-si-speech-v2
    config['1D'] = True

    # SIGOPT_API_TOKEN must be set in terminal
    experiments = ['554496', '554870', '555066', '555076', '555136', '555199']  # speech sigopt experiments
    # experiments = ['554711', '554872', '554931', '554967', '555008', '555070']  # song

    gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    actors = np.unique(np.load(config['datafolder'] + 'actors.npy'))
    f = open(config['datafolder'] + 'actor_index_dict', 'rb')
    dict_actors_index = pickle.load(f)
    f.close()

    train_with_found_parameters(config, dict_actors_index, actors, experiments, gpu)

    train_mix_actors(config, dict_actors_index, actors, experiments, gpu)


if __name__ == '__main__':
    main()
