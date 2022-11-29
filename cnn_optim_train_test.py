import os
import pickle
import itertools
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import RandomSampler
import sigopt
from sklearn.model_selection import train_test_split

from cnn import train_and_evaluate, CNN, train_epoch, evaluate_epoch, CustomDataset
from onedim_cnn import CNN_ONEDIM, CustomDatasetOneD

os.environ['SIGOPT_API_TOKEN'] = 'EKGUACSQRFUEDTERDTXDHTRMTFWRYQJCYNPPPYAGNEMLUBOU'


def create_dataset(data_dir, num_cl, ids, onedim=False):
    x = torch.load(data_dir + 'x.pt')
    y = np.load(data_dir + 'y.npy').astype(int)

    x = x[ids]
    y = y[ids]
    if onedim:
        dataset = CustomDatasetOneD(x, y, num_cl)
    else:
        dataset = CustomDataset(x, y, num_cl)
    return dataset


def create_dataloader(datafolder, ids, batchsize, num_cl, onedim=False):
    dataset = create_dataset(datafolder, num_cl, ids, onedim)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=0, pin_memory=True, sampler=sampler)
    return dataloader


def initiate_cnn(train_loader, lr, device, dropout1, dropout2, dropout3, onedim=False):
    if onedim:
        model = CNN_ONEDIM().float()
    else:
        model = CNN(dropout1, dropout2, dropout3).float()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, eps=1e-08)
    criterion = nn.BCELoss()
    train_epoch(model, train_loader, optimizer, criterion, device=device)
    return model, optimizer, criterion


def train_and_test_full_model(train_data, test_data, hyperparameters, path, config, gpu, dict_actors_index,
                              mode, output=None):

    train_data, val_data = train_test_split(train_data, test_size=0.2)
    lr = hyperparameters['learning_rate']
    batchsize_train = hyperparameters['batchsize_train']
    dropout1 = hyperparameters['dropout1']
    dropout2 = hyperparameters['dropout2']
    dropout3 = hyperparameters['dropout3']

    if mode == 'si':
        train_data = list(itertools.chain(*[dict_actors_index[actor] for actor in train_data]))
        val_data = list(itertools.chain(*[dict_actors_index[actor] for actor in val_data]))
        test_data = list(itertools.chain(*[dict_actors_index[actor] for actor in test_data]))

    train_loader = create_dataloader(config['datafolder'], train_data, batchsize=batchsize_train,
                                     num_cl=config['num_cl'], onedim=config['1D'])
    val_loader = create_dataloader(config['datafolder'], val_data, batchsize=config['batchsize_test'],
                                   num_cl=config['num_cl'], onedim=config['1D'])
    test_loader = create_dataloader(config['datafolder'], test_data, batchsize=config['batchsize_test'],
                                    num_cl=config['num_cl'], onedim=config['1D'])

    print('\n\n Creating and training full model.')
    model, optimizer, criterion = initiate_cnn(train_loader, lr, gpu, dropout1, dropout2, dropout3, onedim=config['1D'])

    if output:
        output.write('\n\n Creating and training full model:\n')
    history, best_weights = train_and_evaluate(config['epochs'], model, train_loader, val_loader,
                                               optimizer, criterion, device=gpu, patience=200,
                                               num_cl=config['num_cl'], output=output, path=path)

    print('Training completed. Testing...')
    if output:
        output.write('Training completed. Testing...')
    # test med test_actors og rapporter resultatet
    avgprecs, test_loss, test_correct, concat_labels, concat_pred = evaluate_epoch(model, test_loader,
                                                                                   criterion,
                                                                                   config['num_cl'], gpu)
    history['test_loss'] = test_loss
    history['test_acc'] = test_correct / len(test_loader) * 100
    print('Accuracy on test set:', history['test_acc'])
    if output:
        output.write('Accuracy on test set:' + str(history['test_acc']))
    torch.save(history, path + 'full_training_history_dict')
    torch.save(best_weights, path + 'model_state_dict')
    np.save(path + 'concat_labels', concat_labels)
    np.save(path + 'concat_pred', concat_pred)


def si_run(config, actors, device, run, dict_actors_index, path, output=None):
    """
    Method for tuning hyperparameters using k-fold cross-validation.
    A set of hyperparameters will be chosen and the actors received are divided into k folds and each fold being used
    for evaluation once.
    When the cross-validation is completed and the scores reported, the process is repeated with a new set of
    hyperparameters.
    """
    torch.manual_seed(config['seed'])
    kfold = KFold(n_splits=config['inner_k_folds'], shuffle=True)
    foldperf = {}

    for fold, (train_actors, val_actors) in enumerate(kfold.split(actors)):

        if output:
            output.write(f'FOLD {fold + 1}\n')
            output.write('--------------------------------\n')
        else:
            print(f'FOLD {fold + 1}')
            print('--------------------------------')

        """ using the dictionary dict_actors_index {actor: [indices]} to create lists over the indices in the x-tensor
            belonging to the actors in the different sets"""
        train_actors = [actors[i] for i in train_actors]
        val_actors = [actors[i] for i in val_actors]
        train_ids = list(itertools.chain(*[dict_actors_index[actor] for actor in train_actors]))
        val_ids = list(itertools.chain(*[dict_actors_index[actor] for actor in val_actors]))

        train_loader = create_dataloader(config['datafolder'], train_ids, batchsize=run.params.batchsize_train,
                                         num_cl=config['num_cl'], onedim=config['1D'])
        val_loader = create_dataloader(config['datafolder'], val_ids, batchsize=config['batchsize_test'],
                                       num_cl=config['num_cl'], onedim=config['1D'])

        # creating NN, optimizer and loss function
        model, optimizer, criterion = initiate_cnn(train_loader, run.params.learning_rate, device,
                                                   run.params.dropout1, run.params.dropout2, run.params.dropout3,
                                                   onedim=config['1D'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader),
                                                        epochs=config['epochs'])

        history, best_weights = train_and_evaluate(config['epochs'], model, train_loader, val_loader,
                                                   optimizer, criterion, device=device, patience=200,
                                                   num_cl=config['num_cl'], output=output,
                                                   scheduler=scheduler, path=path)

        foldperf['fold{}'.format(fold + 1)] = history

    # Reporting of scores per fold
    vl_f, tl_f, va_f, ta_f = [], [], [], []
    # avgp_f = []
    for f in range(1, config['inner_k_folds'] + 1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        vl_f.append(np.mean(foldperf['fold{}'.format(f)]['val_loss']))
        ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        va_f.append(np.mean(foldperf['fold{}'.format(f)]['val_acc']))
        if output:
            output.write('--------------------------------\n')
            output.write(f'Results fold {f}\n')
            output.write(f'Train loss: {tl_f[-1]}\n')
            output.write(f'Validation loss: {vl_f[-1]}\n')
            output.write(f'Train accuracy: {ta_f[-1]}\n')
            output.write(f'Validation accuracy: {va_f[-1]}\n')
            output.write('--------------------------------\n')
        else:
            print('--------------------------------')
            print(f'Results fold {f}')
            print(f'Train loss: {tl_f[-1]}')
            print(f'Validation loss: {vl_f[-1]}')
            print(f'Train accuracy: {ta_f[-1]}')
            print(f'Validation accuracy: {va_f[-1]}')
            print('--------------------------------')

    if output:
        output.write('Performance of {} fold cross validation'.format(config['inner_k_folds']))
        output.write('Average Training Loss: {:.3f} \t Average Val Loss: {:.3f} \t Average Training Acc: {:.2f} \t '
                     'Average Val Acc: {:.2f}'.format(np.mean(tl_f), np.mean(vl_f), np.mean(ta_f), np.mean(va_f)))
    else:
        print('Performance of {} fold cross validation'.format(config['inner_k_folds']))
        print('Average Training Loss: {:.3f} \t Average Val Loss: {:.3f} \t Average Training Acc: {:.2f} \t '
              'Average Val Acc: {:.2f}'.format(np.mean(tl_f), np.mean(vl_f), np.mean(ta_f), np.mean(va_f)))

    torch.save(foldperf, path + 'foldperf')

    # log results to hyperparameter-visualization
    run.log_metric('Average validation accuracy', np.mean(va_f))
    run.log_metric('Average validation loss', np.mean(vl_f))
    run.log_metric('Accuracy fold 1', va_f[0])
    run.log_metric('Accuracy fold 2', va_f[1])
    run.log_metric('Accuracy fold 3', va_f[2])


def sd_run(config, data, device, run, path, output):
    torch.manual_seed(config['seed'])
    kfold = KFold(n_splits=config['inner_k_folds'], shuffle=True)
    foldperf = {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(data)):

        train_loader = create_dataloader(config['datafolder'], train_ids, batchsize=run.params.batchsize_train,
                                         num_cl=config['num_cl'], onedim=config['1D'])
        val_loader = create_dataloader(config['datafolder'], val_ids, batchsize=config['batchsize_test'],
                                        num_cl=config['num_cl'], onedim=config['1D'])

        # creating NN, optimizer and loss function
        model, optimizer, criterion = initiate_cnn(train_loader, run.params.learning_rate, device,
                                                   run.params.dropout1, run.params.dropout2, run.params.dropout3,
                                                   onedim=config['1D'])

        history, best_weights = train_and_evaluate(config['epochs'], model, train_loader, val_loader,
                                                   optimizer, criterion, device=device, patience=20,
                                                   num_cl=config['num_cl'], output=output, path=path)

        foldperf['fold{}'.format(fold + 1)] = history

    vl_f, tl_f, va_f, ta_f = [], [], [], []
    # avgp_f = []
    for f in range(1, config['inner_k_folds'] + 1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        vl_f.append(np.mean(foldperf['fold{}'.format(f)]['val_loss']))
        ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        va_f.append(np.mean(foldperf['fold{}'.format(f)]['val_acc']))
        # avgp_f.append(np.mean(foldperf['fold{}'.format(f)]['test_perfs']))
        if output:
            output.write('--------------------------------\n')
            output.write(f'Results fold {f}\n')
            output.write(f'Train loss: {tl_f[-1]}\n')
            output.write(f'Validation loss: {vl_f[-1]}\n')
            output.write(f'Train accuracy: {ta_f[-1]}\n')
            output.write(f'Validation accuracy: {va_f[-1]}\n')
            # output.write(f'Average precision: {avgp_f[-1]}')
            output.write('--------------------------------\n')
        else:
            print('--------------------------------')
            print(f'Results fold {f}')
            print(f'Train loss: {tl_f[-1]}')
            print(f'Validation loss: {vl_f[-1]}')
            print(f'Train accuracy: {ta_f[-1]}')
            print(f'Validation accuracy: {va_f[-1]}')
            # print(f'Average precision: {avgp_f[-1]}')
            print('--------------------------------')

    if output:
        output.write('Performance of {} fold cross validation'.format(config['inner_k_folds']))
        output.write('Average Training Loss: {:.3f} \t Average Val Loss: {:.3f} \t Average Training Acc: {:.2f} \t '
              'Average Val Acc: {:.2f}'.format(np.mean(tl_f), np.mean(vl_f), np.mean(ta_f), np.mean(va_f)))
    else:
        print('Performance of {} fold cross validation'.format(config['inner_k_folds']))
        print('Average Training Loss: {:.3f} \t Average Val Loss: {:.3f} \t Average Training Acc: {:.2f} \t '
              'Average Val Acc: {:.2f}'.format(np.mean(tl_f), np.mean(vl_f), np.mean(ta_f), np.mean(va_f)))

    torch.save(foldperf, path + 'foldperf')

    # metrics = {'Average validation accuracy': np.mean(testa_f), 'Average validation loss': np.mean(testl_f),
    # 'Accuracy fold 1': testa_f[0], 'Accuracy fold 2': testa_f[1], 'Accuracy fold 3': testa_f[2]}

    # log results to hyperparameter-visualization
    run.log_metric('Average validation accuracy', np.mean(va_f))
    run.log_metric('Average validation loss', np.mean(vl_f))
    run.log_metric('Accuracy fold 1', va_f[0])
    run.log_metric('Accuracy fold 2', va_f[1])
    run.log_metric('Accuracy fold 3', va_f[2])


def create_parallell_experiments(fold, train_data, test_data, config, gpu, fold_path, dict_actors_index, model_name=None):
    if model_name:
        name = model_name + '-outer-' + str(fold + 1)
    else:
        name = config['model_name'] + '-outer-' + str(fold + 1)
    output = open(fold_path + 'output' + '.log', 'w')
    #output = None
    experiment = sigopt.create_experiment(name=name,
                                          type='offline',
                                          parameters=[dict(name='batchsize_train', type='int', grid=[8, 16]),
                                                      dict(name='learning_rate', type='double',
                                                           bounds=dict(min=0.0005, max=0.13)),
                                                      dict(name='dropout1', type='double', bounds=dict(min=0, max=0.5)),
                                                      dict(name='dropout2', type='double', bounds=dict(min=0, max=0.5)),
                                                      dict(name='dropout3', type='double',
                                                           bounds=dict(min=0, max=0.5))],
                                          metrics=[dict(name='Average validation accuracy', strategy='optimize',
                                                        objective='maximize'),
                                                   dict(name='Average validation loss', strategy='optimize',
                                                        objective='minimize'),
                                                   dict(name='Accuracy fold 1', strategy='store',
                                                        objective='maximize'),
                                                   dict(name='Accuracy fold 2', strategy='store',
                                                        objective='maximize'),
                                                   dict(name='Accuracy fold 3', strategy='store',
                                                        objective='maximize')],
                                          parallel_bandwidth=1,
                                          budget=100)

    for run in experiment.loop():
        with run:
            if config['mode'] == 'si':
                si_run(config, train_data, gpu, run, dict_actors_index, fold_path, output)
            else:
                sd_run(config, train_data, gpu, run, fold_path, output)

    best_runs = experiment.get_best_runs()
    best_accuracy = 0
    hyperparameters = {}
    for run in best_runs:
        accuracy = run.values['Average validation accuracy'].value
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            hyperparameters = run.assignments

    train_and_test_full_model(train_data, test_data, hyperparameters, fold_path, config, gpu,
                              dict_actors_index, mode=config['mode'], output=output)
    output.close()


def main():
    config = dict()
    config['batchsize_test'] = 32
    config['epochs'] = 4000
    config['num_cl'] = 6
    config['seed'] = 0
    config['inner_k_folds'] = 3
    config['outer_k_folds'] = 6 # 4 for sd
    config['datafolder'] = '../data-processing/1D_processed_data/v2/song/'
    config['path'] = 'temp_data/1D_cnn/speaker_independent/v3/song/'
    config['model_name'] = '1D-si-song-v3'
    config['mode'] = 'si'
    config['1D'] = False

    actors = np.unique(np.load(config['datafolder'] + 'actors.npy'))
    gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    f = open(config['datafolder'] + 'actor_index_dict', 'rb')
    dict_actors_index = pickle.load(f)
    f.close()
    
    if config['mode'] == 'si':
        kfold = KFold(n_splits=config['outer_k_folds'], shuffle=True)
        for fold, (train_actors, test_actors) in enumerate(kfold.split(actors)):
            fold_path = f'outer-fold-{fold + 1}/'
            path = config['path'] + fold_path
            train_actors = [actors[i] for i in train_actors]
            test_actors = [actors[i] for i in test_actors]
            if not os.path.exists(path):
                os.makedirs(path)
            print('\n Beginning outer cross-validation fold', fold + 1, '.\n')
            create_parallell_experiments(fold, train_actors, test_actors, config, gpu, path, dict_actors_index)

    elif config['mode'] == 'sd':
        for actor in actors:
            model_name = 'actor-' + str(actor) + '-' + config['model_name']
            samples = dict_actors_index[actor]
            kfold = KFold(n_splits=config['outer_k_folds'], shuffle=True)
            for fold, (train_samples, test_samples) in enumerate(kfold.split(samples)):
                fold_path = f'outer-fold-{fold + 1}/'
                path = config['path'] + f'actor-{actor}/' + fold_path
                if not os.path.exists(path):
                    os.makedirs(path)
                print('\n Beginning outer cross-validation fold', fold + 1, '.\n')
                create_parallell_experiments(fold, train_samples, test_samples, config, gpu, path, dict_actors_index,
                                             model_name=model_name)

    print('Finished')


if __name__ == '__main__':
    main()
