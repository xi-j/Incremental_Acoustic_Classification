from comet_ml import Experiment
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse

from src.wav2clip_classifier import w2c_classifier
from src.CNNs import Cnn14, Cnn6
from src.UrbanSound import UrbanSoundDataset, UrbanSoundExposureGenerator
from src.TAU2019 import TAUDataset, TAUExposureGenerator
from src.replay import Replay, ReplayExposureBlender, classwise_accuracy
from src.novelty_detect import make_novelty_detector

if __name__ == '__main__':
    
    # init hyperparameters, comet, ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="xilin_config")
    args = vars(parser.parse_args())
    
    from configs import *
    cfg = globals()[args['cfg']]()

    hyperparams = cfg['hyperparams']
    experiment_name = cfg['experiment_name']
    experiment = Experiment(
        api_key = cfg['comet']['api_key'],
        project_name = cfg['comet']['project_name'],
        workspace=cfg['comet']['workspace'],
    )
    experiment.log_parameters(hyperparams)
    tags = cfg['dataset'], hyperparams['model'], 'Initial '+str(hyperparams['initial_K'])
    experiment.add_tags(tags)
    experiment.set_name(experiment_name)
    #####################################################################################

    # init initial train/val set, exposures
    assert cfg['dataset'] in ['UrbanSound8K', 'TAU']
    if cfg['dataset'] == 'UrbanSound8K':
        exposure_generator = UrbanSoundExposureGenerator(
            cfg['dataset_path'], 
            hyperparams['train_val_folders'], 
            sr=hyperparams['sr'], 
            exposure_size=hyperparams['exposure_size'], 
            exposure_val_size=hyperparams['exposure_val_size'], 
            initial_K=hyperparams['initial_K']
        )
    elif cfg['dataset'] == 'TAU':
        exposure_generator = TAUExposureGenerator(
            cfg['dataset_path'], 
            hyperparams['test_size'],
            sr=hyperparams['sr'], 
            exposure_size=hyperparams['exposure_size'], 
            exposure_val_size=hyperparams['exposure_val_size'], 
            initial_K=hyperparams['initial_K']
        )
        TAU_test = exposure_generator.get_test_set()
        
    initial_tr, initial_val, seen_classes = exposure_generator.get_initial_set()
    experiment.log_parameters({'inital_classes': seen_classes})

    exposure_tr_list = []
    exposure_val_list = []
    exposure_label_list = []

    for i in range(len(exposure_generator)):
        exposure_tr, exposure_val, label  = exposure_generator[i]  
        exposure_tr_list.append(exposure_tr)
        exposure_val_list.append(exposure_val)
        exposure_label_list.append(label)

    initial_tr_loader = DataLoader(initial_tr, batch_size=hyperparams['batch_size'], 
                                shuffle=True, num_workers=4)
    initial_val_loader = DataLoader(initial_val, batch_size=hyperparams['batch_size'], 
                                    shuffle=True, num_workers=4)
    #####################################################################################
    

    # init model, device, ...
    device = hyperparams['device']

    if hyperparams['model'] == 'Wav2CLIP':
        MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"
        scenario = hyperparams['scenario']
        ckpt = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=device, progress=True)
        model = w2c_classifier(ckpt=ckpt, scenario=scenario)
        
    model.to(device)
    #####################################################################################
    # init universal functions, arg, ...
    criterion = nn.CrossEntropyLoss()
    novelty_detector = make_novelty_detector(mode=hyperparams['novelty_detector'])
    lmbda = lambda epoch: hyperparams['reduce_lr_factor']
    exposure_train_size = hyperparams['exposure_size'] - hyperparams['exposure_val_size']
    loss_cycle = 100
    true_seen_classes = seen_classes.copy()
    initial_seen_classes = seen_classes.copy()
    
    exposure_order_x = []
    exposure_order_y = []
    for c in seen_classes:
        exposure_order_x.append(0)
        exposure_order_y.append(c)
    
    #####################################################################################


    # train model on the initial classes
    train_loss_list = []
    val_acc_list = []
    val_all_acc_list = {}
    for i in range(10):
        val_all_acc_list[i] = []
    
    best_pretrain_acc = -9999
    best_pretrain_classwise_acc = []
    since_reduce = 0
    since_best = 0
        
    loss_counter = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    os.makedirs(os.path.join('ckpts', experiment_name, 'pretrain'))

    print('Train on Initial Classes', seen_classes)
    print('------------------------------------')
    with experiment.train():
        for epoch in tqdm(range(hyperparams['num_epochs']), desc='Epoch'):
            model.train()
            for x, y in tqdm(initial_tr_loader, desc='Training'):
                x, y = x.to(device), y.to(device)
                yhat = model(x)
                loss = criterion(yhat, y)
                train_loss_list.append(loss.item())
                loss_counter += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predictions = []
            truths = []
            model.eval()
            with torch.no_grad():
                for x, y in tqdm(initial_val_loader, desc='Validating'):
                    x, y = x.to(device), y.to(device)
                    yhat = torch.argmax(model(x), 1)
                    truths.extend(y.tolist())
                    predictions.extend(yhat.tolist())
    
            acc, classwise_acc = classwise_accuracy(np.array(predictions).flatten(),
                                                np.array(truths).flatten(),
                                                11,
                                                seen_classes
                                                )
            val_acc_list.append(acc)
            print()
            print('Accuracy: ', acc)
            
            for i in range(len(seen_classes)):
                val_all_acc_list[seen_classes[i]].append(classwise_acc[i])
                experiment.log_metric(
                    f"initial val accuracy class {seen_classes[i]}", classwise_acc[i], step=epoch)

            if acc > best_pretrain_acc:
                since_best = 0
                since_reduce = 0
                best_pretrain_acc = acc
                best_pretrain_classwise_acc = classwise_acc
                torch.save(model.state_dict(), 
                    (os.path.join('ckpts',
                    experiment_name, 
                    'pretrain', 
                    hyperparams['model']
                    +'_'.join(map(str, seen_classes))
                    +'_'+str(best_pretrain_acc))+'.pt')
                )

            else:
                since_reduce += 1
                since_best += 1
                if hyperparams['early_stop']:
                    if since_best == hyperparams['early_stop_wait']:
                        print('Early Stop!')
                        break
                if since_reduce == hyperparams['reduce_lr_wait']:
                    since_reduce = 0
                    scheduler.step()
                    print('Learning rate reduced to', optimizer.param_groups[0]["lr"])

    print('Best Pretrain Acc Classes:', best_pretrain_classwise_acc)
    experiment.log_parameters({'Best Pretrain Acc Classes': best_pretrain_classwise_acc})

    best_prev_classwise_acc = best_pretrain_classwise_acc
    best_prev_acc = best_pretrain_acc
    #####################################################################################


    # train model on exposures incrementally
    prev_tr = Replay(initial_tr, seen_classes, exposure_train_size)
    prev_val = Replay(initial_val, seen_classes, hyperparams['exposure_val_size'])

    exposure_labels = []
    
    for i, label in enumerate(exposure_label_list):
        exposure_labels.append(label)

    print('Exposure labels:', exposure_labels)
    
    for i, label in enumerate(exposure_label_list):
        os.makedirs(os.path.join('ckpts', experiment_name, 'exposure' + str(i)))

        print('')
        print('Exposure', i)
        print('Class', label)
        exposure_order_x.append(i+1)
        exposure_order_y.append(label)
        print('------------------------------------')

        best_exposure_acc = -9999
        best_exposure_classwise_acc = []
        since_best = 0
        since_reduce = 0

        loss_counter = 0

        exposure_tr = exposure_tr_list[i]
        exposure_val = exposure_val_list[i]

        new_tr = ReplayExposureBlender(prev_tr, exposure_tr, seen_classes, 
                                       resize=int(hyperparams['imbalance_ratio']*exposure_train_size))
        new_tr_loader = DataLoader(new_tr, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)

        new_val = ReplayExposureBlender(prev_val, exposure_val, seen_classes)
        new_val_loader = DataLoader(new_val, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)

        true_novelty = label not in true_seen_classes

        if true_novelty == True:
            true_max_drop_class = -1
            true_num_drop_class = 0
        else:
            true_max_drop_class = seen_classes[true_seen_classes.index(label)]
            true_num_drop_class = 1
            
        if i == 0:
            model.load_state_dict(
                torch.load(os.path.join('ckpts', 
                        experiment_name,
                        'pretrain', 
                        hyperparams['model']
                        +'_'.join(map(str, seen_classes))
                        +'_'+str(best_pretrain_acc)+'.pt')
                )
            )

        else:
            model.load_state_dict(
                torch.load(os.path.join('ckpts', 
                        experiment_name,
                        'exposure' + str(i - 1), 
                        hyperparams['model']
                        +'_'.join(map(str, seen_classes))
                        +'_'+str(best_prev_acc)+'.pt')
                )
            )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

        with experiment.train():   
            for epoch in tqdm(range(hyperparams['num_epochs_ex']), desc='Epoch'):
                model.train()
                for x, y in tqdm(new_tr_loader, desc='Training'):
                    x, y = x.to(device), y.to(device)
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    train_loss_list.append(loss.item())
                    loss_counter += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                predictions = []
                truths = []
                model.eval()
                with torch.no_grad():
                    for x, y in tqdm(new_val_loader, desc='Validating'):
                        x, y = x.to(device), y.to(device)
                        yhat = torch.argmax(model(x), 1)
                        truths.extend(y.tolist())
                        predictions.extend(yhat.tolist())
        
                acc, classwise_acc = classwise_accuracy(np.array(predictions).flatten(),
                                                    np.array(truths).flatten(),
                                                    11,
                                                    seen_classes + [new_tr.pseudo_label]
                                                    )

                print()
                print('Accuracy: ', acc)
                
                for sc_i in range(len(seen_classes)):
                    print(f'Class {seen_classes[sc_i]} accuracy: {classwise_acc[sc_i]}') 
                    if seen_classes[sc_i] in initial_seen_classes:   
                        experiment.log_metric(
                            f"novel detection val accuracy per epoch class {seen_classes[sc_i]}", 
                            classwise_acc[sc_i], 
                            step=hyperparams['num_epochs']+i*hyperparams['num_epochs_ex']+epoch
                        )
                    else:
                        experiment.log_metric(
                            f"novel detection val accuracy per epoch pseudo class {seen_classes[sc_i]}", 
                            classwise_acc[sc_i], 
                            step=hyperparams['num_epochs']+i*hyperparams['num_epochs_ex']+epoch
                        )
                    
                    
                print(f'Class {new_tr.pseudo_label} accuracy: {classwise_acc[-1]}')

                if acc > best_exposure_acc:
                    since_best = 0
                    since_reduce = 0
                    best_exposure_acc = acc
                    best_exposure_classwise_acc = classwise_acc[:-1]

                else:
                    since_reduce += 1
                    since_best += 1
                    if hyperparams['early_stop']:
                        if since_best == hyperparams['early_stop_wait']:
                            print('Early Stop!')
                            break
                    if since_reduce == hyperparams['reduce_lr_wait']:
                        since_reduce = 0
                        scheduler.step()
                        print('Learning rate reduced to', optimizer.param_groups[0]["lr"])

            print('Prev Acc:', best_prev_classwise_acc)
            print('Exposure Acc:', best_exposure_classwise_acc)

            for sc_i in range(len(seen_classes)):    
                if seen_classes[sc_i] in initial_seen_classes:   
                    experiment.log_metric(
                        f"novel detection val accuracy per exposure class {seen_classes[sc_i]}", 
                        best_exposure_classwise_acc[sc_i], 
                        step=i+1
                    )
                else:
                    experiment.log_metric(
                        f"novel detection val accuracy per exposure pseudo class {seen_classes[sc_i]}", 
                        best_exposure_classwise_acc[sc_i], 
                        step=i+1
                    )
            
            novelty_detected, max_drop_class, num_drop_class = novelty_detector(
                best_prev_classwise_acc, best_exposure_classwise_acc, seen_classes, hyperparams['threshold'])

            print('Novelty Detected ' + ['Incorrect', 'Correct'][int(novelty_detected == true_novelty)])
            print('Seen Class Detected ' + ['Incorrect', 'Correct'][int(max_drop_class == true_max_drop_class)])
            print('Drop Class Num ' + ['Incorrect', 'Correct'][int(num_drop_class == true_num_drop_class)])

            print("Novelty:", true_novelty, "  Novelty Detected:", novelty_detected)
            print("Seen Class:", true_max_drop_class, "  Seen Class Detected:", max_drop_class)
            print("Drop Class Num :", true_num_drop_class, "  Drop Class Num Detected:", num_drop_class)

            if novelty_detected == True:
                print('New Class Retrain')
                inferred_label = new_tr.pseudo_label
                
            else:
                print('Old Class Retrain')
                inferred_label = max_drop_class

            new_tr = ReplayExposureBlender(prev_tr, exposure_tr, seen_classes, label=inferred_label)
            new_val = ReplayExposureBlender(prev_val, exposure_val, seen_classes, label=inferred_label)

            new_tr_loader = DataLoader(new_tr, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)
            new_val_loader = DataLoader(
                new_val, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)

            best_retrain_acc = -9999
            best_retrain_classwise_acc = []
            since_best = 0
            since_reduce = 0

            loss_counter = 0

            if i == 0:
                model.load_state_dict(
                    torch.load(os.path.join('ckpts', 
                            experiment_name,
                            'pretrain', 
                            hyperparams['model']
                            +'_'.join(map(str, seen_classes))
                            +'_'+str(best_pretrain_acc)+'.pt')
                    )
                )

            else:
                model.load_state_dict(
                    torch.load(os.path.join('ckpts', 
                            experiment_name,
                            'exposure' + str(i - 1), 
                            hyperparams['model']
                            +'_'.join(map(str, seen_classes))
                            +'_'+str(best_prev_acc)+'.pt')
                    )
                )

            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

            with experiment.train():   
                for epoch in tqdm(range(hyperparams['num_epochs']), desc='Epoch'):
                    model.train()
                    for x, y in tqdm(new_tr_loader, desc='Training'):
                        x, y = x.to(device), y.to(device)
                        yhat = model(x)
                        loss = criterion(yhat, y)
                        train_loss_list.append(loss.item())
                        loss_counter += 1

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    predictions = []
                    truths = []
                    model.eval()
                    with torch.no_grad():
                        for x, y in tqdm(new_val_loader, desc='Validating'):
                            x, y = x.to(device), y.to(device)
                            yhat = torch.argmax(model(x), 1)
                            truths.extend(y.tolist())
                            predictions.extend(yhat.tolist())
            
                    if inferred_label not in seen_classes:
                        seen_classes = seen_classes + [inferred_label]
                        true_seen_classes = true_seen_classes + [label]

                    acc, classwise_acc = classwise_accuracy(np.array(predictions).flatten(),
                                                        np.array(truths).flatten(),
                                                        11,
                                                        seen_classes
                                                        )

                    print()
                    print('Accuracy: ', acc)
                    for sc_i in range(len(seen_classes)):
                        print(f'Class {seen_classes[sc_i]} accuracy: {classwise_acc[sc_i]}') 
                        if seen_classes[sc_i] in initial_seen_classes:   
                            experiment.log_metric(
                                f"retrain val accuracy per epoch class {seen_classes[sc_i]}", 
                                classwise_acc[sc_i], 
                                step=hyperparams['num_epochs']+i*hyperparams['num_epochs']+epoch
                            )
                        else:
                            experiment.log_metric(
                                f"retrain val accuracy per epoch pseudo class {seen_classes[sc_i]}", 
                                classwise_acc[sc_i], 
                                step=hyperparams['num_epochs']+i*hyperparams['num_epochs']+epoch
                            )

                    if acc > best_retrain_acc:
                        since_best = 0
                        since_reduce = 0
                        best_retrain_acc = acc
                        best_retrain_classwise_acc = classwise_acc
                        torch.save(model.state_dict(), 
                            (os.path.join('ckpts',
                            experiment_name, 
                            'exposure' + str(i), 
                            hyperparams['model']
                            +'_'.join(map(str, seen_classes))
                            +'_'+str(best_retrain_acc))+'.pt')
                        )

                    else:
                        since_reduce += 1
                        since_best += 1
                        if hyperparams['early_stop']:
                            if since_best == hyperparams['early_stop_wait']:
                                print('Early Stop!')
                                break
                        if since_reduce == hyperparams['reduce_lr_wait']:
                            since_reduce = 0
                            scheduler.step()
                            print('Learning rate reduced to', optimizer.param_groups[0]["lr"])

                print('Prev Acc:', best_prev_classwise_acc)
                print('Retrain Acc:', best_retrain_classwise_acc)
                
                best_prev_classwise_acc = best_retrain_classwise_acc
                best_prev_acc = best_retrain_acc
                
                for sc_i in range(len(seen_classes)):
                    if seen_classes[sc_i] in initial_seen_classes: 
                        experiment.log_metric(
                            f"retrain val accuracy per exposure class {seen_classes[sc_i]}", 
                            best_retrain_classwise_acc[sc_i], 
                            step=i+1
                        )
                    else:
                        experiment.log_metric(
                            f"retrain val accuracy per exposure pseudo class {seen_classes[sc_i]}", 
                            best_retrain_classwise_acc[sc_i],
                            step=i+1
                        )   

                prev_tr.update(exposure_tr, inferred_label)
                prev_val.update(exposure_val, inferred_label)
                

    print('True labels:', true_seen_classes)
    experiment.log_parameters({'True Seen Classes': true_seen_classes})
    
    f = plt.figure(figsize=(12,12))
    ax = f.add_subplot(1,1,1)
    ax.scatter(exposure_order_x, exposure_order_y)
    ax.set_ylim(0, 10)
    ax.set_title('Exposure Labels')
    ax.set_xlabel('Exposures')
    plt.show(block=False)
    experiment.log_figure(figure_name='Exposure Arrival Ordering')

















