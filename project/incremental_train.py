from comet_ml import Experiment
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse

from src.wav2clip_classifier import w2c_classifier
from src.CNNs import Cnn14, Cnn6
from src.UrbanSound import UrbanSoundDataset, UrbanSoundExposureGenerator
from src.replay import ReplayExposureBlender, classwise_accuracy
from src.novelty_detect import make_novelty_detector

if __name__ == '__main__':
    
    # init hyperparameters, comet, ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="test_config")
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
    tags = 'UrbanSound', 'Wav2CLIP', 'Initial 4', 'Exposure 1+1', 'seen vs unseen', 'Novelty Detector'
    experiment.add_tags(tags)
    experiment.set_name(experiment_name)
    #####################################################################################


    # init initial train/val set, exposures
    exposure_generator = UrbanSoundExposureGenerator(
        '../../Datasets/UrbanSound8K', 
        hyperparams['train_val_folders'], 
        sr=hyperparams['sr'], 
        exposure_size=hyperparams['exposure_size'], 
        exposure_val_size=hyperparams['exposure_val_size'], 
        initial_K=hyperparams['initial_K']
    )

    prev_tr, prev_val, seen_classes = exposure_generator.get_initial_set()
    experiment.log_parameters({'inital_classes': seen_classes})

    exposure_tr_list = []
    exposure_val_list = []
    exposure_label_list = []

    for i in range(len(exposure_generator)):
        exposure_tr, exposure_val, label  = exposure_generator[i]  
        exposure_tr_list.append(exposure_tr)
        exposure_val_list.append(exposure_val)
        exposure_label_list.append(label)

    initial_tr_loader = DataLoader(prev_tr, batch_size=hyperparams['batch_size'], 
                                shuffle=True, num_workers=4)
    initial_val_loader = DataLoader(prev_val, batch_size=hyperparams['batch_size'], 
                                    shuffle=True, num_workers=4)
    #####################################################################################
    

    # init model, device, ...
    device = hyperparams['device']

    if hyperparams['model'] == 'Wav2CLIP':
        MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"
        scenario = 'finetune'
        ckpt = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=device, progress=True)
        model = w2c_classifier(ckpt=ckpt, scenario=scenario)
        
    model.to(device)
    #####################################################################################


    # init universal functions, arg, ...
    criterion = nn.CrossEntropyLoss()
    novelty_detector = make_novelty_detector(mode=hyperparams['novelty_detector'])
    lmbda = lambda epoch: hyperparams['reduce_lr_factor']
    #####################################################################################


    # train model on the initial classes
    train_loss_list = []
    val_acc_list = []
    val_all_acc_list = {}
    for i in range(10):
        val_all_acc_list[i] = []
    
    best_pretrain_acc = -9999
    best_pretrain_acc_classes = []
    since_reduce = 0
    since_best = 0
        
    loss_counter = 0
    loss_cycle = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    os.makedirs(os.path.join('ckpts', experiment_name, 'pretrain'))

    print('Train on Inital Classes', seen_classes)
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
                if loss_counter % loss_cycle == 0:
                    print()
                    print('Average running loss:', sum(train_loss_list[-loss_cycle:]) / loss_cycle)

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
                                                10,
                                                seen_classes
                                                )
            val_acc_list.append(acc)
            print()
            print('Accuracy: ', acc)
            
            for i in range(len(seen_classes)):
                val_all_acc_list[seen_classes[i]].append(classwise_acc[i])
                #experiment.log_metric(f"Validation accuracy class {seen_classes[i]}", classwise_acc[i], step=epoch)
                print(f'Class {seen_classes[i]} accuracy: {classwise_acc[i]}')

            if acc > best_pretrain_acc:
                best_pretrain_acc = acc
                best_pretrain_acc_classes = classwise_acc
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
                        break
                if since_reduce == hyperparams['reduce_lr_wait']:
                    since_reduce = 0
                    scheduler.step()
                    print('Learning rate reduced to', optimizer.param_groups[0]["lr"])

    print('Best Pretrain Acc Classes:', best_pretrain_acc_classes)
    experiment.log_parameters({'Best Pretrain Acc Classes': best_pretrain_acc_classes})
    #####################################################################################


    # train model on exposures incrementally












