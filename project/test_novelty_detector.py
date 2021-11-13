if __name__ == '__main__':
    from comet_ml import Experiment
    import torch
    import torch.nn as nn
    from torchvision.transforms import ToTensor
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os

    from src.CNNs import Cnn14, Cnn6
    from src.classifier import SimpleClassifier, Wav2CLIPClassifier
    from src.wav2clip_classifier import w2c_classifier
    from src.UrbanSound import UrbanSoundDataset, UrbanSoundExposureGenerator
    from src.replay import ReplayExposureBlender, classwise_accuracy

    experiment = Experiment(
        api_key="pDxZIMJ2Bh2Abj9kefxI8jJvK",
        project_name="general",
        workspace="wjk0925",
    )
    hyperparams = {
        'sr': 16000,
        'exposure_size': 300, 
        'exposure_val_size': 50, 
        'initial_K': 4,
        'batch_size': 4,
        'num_epochs': 30,
        'num_epochs_ex': 10,
        'lr': 5e-6,
        'model': 'Wav2CLIP'
    }

    experiment.log_parameters(hyperparams)
    tags = 'UrbanSound', 'Wav2CLIP', 'Initial 4', 'Exposure 1+1', 'seen vs unseen', 'Novelty Detector'
    experiment.add_tags(tags)
    experiment.set_name(' '.join(tags))
    ############################################################################
    def normalize_tensor_wav(x, eps=1e-10, std=None):
        mean = x.mean(-1, keepdim=True)
        if std is None:
            std = x.std(-1, keepdim=True)
        return (x - mean) / (std + eps)

    exposure_generator = UrbanSoundExposureGenerator(
        '../../Datasets/UrbanSound8K', 
        range(1, 10), 
        sr=hyperparams['sr'], 
        exposure_size=hyperparams['exposure_size'], 
        exposure_val_size=hyperparams['exposure_val_size'], 
        initial_K=hyperparams['initial_K']
    )

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

    ############################################################################
    device = torch.device('cuda')

    if hyperparams['model'] == 'CNN14':
        model = Cnn14(sample_rate=16000, window_size=1024, hop_size=512, 
                    mel_bins=64, fmin=0, fmax=None, classes_num=10).to(device)

    elif hyperparams['model'] == 'Wav2CLIP':
        #model = Wav2CLIPClassifier()
        MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"
        scenario = 'finetune'
        ckpt = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=device, progress=True)
        model = w2c_classifier(ckpt=ckpt, scenario=scenario)
        
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
    #                              lr=hyperparams['lr'])

    train_loss_list = []
    val_acc_list = []
    val_all_acc_list = {}
    best_pretrain_acc = -9999
    since_best = 0
    for i in range(10):
        val_all_acc_list[i] = []
        
    loss_counter = 0
    loss_cycle = 50

    # Stage 1: Pre-train the model with inital classes
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    lmbda = lambda epoch: 2/3
    pretrain_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(pretrain_optimizer, lr_lambda=lmbda)

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

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()

            predictions = []
            truths = []
            model.eval()
            with torch.no_grad():
                for x, y in tqdm(initial_val_loader, desc='Validating'):
                    x, y = x.to(device), y.to(device)
                    yhat = torch.argmax(model(x), 1)
                    truths.extend(y.tolist())
                    predictions.extend(yhat.tolist())
    
            #acc = sum(np.array(truths)==np.array(predictions))/len(truths)
            acc, acc_classes = classwise_accuracy(np.array(predictions).flatten(),
                                                np.array(truths).flatten(),
                                                10,
                                                seen_classes
                                                )
            val_acc_list.append(acc)
            experiment.log_metric("Validation accuracy", acc, step=epoch)
            print()
            print('Accuracy: ', acc)
            
            for i in range(len(seen_classes)):
                val_all_acc_list[seen_classes[i]].append(acc_classes[i])
                experiment.log_metric(f"Validation accuracy class {seen_classes[i]}", acc_classes[i], step=epoch)
                print(f'Class {seen_classes[i]} accuracy: {acc_classes[i]}')

            if acc > best_pretrain_acc:
                best_pretrain_acc = acc

                torch.save(model.state_dict(), 
                    (os.path.join('saved_models/pretrain', 
                    hyperparams['model']
                    +'_'.join(map(str, seen_classes))
                    +'_'+str(best_pretrain_acc))+'.pt')
                )

            else:
                since_best += 1
                if since_best == 2:
                    since_best = 0
                    pretrain_scheduler.step()
                    print('Pretrain learning rate reduced to', pretrain_optimizer.param_groups[0]["lr"])

    
    # Stage 2: Feed an exposure 
    # Find a class we have already seen, and assign a new label to it
    # Check how is the accuracy
    for i, label in enumerate(exposure_label_list):
        if label in seen_classes:
            exposure_tr = exposure_tr_list[i]
            exposure_val = exposure_val_list[i]
            break


    for i, label in enumerate(exposure_label_list):
        exposure_tr = exposure_tr_list[i]
        exposure_val = exposure_val_list[i]

    experiment.log_parameters({'next_seen_class': label})
    new_tr = ReplayExposureBlender(initial_tr, exposure_tr, seen_classes, label, downsample=4)
    new_tr_loader = DataLoader(new_tr, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)
            
    model.load_state_dict(
        torch.load(os.path.join('saved_models', 
                hyperparams['model']
                +'_'.join(map(str, seen_classes))
                +'_'+str(last_acc)+'.pt')
        )
    )
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
                                lr=hyperparams['lr'])

    with experiment.train():   
        for epoch in tqdm(range(hyperparams['num_epochs_ex']), desc='Epoch'):
            model.train()
            for x, y in tqdm(new_tr_loader, desc='Training'):
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
    
            #acc = sum(np.array(truths)==np.array(predictions))/len(truths)
            acc, acc_classes = classwise_accuracy(np.array(predictions).flatten(),
                                                np.array(truths).flatten(),
                                                10,
                                                seen_classes
                                                )
            val_acc_list.append(acc)
            experiment.log_metric("Validation accuracy", acc, step=hyperparams['num_epochs']+epoch)
            print()
            print('Accuracy: ', acc)
            
            for i in range(len(seen_classes)):
                val_all_acc_list[seen_classes[i]].append(acc_classes[i])
                experiment.log_metric(f"Validation accuracy class {seen_classes[i]}", acc_classes[i], 
                                    step=hyperparams['num_epochs']+epoch)
                print(f'Class {seen_classes[i]} accuracy: {acc_classes[i]}')
            
    # Find a class we have never seen, and assign a new label to it
    # Check how is the accuracy
    for i, label in enumerate(exposure_label_list):
        if label not in seen_classes:
            exposure_tr = exposure_tr_list[i]
            exposure_val = exposure_val_list[i]
            break

    experiment.log_parameters({'next_unseen_class': label})
    new_tr = ReplayExposureBlender(initial_tr, exposure_tr, seen_classes, label, downsample=4)
    new_tr_loader = DataLoader(new_tr, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)
            
    model.load_state_dict(
        torch.load(os.path.join('saved_models', 
                hyperparams['model']
                +'_'.join(map(str, seen_classes))
                +'_'+str(last_acc)+'.pt')
        )
    )
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
                                lr=hyperparams['lr'])

    with experiment.train():                  
        for epoch in tqdm(range(hyperparams['num_epochs_ex']), desc='Epoch'):
            model.train()
            for x, y in tqdm(new_tr_loader, desc='Training'):
                x = normalize_tensor_wav(x)
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
                    x = normalize_tensor_wav(x)
                    x, y = x.to(device), y.to(device)
                    yhat = torch.argmax(model(x), 1)
                    truths.extend(y.tolist())
                    predictions.extend(yhat.tolist())
    
            #acc = sum(np.array(truths)==np.array(predictions))/len(truths)
            acc, acc_classes = classwise_accuracy(np.array(predictions).flatten(),
                                                np.array(truths).flatten(),
                                                10,
                                                seen_classes
                                                )
            val_acc_list.append(acc)
            experiment.log_metric("Validation accuracy", acc, 
                                step=hyperparams['num_epochs']+hyperparams['num_epochs_ex']+epoch)
            print()
            print('Accuracy: ', acc)
            
            for i in range(len(seen_classes)):
                val_all_acc_list[seen_classes[i]].append(acc_classes[i])
                experiment.log_metric(f"Validation accuracy class {seen_classes[i]}", acc_classes[i], 
                                    step=hyperparams['num_epochs']+hyperparams['num_epochs_ex']+epoch)
                print(f'Class {seen_classes[i]} accuracy: {acc_classes[i]}')
