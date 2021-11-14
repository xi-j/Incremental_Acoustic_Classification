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
    from src.replay import ReplayExposureBlender, Replay, classwise_accuracy
    from src.novelty_detect import make_novelty_detector

    experiment = Experiment(
        api_key="kOAHVqhBnkw2R6FQr6b0uOemJ",
        project_name="cs545",
        workspace="xi-j",
    )
    hyperparams = {
        'sr': 16000,
        'exposure_size': 300, 
        'exposure_val_size': 50, 
        'replay_tr': 250,
        'replay_val': 50,
        'initial_K': 4,
        'batch_size': 4,
        'num_epochs': 2,
        'num_epochs_ex': 2,
        'lr': 5e-6,
        'model': 'Wav2CLIP'
        'imb_ratio': 0.5
    }
    hyperparams['exposure_tr_size'] = hyperparams['exposure_size'] - hyperparams['exposure_val_size']

    experiment_name = 'novelty_detector_replay_test'
    experiment.log_parameters(hyperparams)
    tags = 'UrbanSound', 'Wav2CLIP', 'Initial 4', 'Exposure 1+1', 'seen vs unseen', 'Novelty Detector'
    experiment.add_tags(tags)
    experiment.set_name(experiment_name)
    ############################################################################
    def normalize_tensor_wav(x, eps=1e-10, std=None):
        mean = x.mean(-1, keepdim=True)
        if std is None:
            std = x.std(-1, keepdim=True)
        return (x - mean) / (std + eps)

    exposure_generator = UrbanSoundExposureGenerator(
        #'../../Datasets/UrbanSound8K', 
        'UrbanSound8K',
        range(1, 10), 
        sr=hyperparams['sr'], 
        exposure_size=hyperparams['exposure_size'], 
        exposure_val_size=hyperparams['exposure_val_size'], 
        initial_K=hyperparams['initial_K']
    )

    initial_tr, initial_val, seen_classes = exposure_generator.get_initial_set()
    experiment.log_parameters({'inital_classes': seen_classes})
 
    for i in range(10):
        if i not in seen_classes:
            new_class_label = i
            break

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

    ############### DECLARE MODEL #############################################
    device = torch.device('cuda:1')

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

    train_loss_list = []
    val_acc_list = []
    val_all_acc_list = {}
    best_pretrain_acc = -9999
    best_pretrain_acc_classes = [0,0,0,0]
    since_best = 0
    
    for i in range(10):
        val_all_acc_list[i] = []
        
    loss_counter = 0
    loss_cycle = 80

    ############### TRAIN ON INITIAL CLASSES ##################################
    # Stage 1: Pre-train the model with initial classes
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    lmbda = lambda epoch: 2/3
    pretrain_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(pretrain_optimizer, lr_lambda=lmbda)

    os.makedirs(os.path.join('saved_models', experiment_name, 'pretrain'))

    print('PRETRAIN')
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
                best_pretrain_acc_classes = acc_classes
                torch.save(model.state_dict(), 
                    (os.path.join('saved_models',
                    experiment_name, 
                    'pretrain', 
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

    print('Best Pretrain Acc Classes:', best_pretrain_acc_classes)
    experiment.log_parameters({'Best Pretrain Acc Classes': best_pretrain_acc_classes})

    replay_tr = Replay(initial_tr, seen_classes, hyperparams['replay_tr'])
    replay_val = Replay(initial_val, seen_classes, hyperparams['replay_val'])

    ############### TRAIN ON EXPOSURES ########################################
    
    novelty_detector = make_novelty_detector()
    for i, label in enumerate(exposure_label_list):
        os.makedirs(os.path.join('saved_models', experiment_name, 'exposure' + str(i)))
        for seen_class in seen_classes:
            for sc_i in range(len(val_all_acc_list[seen_class])):
                experiment.log_metric(
                    f"Validation accuracy class {seen_class} ex {sc_i} label {label}", val_all_acc_list[seen_class][sc_i], step=sc_i
                )        
        
        print('')
        print('Exposure', i)
        print('Class', label)
        print('------------------------------------')

        best_exposure_acc = -9999
        best_exposure_acc_classes = [0,0,0,0]
        since_best = 0

        exposure_tr = exposure_tr_list[i]
        exposure_val = exposure_val_list[i]

        true_novelty = label not in seen_classes

        if true_novelty == True:
            true_max_drop_class = -1
            true_num_drop_class = 0
        else:
            true_max_drop_class = label
            true_num_drop_class = 1

        new_tr = ReplayExposureBlender(replay_tr, exposure_tr, 
                                       seen_classes, 
                                       resize=int(hyperparams['imb_ratio']*hyperparams['exposure_tr_size'])
                                      )
        new_tr_loader = DataLoader(new_tr, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)

        #new_val = ReplayExposureBlender(initial_val, exposure_val, seen_classes)
        #new_val_loader = DataLoader(new_val, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)
            
        model.load_state_dict(
            torch.load(os.path.join('saved_models', 
                    experiment_name,
                    'pretrain', 
                    hyperparams['model']
                    +'_'.join(map(str, seen_classes))
                    +'_'+str(best_pretrain_acc)+'.pt')
            )
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
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
                    if loss_counter % loss_cycle == 0:
                        print()
                        print('Average running loss:', sum(train_loss_list[-loss_cycle:]) / loss_cycle)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    
                model.eval()            
                predictions = []
                truths = []
                with torch.no_grad():
                    for x, y in tqdm(initial_val_loader, desc='Validating'):
                        x, y = x.to(device), y.to(device)
                        yhat = torch.argmax(model(x), 1)
                        truths.extend(y.tolist())
                        predictions.extend(yhat.tolist())

                acc, acc_classes = classwise_accuracy(np.array(predictions).flatten(),
                                                    np.array(truths).flatten(),
                                                    10,
                                                    seen_classes
                                                    )

                print('Total Accuracy: ', acc)      
                
                experiment.log_metric(f"Validation accuracy ex {i} label {label}", acc, step=epoch)   
                for sc_i in range(len(seen_classes)):
                    experiment.log_metric(f"Validation accuracy class {seen_classes[sc_i]} ex {i} label {label}", acc_classes[sc_i], 
                                        step=hyperparams['num_epochs']+epoch)
                    print(f'Class {seen_classes[sc_i]} accuracy: {acc_classes[sc_i]}')

                print(f'Class {new_class_label} accuracy: {acc_classes[-1]}')

                if epoch == hyperparams['num_epochs_ex'] - 1:#acc > best_exposure_acc:
                    best_exposure_acc = acc
                    best_exposure_acc_classes = acc_classes.copy()
                    torch.save(model.state_dict(), 
                        (os.path.join('saved_models',
                        experiment_name, 
                        'exposure' + str(i), 
                        hyperparams['model']
                        +'_'.join(map(str, seen_classes))
                        +'_'+str(best_exposure_acc))+'.pt')
                    )

                else:
                    since_best += 1
                    if since_best == 2:
                        since_best = 0
                        scheduler.step()
                        print('Exposure learning rate reduced to', optimizer.param_groups[0]["lr"])
                        

            print('Pretrain Acc:', best_pretrain_acc_classes)
            print('Exposure Acc:', best_exposure_acc_classes)

            for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                print('Threshold Used', threshold)
                novelty_detected, max_drop_class, num_drop_class = novelty_detector(
                    best_pretrain_acc_classes, best_exposure_acc_classes, seen_classes, threshold
                )

                print('Novelty Detected ' + ['Incorrect', 'Correct'][int(novelty_detected == true_novelty)])
                print('Seen Class Detected ' + ['Incorrect', 'Correct'][int(max_drop_class == true_max_drop_class)])
                print('Drop Class Num ' + ['Incorrect', 'Correct'][int(num_drop_class == true_num_drop_class)])

                print("Novelty:", true_novelty, "  Novelty Detected:", novelty_detected)
                print("Seen Class:", true_max_drop_class, "  Seen Class Detected:", max_drop_class)
                print("Drop Class Num :", true_num_drop_class, "  Drop Class Num Detected:", num_drop_class)
