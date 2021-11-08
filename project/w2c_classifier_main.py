if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torchvision.transforms import ToTensor
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    #from src.UrbanSound import UrbanSoundDataset
    from src.UrbanSound import UrbanSoundDataset

    from src.wav2clip_classifier import w2c_classifier


    urban_tr = UrbanSoundDataset('../../Datasets/UrbanSound8K', [1,2,3,4,5,6,7,8,9], sr=16000, transform=ToTensor())
    urban_tr_loader = DataLoader(urban_tr, batch_size=32, shuffle=True,num_workers=4)

    urban_eval = UrbanSoundDataset('../../Datasets/UrbanSound8K', [10], sr=16000, transform=ToTensor())
    urban_eval_loader = DataLoader(urban_eval, batch_size=32, shuffle=False,num_workers=4)


    device = torch.device('cuda')

    MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"

    scenario = 'finetune'
    ckpt = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=device, progress=True)
    model =w2c_classifier(ckpt=ckpt, scenario=scenario)
    model.to(device)

    mlp_lr = 3e-5
    encoder_lr = mlp_lr / 10


    criterion = nn.CrossEntropyLoss()

    if scenario == 'finetune':
        optimizer = torch.optim.Adam([{'params': model.mlp.parameters()},
                                      {'params': model.wav2clip_encoder.parameters(), 'lr': encoder_lr}], lr=mlp_lr)
    else:
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
                             lr=mlp_lr)


    loss_list = []
    loss_counter = 0

    max_acc = 0

    
    for epoch in range(10):
        print(f'Epoch {epoch}')
        model.train()
        for x, labels in urban_tr_loader:
            x, labels = x.to(device), labels.to(device)
            predicts = model(x)

            loss = criterion(predicts, labels)

            loss_list.append(loss.item())
            loss_counter += 1
            if loss_counter % 50 == 0:
                print('Average running loss:', sum(loss_list[-10:]) / 10)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predictions = []
        truths = []

        model.eval()
        with torch.no_grad():
            for x, labels in urban_eval_loader:
                x, labels = x.to(device), labels.to(device)

                predicts = model(x)
                predicts = torch.argmax(predicts,1)

                truths.extend(labels.tolist())
                predictions.extend(predicts.tolist())

        acc = sum(np.array(truths)==np.array(predictions))/len(truths)

        print('Accuracy: ', acc)