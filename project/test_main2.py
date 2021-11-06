if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torchvision.transforms import ToTensor
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os

    from src.MLP import mlp

    #from src.UrbanSound import UrbanSoundDataset
    from src.UrbanSoundEmbedding import UrbanSoundDataset

    import time

    import wav2clip


    urban_tr = UrbanSoundDataset('../../Datasets/UrbanSound8K', [2,3,4,5,6,7,8,9,10], sr=16000, transform=ToTensor())
    urban_tr_loader = DataLoader(urban_tr, batch_size=32, shuffle=True,num_workers=4)

    urban_eval = UrbanSoundDataset('../../Datasets/UrbanSound8K', [1], sr=16000, transform=ToTensor())
    urban_eval_loader = DataLoader(urban_eval, batch_size=32, shuffle=False,num_workers=4)


    device = torch.device('cuda')
    model = mlp()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-5)

    loss_list = []
    loss_counter = 0

    max_acc = 0

    predictions = []
    truths = []

    encoder = wav2clip.get_model(device=device,pretrained=True)
    #embeddings = wav2clip.embed_audio(audio, model)

    for epoch in range(25):
        print(f'Epoch {epoch}')
        for x, labels in urban_tr_loader:
            #stime = time.time()
            x = np.array(x)
            labels = labels.to(device)
            embeddings = wav2clip.embed_audio(x, encoder)
            embeddings = (torch.from_numpy(embeddings)).to(device)
            predicts = model(embeddings)
            #print(predicts.shape)
            #print(predicts)
            #print(labels)
            loss = criterion(predicts, labels)

            loss_list.append(loss.item())
            loss_counter += 1
            if loss_counter % 50 == 0:
                print('Average running loss:', sum(loss_list[-10:]) / 10)
            #print(loss)
            #print(accuracy(labels, predicts))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #etime = time.time()

            #print(etime - stime)

        predictions = []
        truths = []

        with torch.no_grad():
            for x, labels in urban_eval_loader:
                x = np.array(x)
                labels = labels.to(device)

                embeddings = wav2clip.embed_audio(x, encoder)
                embeddings = (torch.from_numpy(embeddings)).to(device)
                predicts = model(embeddings)
                predicts = torch.argmax(predicts,1)

                truths.extend(labels.tolist())
                predictions.extend(predicts.tolist())

        acc = sum(np.array(truths)==np.array(predictions))/len(truths)

        print('Accuracy: ', acc)