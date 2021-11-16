from src.UrbanSound import UrbanSoundDataset
from src.wav2clip_classifier import w2c_classifier

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torchvision.transforms import ToTensor
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from sklearn.metrics import confusion_matrix

    urban_eval = UrbanSoundDataset('../../Datasets/UrbanSound8K', [10], sr=16000, transform=ToTensor())
    urban_eval_loader = DataLoader(urban_eval, batch_size=8, shuffle=False,num_workers=4)


    device = torch.device('cuda')

    MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"

    scenario = 'finetune'
    ckpt = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=device, progress=True)
    model =w2c_classifier(ckpt=ckpt, scenario=scenario)
    model.to(device)

    model.load_state_dict(
        torch.load('ckpts/incremental_train1637010564/exposure10/Wav2CLIP2_4_1_5_0_3_6_7_8_9_0.958.pt')
        )


    truths = []
    predictions = []
        
    model.eval()
    with torch.no_grad():
        for x, labels in urban_eval_loader:
            x, labels = x.to(device), labels.to(device)

            predicts = model(x)
            predicts = torch.argmax(predicts,1)

            truths.extend(labels.tolist())
            predictions.extend(predicts.tolist())

    truths = np.array(truths)
    predictions = np.array(predictions)

    # 3 - 0,8 - 3, 9 - 6, 7 - 7, 0 - 8, 6 - 9
    mapping = {0:3, 3:8, 6:9, 7:7, 8:0, 9:6}
    for i in range(len(predictions)):
        if predictions[i] in [2,4,1,5]:
            continue
        else:
            predictions[i] = mapping[predictions[i]]

    acc = sum(truths==predictions)/len(truths)

    print('Overrall Accuracy: ', acc)

    for l in range(10):
        label_truths = truths[truths == l]
        label_predictions = predictions[truths == l]
        label_acc = sum(label_truths==label_predictions)/len(label_truths)
        print('Class {} Accuracy:'.format(l), label_acc)

    matrix = np.array(confusion_matrix(truths, predictions, labels=range(10)))

    print(matrix)
    print(np.sum(matrix, 1))

    # 3 - 0,8 - 3, 9 - 6, 7 - 7, 0 - 8, 6 - 9