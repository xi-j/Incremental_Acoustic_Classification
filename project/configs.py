import time

def test_config():
    cfg = {
        'experiment_name' : 'test_config',
        'dataset_path' : '../../Datasets/UrbanSound8K',

        'comet' : {
            'api_key' : 'pDxZIMJ2Bh2Abj9kefxI8jJvK',
            'project_name' : 'cs545',
            'workspace' : 'wjk0925',
        },

        'hyperparams' : {
            'sr': 16000,
            'exposure_size': 300, 
            'exposure_val_size': 50, 
            'initial_K': 4,
            'train_val_folders' : [1,2,3,4,5,6,7,8,9,10],
            'eval_folder' : [11],
            'batch_size': 4,
            'num_epochs': 3,
            'num_epochs_ex' : 3,
            'lr' : 3e-5,
            'reduce_lr_wait' : 2,
            'reduce_lr_factor' : 2/3,
            'early_stop' : False,
            'early_stop_wait' : 10,
            'model' : 'Wav2CLIP',
            'device' : 'cuda',
            'novelty_detector' : 'confusion',
            'imbalance_ratio' : 0.25,
            'threshold' : 0.5,
        }

    }

    return cfg

def incremental_train():
    experiment_time = int(time.time())

    cfg = {
        'experiment_name' : 'incremental_train_' + str(experiment_time),
        'dataset_path' : '../../Datasets/UrbanSound8K',

        'comet' : {
            'api_key' : 'pDxZIMJ2Bh2Abj9kefxI8jJvK',
            'project_name' : 'cs545',
            'workspace' : 'wjk0925',
        },

        'hyperparams' : {
            'sr': 16000,
            'exposure_size': 300, 
            'exposure_val_size': 50, 
            'initial_K': 4,
            'train_val_folders' : [1,2,3,4,5,6,7,8,9],
            'eval_folder' : [10],
            'batch_size': 8,
            'num_epochs': 10,
            'num_epochs_ex' : 6,
            'lr' : 3e-5,
            'reduce_lr_wait' : 2,
            'reduce_lr_factor' : 2/3,
            'early_stop' : False,
            'early_stop_wait' : 10,
            'model' : 'Wav2CLIP',
            'device' : 'cuda',
            'novelty_detector' : 'confusion',
            'imbalance_ratio' : 0.25,
            'threshold' : 0.5,
        }

    }

    return cfg

def incremental_train_2():
    experiment_time = int(time.time())

    cfg = {
        'experiment_name' : 'incremental_train_' + str(experiment_time),
        'dataset_path' : '../../Datasets/UrbanSound8K',

        'comet' : {
            'api_key' : 'pDxZIMJ2Bh2Abj9kefxI8jJvK',
            'project_name' : 'cs545',
            'workspace' : 'wjk0925',
        },

        'hyperparams' : {
            'sr': 16000,
            'exposure_size': 150, 
            'exposure_val_size': 30, 
            'initial_K': 4,
            'train_val_folders' : [1,2,3,4,5,6,7,8,9],
            'eval_folder' : [10],
            'batch_size': 6,
            'num_epochs': 10,
            'num_epochs_ex' : 6,
            'lr' : 3e-5,
            'reduce_lr_wait' : 2,
            'reduce_lr_factor' : 2/3,
            'early_stop' : False,
            'early_stop_wait' : 10,
            'model' : 'Wav2CLIP',
            'device' : 'cuda',
            'novelty_detector' : 'confusion',
            'imbalance_ratio' : 0.2,
            'threshold' : 0.6,
        }

    }

    return cfg