import time

def xilin_config():
    cfg = {
        'experiment_name' : 'xilin_incremental_tau_in_1_9_2_5',
        'dataset' : 'TAU',
        'dataset_path' : '/mnt/data/DCASE2019/Task1/TAU-urban-acoustic-scenes-2019-development',

        'comet' : {
            'api_key' : 'kOAHVqhBnkw2R6FQr6b0uOemJ',
            'project_name' : 'cs545',
            'workspace' : 'xi-j',
        },

        'hyperparams' : {
            'sr': 16000,
            'exposure_size': 300, 
            'exposure_val_size': 50, 
            'initial_K': 4,
            'train_val_folders' : [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'eval_folder' : [10],
            'test_size' : 240,
            'batch_size': 4,
            'num_epochs': 10,
            'num_epochs_initial' : 10, 
            'num_epochs_ex' : 10,
            'lr' : 3e-5,
            'reduce_lr_wait' : 2,
            'reduce_lr_factor' : 2/3,
            'early_stop' : False,
            'early_stop_wait' : 10,
            'model' : 'Wav2CLIP',
            'scenario' : 'finetune',
            'device' : 'cuda:3',
            'novelty_detector' : 'confusion',
            'imbalance_ratio' : 0.20,
            'threshold' : 0.8,
        }

    }

    return cfg

def incremental_train():
    experiment_time = int(time.time())

    cfg = {
        'experiment_name' : 'incremental_train_' + str(experiment_time),
        'dataset_path' : 'UrbanSound8K',

        'comet' : {
            'api_key' : 'kOAHVqhBnkw2R6FQr6b0uOemJ',
            'project_name' : 'cs545',
            'workspace' : 'xi-j',
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
            'device' : 'cuda:3',
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