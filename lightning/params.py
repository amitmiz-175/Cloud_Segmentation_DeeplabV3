import torch
from ray import tune

USE_RAY = True

NUM_AVAILABLE_GPU = torch.cuda.device_count()
RESULTS_PATH = 'lightning_results/'

CONSTS = {
    'max_epochs': 4,
    'num_workers': 6,
    # 'train_csv_path': train_csv_path,
    # 'test_csv_path': test_csv_path,
    # 'stats_csv_path': stats_csv_path,
    'num_gpus': NUM_AVAILABLE_GPU,
}

RAY_HYPER_PARAMS = {
    "lr": tune.choice([0.5 * 1e-3]),  # [1e-3, 0.5 * 1e-3, 1e-4]),
    "bsize": tune.choice([32]),  # 48 , 64]),  # [16, 8]),
    "ltype": tune.choice(['ce']),  # , 'MSELoss']),  # ['MARELoss']
    # 'dnorm': tune.grid_search([True, False])  # data_norm, , False
}

NON_RAY_HYPER_PARAMS = {
    "lr": 1 * 1e-3,
    "bsize": 32,
    "ltype": 'ce',  # loss_type
    # 'dnorm': True,  # data_norm
}
