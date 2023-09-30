from ..utils import setProjectpath
import numpy as np

# MODEL TYPES
RNN_TYPE = 'RNN'
XGBOOST_TYPE = "Xgboost"

# RNN cells types
GRU = 'GRU'
LSTM = 'LSTM'
SimpleRNN = 'SimpleRNN'

# criterion
MSE = 'MSE'
MAE = 'MAE'
SMOOTH = 'SMOOHL1'

# optimizer
ADAM = 'ADAM'
SGD = 'SGD'

RNN_PARAMS_LSTM = {
        "input_sequence_length": 21,
        "output_sequence_length": 7,
        "model_type": LSTM,
        "hidden_size": 32,
        "num_layers": 4,
        "output_size": 1,
        "batch_size": 32,
        "epochs": 700,
        "learning_rate": 0.1,
        'dataset_split': 0.7,
        'optimizer': 'SGD',
        'momentum': 0.9,
        'weight_decay': 1e-5,
        'lossFunction': SMOOTH,
        'verbose_mod': 100
}
DEFAULT_OPTIONS = {
    RNN_TYPE: RNN_PARAMS_LSTM,
    XGBOOST_TYPE: ...
}

# File path
project_dir = setProjectpath()
resultsDir = project_dir / 'prevision' / 'results'
RnnDir = resultsDir / RNN_TYPE
XGBoostDir = resultsDir / XGBOOST_TYPE

mse = lambda A, B, ax=None: (np.square(A - B)).mean(axis=ax)
mae = lambda A, B: np.abs(A - B).mean()
