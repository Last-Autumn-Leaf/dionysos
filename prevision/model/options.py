from .utils import XGBOOST_TYPE, RNN_TYPE
from ..pre_processing import ALL_FEATURES


class Options:

    def __init__(self,
                 model_type=RNN_TYPE, learning_rate=0.01, lossFunction='MSE',

                 input_size=25, output_size=1, input_sequence_length=10, output_sequence_length=5,
                 dataset_split=0.7,
                 targetFeatures=None, shuffle=False, hourly=False,

                 # rnn options
                 cell_type='LSTM', hidden_size=128, num_layers=2, dropout=0.2, batch_size=64, epochs=50,
                 optimizer='ADAM', momentum=0.9,
                 weight_decay=1e-5,

                 # xgboost options
                 n_estimators=500, max_depth=15, subsample=0.6, colsample_bytree=0.6, gamma=None, min_child_weight=None,
                 reg_alpha=None, reg_lambda=None, eval_metric='rmse',
                 verbose=True, verbose_mod=20,
                 ):

        self.model_type = model_type
        self.learning_rate = learning_rate
        self.lossFunction = lossFunction

        # data Options
        self.input_size = input_size
        self.output_size = output_size
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.dataset_split = dataset_split
        self.targetFeatures = targetFeatures
        self.shuffle = shuffle
        self.hourly = hourly

        # RNN options
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.output_size = output_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay

        # xgboost options
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.eval_metric = eval_metric

        # logs options :
        self.verbose = verbose
        self.verbose_mod = verbose_mod

        self.verif()

    def getModelOptions(self):
        if self.model_type == RNN_TYPE:
            return {
                'input_size': self.input_size,
                'cell_type': self.cell_type,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_size': self.output_size,
                'output_sequence_length': self.output_sequence_length,
                'input_sequence_length': self.input_sequence_length,
                'dropout': self.dropout
            }
        elif self.model_type == XGBOOST_TYPE:
            return {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'gamma': self.gamma,
                'min_child_weight': self.min_child_weight,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'eval_metric': self.eval_metric,
            }

    def getDataOptions(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'input_sequence_length': self.input_sequence_length,
            'output_sequence_length': self.output_sequence_length,
            'dataset_split': self.dataset_split,
            'targetFeatures': self.targetFeatures,
            'shuffle': self.shuffle,
            'hourly': self.hourly

        }

    def verif(self):
        if self.batch_size == 1:
            print("Current batch size is 1")

    def getXDimension(self):
        return self.input_sequence_length * self.input_size if self.model_type == XGBOOST_TYPE else (
            self.input_size, self.input_sequence_length)
