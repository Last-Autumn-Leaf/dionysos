from prevision import Options, Model, DataLoader, XGBOOST_TYPE, getAllDataFromCSV, RNN_TYPE
import numpy as np

if __name__ == '__main__':
    # param_space = {
    #     'n_estimators': np.arange(50, 300, 50),  # Number of boosting rounds
    #     'max_depth': np.arange(3, 10),  # Maximum depth of the tree
    #     'learning_rate': np.arange(0.05, 0.31, 0.05),  # Learning rate
    #     'subsample': np.arange(0.7, 1.0, 0.1),  # Subsample ratio
    #     'colsample_bytree': np.arange(0.7, 1.0, 0.1),  # Subsample ratio of columns
    # }

    hourly = False
    recursif = True
    options = Options(model_type=XGBOOST_TYPE, input_sequence_length=14,
                      output_sequence_length=1 if recursif or hourly else 7, verbose_mod=100, input_size=27
                      , output_size=25 if hourly else 1, verbose=0, recursif=recursif, hourly=hourly)

    a = Model(options)
    b = DataLoader(options)
    a.train(b, 1, 1)
    res = a.deploy(b)
    print(res)
    # a.fineTuneXGBoostRay(b, None, 100)
    # a.featureImportance(b.getFeatureNames())
