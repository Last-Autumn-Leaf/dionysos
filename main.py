from prevision import Options, Model, DataLoader, XGBOOST_TYPE

if __name__ == '__main__':
    options = Options()
    options.model_type = XGBOOST_TYPE
    options.verbose_mod = 50
    a = Model(options)
    b = DataLoader(options)
    a.train(b, plot=True)
