import xgboost as xgb


class XGBOOST():
    def __init__(self, **modelOptions):
        self.model = xgb.XGBRegressor(**modelOptions)
