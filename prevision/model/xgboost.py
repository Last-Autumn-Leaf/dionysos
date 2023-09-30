import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from .utils import mse, mae, np
import ray
from ray import train, tune
from ray.tune import grid_search


class Xgboost:
    def __init__(self, **modelOptions):
        self.model = xgb.XGBRegressor(**modelOptions)

    # O(k*n_iter*cv)
    def fineTune(self, X_train, Y_train, X_test, Y_test, param_distributions, n_iter=100, cv=3, verbose=False):
        random_search = RandomizedSearchCV(self.model, param_distributions,
                                           n_iter=n_iter, scoring='neg_root_mean_squared_error', cv=cv,
                                           random_state=42, verbose=verbose, n_jobs=-1)
        random_search.fit(X_train, Y_train)
        best_model = random_search.best_estimator_
        val_preds = best_model.predict(X_test)
        val_rmse = np.sqrt(mse(Y_test, val_preds))
        val_mae = mae(Y_test, val_preds)
        print('Validation RMSE:', val_rmse)
        print('Validation MAE:', val_mae)
        print('best params\n', random_search.best_params_)
        if self.model.booster is not None:
            old_val_preds = self.model.predict(X_test)
            old_val_rmse = np.sqrt(mse(Y_test, old_val_preds))
            if old_val_rmse > val_rmse:
                print(
                    f"Better score reached during fine tunning, "
                    f"replacing old model ({old_val_rmse}) with new model ({val_rmse})")
                self.model = best_model
        else:
            self.model = best_model
        best_model.fit(X_train, Y_train,
                       eval_set=[(X_train, Y_train), (X_test, Y_test)],
                       verbose=0)
        eval_results = best_model.evals_result()
        res = best_model.predict(X_test)
        return eval_results, res

    def rayTune(self, X_train, Y_train, X_test, Y_test, search_space=None, n_iter=100):
        metric = "root_mean_squared_error"

        # TODO : It is possible to use two metrics liek rmse and mae
        def train_xgboost(config):
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": config["max_depth"],
                "subsample": config["subsample"],
                "learning_rate": config["learning_rate"],
                "colsample_bytree": config["colsample_bytree"],
                "verbosity": 0,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            rmse_val = float(np.sqrt(mse(Y_test, y_pred)))
            train.report({"root_mean_squared_error": rmse_val, "model": model})

        if not search_space:
            search_space = {
                "max_depth": tune.randint(3, 10),
                'learning_rate': tune.uniform(0.05, 0.3),
                "subsample": tune.uniform(0.7, 1.0),
                "colsample_bytree": tune.uniform(0.7, 1.0),
            }
        ray.init(ignore_reinit_error=True)
        analysis = tune.run(
            train_xgboost,
            config=search_space,
            num_samples=n_iter,
            metric=metric,
            mode="min",
            verbose=False
        )

        best_result = analysis.best_result
        best_model = best_result['model']
        best_hyperparameters = best_result["config"]
        best_score = best_result[metric]
        print(f"Best {metric}:", best_score)
        print("Best hyperparameters:", best_hyperparameters)

        best_model.fit(X_train, Y_train,
                       eval_set=[(X_train, Y_train), (X_test, Y_test)],
                       verbose=0)
        eval_results = best_model.evals_result()
        res = best_model.predict(X_test)
        self.model = best_model
        return eval_results, res
