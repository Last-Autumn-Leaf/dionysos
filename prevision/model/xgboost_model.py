'''
Cette classe permet de créer un modèle XGBoost et de l'entraîner sur les données

Date : 06/06/2023
@Auteur : Ilyas Baktache
'''


# Librairie

# Our lib
import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from pre_processing.pre_process import pre_process

# Data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Machine learning
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.plotting import parallel_coordinates

# Log
from datetime import datetime
import logging as lg

lg.basicConfig(level=lg.INFO)

import random

class ModeleXG:
    """
    Cette classe permet de créer un modèle XGBoost et de l'entraîner sur les données
    """

    @staticmethod
    def train(X_train, X_test, y_train, y_test, n_estimators=500, max_depth=15, learning_rate=0.01, subsample=0.6,
              colsample_bytree=0.6, objective='reg:squarederror', plot=False):
        """
        Entraîne le modèle XGBoost sur les données d'entraînement et retourne le modèle entraîné.

        Arguments:
        - X_train : données d'entraînement (features)
        - X_test : données de validation (features)
        - y_train : étiquettes d'entraînement
        - y_test : étiquettes de validation
        - n_estimators : nombre d'estimateurs (nombre d'itérations de boosting)
        - max_depth : profondeur maximale de chaque arbre de décision
        - learning_rate : taux d'apprentissage (step size shrinkage)
        - subsample : ratio de sous-échantillonnage des instances d'entraînement
        - colsample_bytree : ratio de sous-échantillonnage des colonnes lors de la construction de chaque arbre
        - objective : fonction de perte à minimiser
        - plot : booléen indiquant si la courbe d'apprentissage doit être affichée

        Retourne:
        - Le modèle XGBoost entraîné
        """

        # Définir le modèle
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective
        )

        # Entraîner le modèle en affichant la courbe d'apprentissage
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse', verbose=False)

        # Afficher la courbe d'apprentissage
        if plot:
            results = model.evals_result()
            train_loss = results['validation_0']['rmse']
            val_loss = results['validation_1']['rmse']

            plt.figure(figsize=(15, 8))
            plt.plot(train_loss, label='Train')
            plt.plot(val_loss, label='Validation')
            plt.xlabel('Boosting Round')
            plt.ylabel('RMSE')
            plt.title('XGBoost Loss Evolution')
            plt.legend()
            plt.show()

        return model

    @staticmethod
    def hyperparametres_grid(X_train, y_train, X_test, y_test):
        """
        Effectue une recherche par grille des meilleurs hyperparamètres pour le modèle XGBoost.

        Arguments:
        - X_train : données d'entraînement (features)
        - y_train : étiquettes d'entraînement
        - X_test : données de test (features)
        - y_test : étiquettes de test

        Retourne:
        - Meilleurs hyperparamètres trouvés
        - Meilleur modèle entraîné
        """
        # Définir les hyperparamètres à tester
        param_grid = {
            'n_estimators': [10, 100, 200, 500, 1000],
            'max_depth': [5, 10, 15, 25, 50, 100],
            'learning_rate': [0.5, 0.1, 0.075, 0.05, 0.025, 0.01, 0.001],
            'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
            'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.5, 1.0],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.01, 0.1, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 1.0]
        }

        # Créer le modèle XGBoost
        model = xgb.XGBRegressor(objective='reg:squarederror')

        # Créer l'objet GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error')

        # Effectuer la recherche par grille sur les données d'entraînement
        grid_search.fit(X_train, y_train)

        # Afficher les meilleurs paramètres trouvés
        print("Meilleurs paramètres :", grid_search.best_params_)

        # Obtenir le meilleur modèle entraîné
        best_model = grid_search.best_estimator_

        # Évaluer le modèle sur les données de test
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("RMSE sur les données de test :", rmse)

        return grid_search.best_params_, best_model

    @staticmethod
    def hyperparametres_random(X_train, y_train, X_test, y_test, plot=False, param='MSE'):
        """
        Effectue une recherche aléatoire des meilleurs hyperparamètres pour le modèle XGBoost.

        Arguments:
        - model : modèle XGBoost
        - X_train : données d'entraînement (features)
        - y_train : étiquettes d'entraînement
        - X_test : données de test (features)
        - y_test : étiquettes de test
        - plot : booléen indiquant si le plot des résultats doit être affiché
        - param : paramètre pour la métrique d'évaluation ('MSE' ou 'MAE')

        Retourne:
        - Meilleurs hyperparamètres trouvés
        - Meilleur modèle entraîné
        """
        # Définir les hyperparamètres à tester
        param_grid = {
            'n_estimators': [10, 100, 150, 200, 250, 500, 1000],
            'max_depth': [5, 10, 15, 25, 50, 75, 100],
            'learning_rate': [0.5, 0.1, 0.075, 0.05, 0.025, 0.01, 0.001],
            'subsample': [0.05, 0.1, 0.2, 0.4, 0.6],
            'colsample_bytree': [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.5, 1.0],
            'min_child_weight': [1, 2, 3, 5],
            'reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
            'reg_lambda': [0, 0.001, 0.01, 0.1, 1.0]
        }

        # Créer le modèle XGBoost
        model = xgb.XGBRegressor(objective='reg:squarederror')

        # Effectuer la recherche aléatoire des hyperparamètres
        if param == 'MSE':
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20,
                                               scoring='neg_root_mean_squared_error', cv=5, verbose=1, random_state=42)
        elif param == 'MAE':
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20,
                                               scoring='neg_mean_absolute_error', cv=5, verbose=1, random_state=42)
        random_search.fit(X_train, y_train)

        if plot:
            # Obtenir les résultats sous forme de DataFrame
            results_df = pd.DataFrame(random_search.cv_results_)

            # Supprimer les colonnes inutiles
            columns_to_drop = ['params', 'std_fit_time', 'mean_score_time', 'std_score_time', 'rank_test_score',
                               'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score',
                               'split4_test_score']
            results_df = results_df.drop(columns=columns_to_drop)

            # Renommer les colonnes
            results_df.columns = results_df.columns.str.replace('param_', '')
            results_df.columns = results_df.columns.str.replace('_', ' ')

            # Ajouter les colonnes de temps et de RMSE
            results_df['Time'] = random_search.cv_results_['mean_fit_time']
            results_df['RMSE'] = -random_search.cv_results_['mean_test_score']

            # Adapter les plages des axes pour chaque hyperparamètre
            for col in results_df.columns[:-2]:
                min_value = results_df[col].min()
                max_value = results_df[col].max()
                results_df[col] = (results_df[col] - min_value) / (max_value - min_value)

            # Créer le Parallel Coordinates Plot
            plt.figure(figsize=(12, 6))
            plt.title('Parallel Coordinates Plot - Hyperparameters vs. Time and RMSE')
            plt.xlabel('Hyperparameter')
            plt.ylabel('Normalized Value')
            plt.grid(True)

            # Tracer le Parallel Coordinates Plot
            parallel_coordinates(results_df, class_column='RMSE', colormap='viridis')

            plt.xticks(rotation=45)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().legend_.remove()
            plt.tight_layout()
            plt.show()

        # Obtenir les meilleurs hyperparamètres
        best_params = random_search.best_params_
        print("Meilleurs hyperparamètres :", best_params)

        # Entraîner le modèle final avec les meilleurs hyperparamètres
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)

        # Évaluer les performances du modèle sur les données de test
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print("MSE :", rmse)
        print("MAE :", mae)

        return best_params, best_model

    @staticmethod
    def save_best_model(start_date, end_date):
        """
        Enregistre le meilleur modèle XGBoost entraîné sur les données spécifiées.

        Arguments:
        - start_date : date de début des données
        - end_date : date de fin des données
        """

        # Charger les données
        lg.info(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Chargement des données")
        X, y = pre_process.get_data(start_date, end_date)

        # Segmenter les données
        X_train, X_test, y_train, y_test, prevision_cage = pre_process.split(X, y, random_state=7)
        lg.info(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Données segmentées")

        # Recherche des meilleurs hyperparamètres
        lg.info(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Recherche des meilleurs hyperparamètres")
        best_params, best_model = ModeleXG.hyperparametres_random(X_train, y_train, X_test, y_test)

        # Enregistrer le modèle en pickle
        pickle.dump(best_model, open('prevision/model/xg_boost.pkl', 'wb'))
        lg.info(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Le meilleur modèle a été enregistré")

if __name__ == '__main__' :
    print("Entraînement du modèle XGBoost")