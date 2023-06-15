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

# Machine learning
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Log
from datetime import datetime
import logging as lg

lg.basicConfig(level=lg.INFO)

class utils_model():
    ''' 
    Cette classe regroupe les fonctions communes à tous les modèles
    '''

    @staticmethod
    def compare2planifico(model, X_test, y_test, prevision_cage, plot=False):
        """
        Compare les prédictions du modèle avec les prévisions de cage sur les données de test.

        Arguments:
        - model : modèle entraîné
        - X_test : données de test (features)
        - y_test : étiquettes de test
        - prevision_cage : prévisions de cage
        - plot : booléen indiquant si la courbe des prédictions doit être affichée

        Retourne:
        - Erreur MSE du modèle
        - Erreur MSE de la prévision de cage
        """
        predictions = model.predict(X_test)
        erreur_model = mean_squared_error(predictions, y_test)
        erreur_cage = mean_squared_error(prevision_cage, y_test)
        erreur_model = np.sqrt(erreur_model)
        erreur_cage = np.sqrt(erreur_cage)

        absolut_error = mean_absolute_error(predictions, y_test)
        absolut_error_cage = mean_absolute_error(prevision_cage, y_test)

        print(f"Notre prevision --> MSE : {erreur_model}, MAE : {absolut_error}")
        print(f"Prévision cage --> MSE : {erreur_cage}, MAE : {absolut_error_cage}")

        if erreur_cage > erreur_model:
            print("Notre modèle est meilleur que la prévision de cage")
        else:
            print("La prévision de cage est meilleure que notre modèle")

        if plot:
            plt.figure(figsize=(20, 10))
            plt.plot(y_test.tolist(), color='blue', label='vente')
            plt.plot(predictions, color='red', label='predictions')
            plt.plot(prevision_cage, color='green', label='prevision_cage')
            plt.xlabel('Jours')
            plt.ylabel('Vente')
            plt.legend()
            plt.show()

        return erreur_model, erreur_cage

    @staticmethod
    def feature_importance(best_model, X_train):
        """
        Calcule et affiche l'importance des fonctionnalités du modèle.

        Arguments:
        - best_model : meilleur modèle entraîné
        - X_train : données d'entraînement (features)
        """
        # Obtenir l'importance des fonctionnalités
        importance = best_model.feature_importances_

        # Triez les fonctionnalités par importance décroissante
        feature_names = X_train.columns.tolist()
        
        # Regrouper l'importance des day_0, day_1, day_2, day_3, day_4, day_5, day_6 en une seule colonne day
        day_0_index = feature_names.index('day_0')
        day_importance = np.sum(importance[day_0_index:day_0_index + 7])
        importance[day_0_index] = day_importance
        importance = np.delete(importance, range(day_0_index + 1, day_0_index + 7))
        feature_names = np.delete(feature_names, range(day_0_index + 1, day_0_index + 7))
        feature_names[day_0_index] = 'day'

        # Regrouper l'importance des vente_day_1, vente_day_2, vente_day_3, vente_day_4, vente_day_5, vente_day_6, vente_day_7 en une seule colonne vente_day
        feature_names = list(feature_names)
        vente_day_1_index = feature_names.index('vente_day_1')
        vente_day_importance = np.sum(importance[vente_day_1_index:vente_day_1_index + 7])
        importance[vente_day_1_index] = vente_day_importance
        importance = np.delete(importance, range(vente_day_1_index + 1, vente_day_1_index + 7))
        feature_names = np.delete(feature_names, range(vente_day_1_index + 1, vente_day_1_index + 7))
        feature_names[vente_day_1_index] = 'vente_old'

        # Lister les fonctionnalités par importance décroissante
        feature_names = feature_names.tolist()
        feature_names = np.array(feature_names)
        indices = np.argsort(importance)[::-1]

        # Tracer le graphique de l'importance des fonctionnalités
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices], tick_label=feature_names[indices].tolist())
        plt.xticks(rotation=90)
        plt.xlabel('Fonctionnalités')
        plt.ylabel('Importance')
        plt.title('Importance des fonctionnalités')
        plt.show()

