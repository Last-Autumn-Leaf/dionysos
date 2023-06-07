# Librairie

# Our lib
import os
import sys
sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath('dionysos') ) ) )
from data.pre_process import pre_process

# data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import itertools

# machine learning
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# log
from datetime import datetime
import logging as lg
lg.basicConfig(level=lg.INFO)

class GeneralModel:
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
        day_importance = np.sum(importance[day_0_index:day_0_index+7])
        importance[day_0_index] = day_importance
        importance = np.delete(importance, range(day_0_index+1, day_0_index+7))
        feature_names = np.delete(feature_names, range(day_0_index+1, day_0_index+7))
        feature_names[day_0_index] = 'day'
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


class ModeleXG:
    """
    Cette classe permet de créer un modèle XGBoost et de l'entraîner sur les données
    """

    @staticmethod
    def train(X_train, X_test, y_train, y_test, n_estimators=500, max_depth=15, learning_rate=0.01, subsample=0.6, colsample_bytree=0.6, objective='reg:squarederror', plot=False):
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
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, scoring='neg_root_mean_squared_error', cv=5, verbose=1, random_state=42)
        elif param == 'MAE':
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, scoring='neg_mean_absolute_error', cv=5, verbose=1, random_state=42)
        random_search.fit(X_train, y_train)

        if plot:
            # Obtenir les résultats sous forme de DataFrame
            results_df = pd.DataFrame(random_search.cv_results_)

            # Supprimer les colonnes inutiles
            columns_to_drop = ['params', 'std_fit_time', 'mean_score_time', 'std_score_time', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score']
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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_sequence_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.outlSeqLen = output_sequence_length

    def forward(self, x):
        bsize = x.size(0)
        h0 = torch.zeros(self.num_layers, bsize, self.hidden_size, dtype=torch.double).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -self.outlSeqLen:, :])  # Select the last output_sequence_length time steps and pass through the linear layer
        return out.view(bsize, self.outlSeqLen)

class WindowGenerator(Dataset):
    def __init__(self, X, Y, input_sequence_length, output_sequence_length):
        self.X = X
        self.Y = Y
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

    def __len__(self):
        return len(self.X) - self.input_sequence_length - self.output_sequence_length + 1

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.input_sequence_length]
        y = self.Y[idx + self.input_sequence_length:idx + self.input_sequence_length + self.output_sequence_length]
        return x, y

class modele_rnn():
    '''
    Cette classe permet de créer un modèle RNN et de l'entrainer sur les données
    '''
    @staticmethod
    def get_data(X_train, X_test, y_train, y_test):
        '''
        Cette fonction permet de charger les données et de les préparer pour l'entrainement du modèle
        '''
        # Convertir les DataFrames en tableaux NumPy
        X_train_np = X_train.to_numpy()
        X_test_np = X_test.to_numpy()
        y_train_np = y_train.to_numpy()
        y_test_np = y_test.to_numpy()

        # Convertir les tableaux NumPy en tenseurs PyTorch
        X_train = torch.tensor(X_train_np).double()
        X_test = torch.tensor(X_test_np).double()
        y_train = torch.tensor(y_train_np).double()
        y_test = torch.tensor(y_test_np).double()


        # Créer un objet MinMaxScaler
        scaler = MinMaxScaler()
        # Normaliser les données d'entraînement
        X_train = scaler.fit_transform(X_train)
        # Normaliser les données de test
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
       
    @staticmethod
    def train(X_train, X_test, y_train, y_test, input_sequence_length, output_sequence_length, hidden_size, num_layers, output_size, batch_size, epochs, learning_rate, plot=False):
        
        # Adapter les données pour le modèle
        X_train, X_test, y_train, y_test = modele_rnn.get_data(X_train, X_test, y_train, y_test)

        # Créer les datasets et les dataloaders
        train_dataset = WindowGenerator(X_train, y_train, input_sequence_length, output_sequence_length)
        test_dataset = WindowGenerator(X_test, y_test, input_sequence_length, output_sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = RNN(X_train.shape[1], hidden_size, num_layers, output_size, output_sequence_length).to(device)
        model.double()  # Ajouter cette ligne pour convertir le modèle en double

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []  # Liste pour enregistrer les pertes d'entraînement
        test_losses = []  # Liste pour enregistrer les pertes de test

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for x, y in train_loader:
                x = x.to(device).double()
                y = y.to(device).double()

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)  # Enregistrer la perte d'entraînement pour chaque époque

            model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device).double()
                    y = y.to(device).double()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    test_loss += loss.item()

                test_loss /= len(test_loader)
                test_losses.append(test_loss)  # Enregistrer la perte de test pour chaque époque

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss Evolution')
            plt.legend()
            plt.show()

        return model

    @staticmethod
    def hyperparametres_random_search(X_train, X_test, y_train, y_test, input_sequence_length, output_sequence_length, output_size, batch_size, epochs, learning_rate, hidden_size_range, num_layers_range, param_grid, num_iterations=20):
        # Adapter les données pour le modèle
        X_train, X_test, y_train, y_test = modele_rnn.get_data(X_train, X_test, y_train, y_test)
        
        # Créer les datasets et les dataloaders
        train_dataset = WindowGenerator(X_train, y_train, input_sequence_length, output_sequence_length)
        test_dataset = WindowGenerator(X_test, y_test, input_sequence_length, output_sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Définir les paramètres de la recherche aléatoire
        best_params = {}
        best_model = None
        best_rmse = float('inf')

        # Generate all possible combinations of hyperparameters
        hyperparameter_combinations = list(itertools.product(hidden_size_range, num_layers_range, *param_grid.values()))
        # 
        for _ in range(num_iterations):
            # Randomly select hyperparameter combination
            hyperparameters = random.choice(hyperparameter_combinations)
            hidden_size, num_layers = hyperparameters[:2]
            other_hyperparameters = hyperparameters[2:]

            model = RNN(X_train.shape[1], hidden_size, num_layers, output_size, output_sequence_length).to(device)
            model.double()

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0

                for x, y in train_loader:
                    x = x.to(device).double()
                    y = y.to(device).double()

                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                model.eval()
                test_loss = 0.0

                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(device).double()
                        y = y.to(device).double()
                        outputs = model(x)
                        loss = criterion(outputs, y)
                        test_loss += loss.item()

                    test_loss /= len(test_loader)

                if test_loss < best_rmse:
                    best_rmse = test_loss
                    best_params = {'hidden_size': hidden_size, 'num_layers': num_layers, **dict(zip(param_grid.keys(), other_hyperparameters))}
                    best_model = model

        print("Meilleurs hyperparamètres :", best_params)
        print("RMSE sur les données de test :", np.sqrt(best_rmse))

        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        return best_params, best_model

    @staticmethod
    def load_model(path):
        # Charger le modèle à partir du fichier pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

if __name__ == '__main__':
    # Date de début et de fin de la saison
    start_date = "2023-03-08"
    end_date = "2023-06-05"

    X,y = pre_process.get_data(start_date,end_date)
    X_train, X_test, y_train, y_test, prevision_cage = pre_process.split(X, y,random_state = 42)

    # Initialisation des hyperparamètres
    input_sequence_length = 10
    output_sequence_length = 5
    hidden_size = 128
    num_layers = 2
    output_size = 1
    batch_size = 64
    epochs = 50
    learning_rate = 0.01

    # Entraînement du modèle
    model = modele_rnn.train(X_train, X_test, y_train, y_test, input_sequence_length, output_sequence_length, hidden_size, num_layers, output_size, batch_size, epochs, learning_rate, plot=True)