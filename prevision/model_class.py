# Librairie

# Our lib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('dionysos'))))
from pre_processing.pre_process import pre_process

# Data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

# Machine learning
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log
from datetime import datetime
import logging as lg

lg.basicConfig(level=lg.INFO)

import random

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
        day_importance = np.sum(importance[day_0_index:day_0_index + 7])
        importance[day_0_index] = day_importance
        importance = np.delete(importance, range(day_0_index + 1, day_0_index + 7))
        feature_names = np.delete(feature_names, range(day_0_index + 1, day_0_index + 7))
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


class WindowGenerator(Dataset):
    def __init__(self, X, Y, input_sequence_length, output_sequence_length):
        self.X = X
        self.Y = Y
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

    def __len__(self):
        return len(self.X) - self.input_sequence_length - self.output_sequence_length + 1

    def __getitem__(self, idx):
        # x = self.X[idx:idx + self.input_sequence_length]
        x=torch.cat((self.X[idx:idx + self.input_sequence_length],self.Y[idx:idx + self.input_sequence_length][:,None]),1)
        y = self.Y[
            idx + self.input_sequence_length:idx + self.input_sequence_length + self.output_sequence_length]
        return x, y

    def getInOutSize(self):
        x,y=self.__getitem__(0)
        in_size=x.shape[1] if len(x.shape) > 1 else 1
        out_size=y.shape[1] if len(y.shape) > 1 else 1
        return in_size,out_size


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=5, num_layers=1, output_size=1, output_sequence_length=1, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_sequence_length=output_sequence_length
        self.output_size=output_size

    def forward(self, x, h=None):
        out, _ = self.rnn(x, h)
        out = self.fc(out)
        return out[:,-self.output_sequence_length:,:] if self.output_size !=1 else out[:,-self.output_sequence_length:,0]


class modele_rnn():
    '''
    Cette classe permet de créer un modèle RNN et de l'entrainer sur les données
    '''

    class Options:
        def __init__(self, input_sequence_length=10, output_sequence_length=5, hidden_size=128, num_layers=2,
                     output_size=1, batch_size=64, epochs=50, learning_rate=0.01, dataset_split=0.7,optimizer='ADAM',
                     momentum=0.9, weight_decay=1e-5,verbose=True,verbose_mod=20,lossFunction='MSE'):
            self.input_sequence_length = input_sequence_length
            self.output_sequence_length = output_sequence_length
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size
            self.batch_size = batch_size
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.dataset_split = dataset_split
            self.optimizer=optimizer
            self.momentum=momentum
            self.weight_decay=weight_decay
            self.verbose=verbose
            self.verbose_mod=verbose_mod
            self.lossFunction=lossFunction

        def getModelOptions(self):
            return {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_sequence_length': self.output_sequence_length
            }

    def __init__(self, x, y, options=None):
        self.options = options if options else self.Options()

        self.x = torch.tensor(x.to_numpy(dtype=float)).float()
        self.y = torch.tensor(y.to_numpy(dtype=float)).float()
        # self.x = torch.tensor(MinMaxScaler().fit_transform(self.x))
        self.split_dataset()

        train_dataset = WindowGenerator(self.x_train, self.y_train, self.options.input_sequence_length,
                                        self.options.output_sequence_length)
        test_dataset = WindowGenerator(self.x_test, self.y_test, self.options.input_sequence_length,
                                       self.options.output_sequence_length)

        self.input_size,self.output_size = train_dataset.getInOutSize()

        self.train_loader = DataLoader(train_dataset, batch_size=self.options.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.options.batch_size, shuffle=False)

    def split_dataset(self):
        n = len(self.x)
        dataset_split = self.options.dataset_split
        self.x_train = self.x[:int(n * dataset_split)]
        self.y_train = self.y[:int(n * dataset_split)]
        self.x_test = self.x[int(n * dataset_split):]
        self.y_test = self.y[int(n * dataset_split):]

    def train(self, plot=False):
        model = RNN(input_size=self.input_size,
                    output_size=self.output_size,
                    **self.options.getModelOptions())

        if self.options.lossFunction=='MSE':
            criterion = nn.MSELoss()
        elif self.options.lossFunction=='MAE':
            criterion = nn.L1Loss()
        elif self.options.lossFunction=='SMOOTHL1':
            criterion=nn.SmoothL1Loss()
        else :
            return print("unrecognized loss function")
        #TODO : the ADAM string should be in a macro !
        if self.options.optimizer=='ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.options.learning_rate)
        else :
            optimizer = torch.optim.SGD(model.parameters(), lr=self.options.learning_rate,
                    momentum=self.options.momentum, weight_decay=self.options.weight_decay)

        train_losses = []
        test_losses = []
        counter=0
        for epoch in range(self.options.epochs):
            model.train()
            train_loss = 0.0
            for x, y in self.train_loader:

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)  # Enregistrer la perte d'entraînement pour chaque époque

            model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for x, y in self.test_loader:
                    x = x
                    y = y
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    test_loss += loss.item()

                test_loss /= len(self.test_loader)
                test_losses.append(test_loss)  # Enregistrer la perte de test pour chaque époque
            if self.options.verbose:
                counter=(counter+1)%self.options.verbose_mod
                if counter==0 :
                    print(f"Epoch {epoch + 1}/{self.options.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss Evolution')
            plt.legend()
            plt.show()

            self.test(model)

        return model


    def test(self,model):
        # Plot the sequence and predicted sequence
        model.eval()
        with torch.no_grad():
            # Get the first sequence from the test dataset
            # if we try to test the over fitting we should take train_loader else take test_loader
            x_plot, target_sequence = self.train_loader.dataset[0]
            x_plot = x_plot.unsqueeze(0)  # Add a batch dimension

            predicted_sequence = model(x_plot)
            predicted_sequence = predicted_sequence.squeeze().numpy()

            history_sequence = x_plot[0,:, -1]
            # Plot the original sequence
            plt.plot(range(self.options.input_sequence_length+self.options.output_sequence_length),
                     torch.cat((history_sequence,target_sequence)), label='True sequence',
                        linestyle="-", marker="o",)

            plt.plot(range(self.options.input_sequence_length,
                        self.options.input_sequence_length+self.options.output_sequence_length),predicted_sequence.tolist(),
                        linestyle="-", marker="o",label='predicted sequence')


            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title('Sequence and Predicted Sequence')
            plt.legend()
            plt.show()

    # @staticmethod
    # def hyperparametres_random_search(X_train, X_test, y_train, y_test, input_sequence_length, output_sequence_length, output_size, batch_size, epochs, learning_rate, hidden_size_range, num_layers_range, param_grid, num_iterations=20):
    #     # Adapter les données pour le modèle
    #     X_train, X_test, y_train, y_test = modele_rnn.get_data(X_train, X_test, y_train, y_test)
    #
    #     # Créer les datasets et les dataloaders
    #     train_dataset = WindowGenerator(X_train, y_train, input_sequence_length, output_sequence_length)
    #     test_dataset = WindowGenerator(X_test, y_test, input_sequence_length, output_sequence_length)
    #
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #     # Définir les paramètres de la recherche aléatoire
    #     best_params = {}
    #     best_model = None
    #     best_rmse = float('inf')
    #
    #     # Generate all possible combinations of hyperparameters
    #     hyperparameter_combinations = list(itertools.product(hidden_size_range, num_layers_range, *param_grid.values()))
    #     #
    #     for _ in range(num_iterations):
    #         # Randomly select hyperparameter combination
    #         hyperparameters = random.choice(hyperparameter_combinations)
    #         hidden_size, num_layers = hyperparameters[:2]
    #         other_hyperparameters = hyperparameters[2:]
    #
    #         model = RNN(X_train.shape[1], hidden_size, num_layers, output_size, output_sequence_length).to(device)
    #         model.double()
    #
    #         criterion = nn.MSELoss()
    #         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #
    #         for epoch in range(epochs):
    #             model.train()
    #             train_loss = 0.0
    #
    #             for x, y in train_loader:
    #                 x = x.to(device).double()
    #                 y = y.to(device).double()
    #
    #                 optimizer.zero_grad()
    #                 outputs = model(x)
    #                 loss = criterion(outputs, y)
    #                 loss.backward()
    #                 optimizer.step()
    #
    #                 train_loss += loss.item()
    #
    #             train_loss /= len(train_loader)
    #
    #             model.eval()
    #             test_loss = 0.0
    #
    #             with torch.no_grad():
    #                 for x, y in test_loader:
    #                     x = x.to(device).double()
    #                     y = y.to(device).double()
    #                     outputs = model(x)
    #                     loss = criterion(outputs, y)
    #                     test_loss += loss.item()
    #
    #                 test_loss /= len(test_loader)
    #
    #             if test_loss < best_rmse:
    #                 best_rmse = test_loss
    #                 best_params = {'hidden_size': hidden_size, 'num_layers': num_layers, **dict(zip(param_grid.keys(), other_hyperparameters))}
    #                 best_model = model
    #
    #     print("Meilleurs hyperparamètres :", best_params)
    #     print("RMSE sur les données de test :", np.sqrt(best_rmse))
    #
    #     with open('prevision/model/rnn.pkl', 'wb') as f:
    #         pickle.dump(best_model, f)
    #
    #     return best_params, best_model
    #
    # @staticmethod
    # def load_model(path):
    #     # Charger le modèle à partir du fichier pickle
    #     with open(path, 'rb') as f:
    #         model = pickle.load(f)
    #     return model


def fixSeed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

fixSeed()
if __name__ == '__main__':
    # Date de début et de fin de la saison
    start_date = "2023-03-08"
    end_date = "2023-06-05"

    # Initialisation des hyperparamètres
    parameters = {
        "input_sequence_length": 7,
        "output_sequence_length": 2,
        "hidden_size": 124,
        "num_layers": 32,
        "output_size": 1,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 1,
        'dataset_split': 0.7,
        'optimizer':'SGD',
        'momentum' :0.9,
        'weight_decay':1e-5,
        'lossFunction':'SMOOTHL1'
    }
    options = modele_rnn.Options(**parameters)
    X, y = pre_process.get_data(start_date, end_date)
    rnn_model = modele_rnn(X, y, options=options)

    # Entraînement du modèle
    model = rnn_model.train(plot=True)

