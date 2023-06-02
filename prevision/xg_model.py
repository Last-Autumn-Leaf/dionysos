# Librairie

# data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# machine learning
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error
from scipy.stats import randint
from pandas.plotting import parallel_coordinates

# log
from datetime import datetime
import logging as lg
lg.basicConfig(level=lg.INFO)



class pre_process():
    '''
    Cette class permet de préparer les données pour l'entrainement du modèle
    '''

    def date2day(date_str):
        # On ajoute une colonne 'Jour' qui contient le jour de la semaine
        '''
        Cette fonction permet de convertire une date en jour de la semaine
        correspondant
            * Input :  (Str) Date '%d-%m-%Y'
            * Output : (Str) Jour de la semaine
        '''
        # Obtention du jour de la semaine (0 = lundi, 1 = mardi, ..., 6 = dimanche)
        jour_semaine = date_str.weekday()
        return jour_semaine

    def get_data(big_chemin = 'prevision/data/'):
        '''
        Cette fonction permet de charger les données d'entrainement
            * Input :  (Str) Chemin vers le dossier contenant les données
            * Output : (DataFrame) Données d'entrainement
        '''
        # Chemin vers les fichiers
        attendancePath=big_chemin + "affluence.csv"
        prevSellsPath=big_chemin + "data_vente.csv"
        meteoPath=big_chemin + "archive.csv"

        # Fichier des prévisions d'attendance
        attendanceDf=pd.read_csv(attendancePath)[['date', 'phq_attendance_stats_sum']]
        attendanceDf = attendanceDf.rename(columns={'phq_attendance_stats_sum': 'attendance'})

        # Fichier des prévisions de ventes
        prevSellsDf=pd.read_csv(prevSellsPath,sep=';')

        # Fichier des prévisions météo
        meteoDf=pd.read_csv(meteoPath)
        # Ne pas prendre en compte les 3 premiere ligne du csv
        meteoDf = meteoDf.iloc[3:]
        # Renommer la colonne time en date
        meteoDf = meteoDf.rename(columns={'time': 'date'})

        meteoDf['date']=pd.to_datetime(meteoDf['date'], format='%Y-%m-%d')
        attendanceDf['date']=pd.to_datetime(attendanceDf['date'], format='%Y-%m-%d')
        prevSellsDf['date']=pd.to_datetime(prevSellsDf['date'], format='%d-%m-%Y')

        # Concaténer les DataFrames en utilisant la colonne "date" comme clé de fusion
        df = pd.merge(attendanceDf, prevSellsDf, on='date', how='outer')
        df = pd.merge(df, meteoDf, on='date', how='outer')

        # supprimer les lignes avec des valeurs manquantes 
        df = df.dropna()


        # On ajoute une colonne date
        df['day'] = df['date'].apply(pre_process.date2day)
        # hot encode day
        df = pd.get_dummies(df, columns=['day'])

        X = df.drop(['vente','date'], axis=1)
        y = df['vente']

        return X, y

    def split(X, y,random_state = 7):
        # segmente les données d'entrainement et de validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=random_state)
        prevision_cage = X_test['prevision'].tolist()
        # Supprimer la colonne prevision
        X_train = X_train.drop(['prevision'], axis=1)
        X_test = X_test.drop(['prevision'], axis=1)

        return X_train, X_test, y_train, y_test, prevision_cage

class modele_xg():
        '''
        Cette classe permet de créer un modèle XGBoost et de l'entrainer sur les données
        '''
        def train(n_estimators=500, max_depth=15, learning_rate=0.01, subsample=0.6, colsample_bytree=0.6, objective='reg:squarederror',plot = False):
            
            # définir le modèle
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,  # Number of trees (boosting rounds)
                max_depth=max_depth,  # Maximum depth of each tree
                learning_rate=learning_rate,  # Step size shrinkage
                subsample=subsample,  # Subsample ratio of the training instances
                colsample_bytree=colsample_bytree,  # Subsample ratio of columns when constructing each tree
                objective=objective  # Loss function to be minimized
            )

            # Train the model en affichant la courbe d'apprentissage
            model.fit( X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse', verbose=False)
            
            # afficher la courbe d'apprentissage
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

        def compare2planifico(model,X_test,y_test,prevision_cage,plot = False):
            predictions = model.predict(X_test)
            erreur_model = mean_squared_error(predictions, y_test)
            erreur_cage  = mean_squared_error(prevision_cage, y_test)
            print("Notre prevision : Mean Absolute Error : {}".format(erreur_model))
            print("Prévision cage : Mean Absolute Error : {}".format(erreur_cage))

            if plot:
                plt.figure(figsize=(20,10))
                plt.plot(y_test.tolist(), color='blue',label='vente')
                plt.plot(predictions, color='red',label='predictions')
                plt.plot(prevision_cage, color='green',label='prevision_cage')
                plt.xlabel('Jours')
                plt.ylabel('Vente')
                plt.legend()
                plt.show()
            
            return erreur_model,erreur_cage

        def hyperparametres_grid(X_train, y_train, X_test, y_test):
            # Définir les hyperparamètres à tester
            param_grid = {
                'n_estimators': [10,100, 200, 500, 1000],
                'max_depth': [5,10, 15,25,50,100],
                'learning_rate': [ 0.5, 0.1, 0.075 ,0.05,0.025 ,0.01, 0.001],
                'subsample': [0.2 , 0.4 ,0.6, 0.8, 1.0],
                'colsample_bytree': [0.2 , 0.4 ,0.6, 0.8, 1.0],
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
    
        def hyperparametres_random(X_train, y_train, X_test, y_test,plot = False):
            # Définir les hyperparamètres à tester
            param_grid = {
                            'n_estimators': [10,100, 200, 500, 1000],
                            'max_depth': [5,10, 15,25,50,100],
                            'learning_rate': [ 0.5, 0.1, 0.075 ,0.05,0.025 ,0.01, 0.001],
                            'subsample': [0.2 , 0.4 ,0.6, 0.8, 1.0],
                            'colsample_bytree': [0.2 , 0.4 ,0.6, 0.8, 1.0],
                            'gamma': [0, 0.1, 0.5, 1.0],
                            'min_child_weight': [1, 3, 5],
                            'reg_alpha': [0, 0.01, 0.1, 1.0],
                            'reg_lambda': [0, 0.01, 0.1, 1.0]
                        }
            # Créer le modèle XGBoost
            model = xgb.XGBRegressor(objective='reg:squarederror')

            # Effectuer la recherche aléatoire des hyperparamètres
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, scoring='neg_root_mean_squared_error', cv=5, verbose=1, random_state=42)
            random_search.fit(X_train, y_train)

            if plot : 
                # Obtenir les résultats sous forme de DataFrame
                results_df = pd.DataFrame(random_search.cv_results_)

                # drop std_fit_time, mean_score_time,std_score_time,rank_test_score, split0_test_score, split1_test_score, split2_test_score, split3_test_score, split4_test_score
                results_df = results_df.drop(columns=['params','std_fit_time', 'mean_score_time','std_score_time','rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score'])
                # change le nom des colonnes avec param sans le prefixe
                results_df.columns = results_df.columns.str.replace('param_', '')
                results_df.columns = results_df.columns.str.replace('_', ' ')
                # Supprimer les colonnes d'index et de paramètres
                results_df = results_df.iloc[:, 2:]

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
                # supprime la legend
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
            print("RMSE :", rmse)

            return best_params, best_model

if __name__ == '__main__':

    # Charger les données
    lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Chargement des données")
    X, y = pre_process.get_data()

    # Segmenter les données
    X_train, X_test, y_train, y_test, prevision_cage = pre_process.split(X, y,random_state = 7)
    lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Données segmentées")
    # Entrainement du modèle
    model = modele_xg.train(n_estimators=500, max_depth=15, learning_rate=0.01, subsample=0.6, colsample_bytree=0.6, objective='reg:squarederror',plot = False)
    lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Modèle entrainé")
    # Comparer les prévisions de notre modèle avec celles de planifico
    lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Comparaison des prévisions")
    erreur_model,erreur_cage = modele_xg.compare2planifico(model,X_test,y_test,prevision_cage,plot = False)
    # Hyperparamètres
    lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Recherche des meilleurs hyperparamètres")
    best_params, best_model = modele_xg.hyperparametres_random(X_train, y_train, X_test, y_test)
    # enregistrer le modèle en pickle
    pickle.dump(best_model, open('prevision/model.pkl', 'wb'))
    lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "]Meilleur modèle enregistré")