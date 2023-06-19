'''
Pré-processing des données pour les modèles de prévision de ventes et d'affluence

Date : 2023-06-06
Auteur : Ilyas Baktache
'''

# Importation des librairies
# Data manipulation
import pandas as pd
# Machine learning
from sklearn.model_selection import train_test_split
# Time
from datetime import datetime

# request
import requests
import json

# Our Lib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pre_processing.constante as constante
from pathlib import Path

# Logging
import logging as lg
lg.basicConfig(level=lg.INFO)

class utils_preprocess():
    '''
    Regroupe les fonctions utilitaires
    '''
    project_name="dionysos"
    def setProjectpath():
        project_dir = Path.cwd()
        while project_dir.name != utils_preprocess.project_name:
            project_dir = project_dir.parent
            if project_dir.parent==project_dir:
                ValueError(f"Le dossier parent '{utils_preprocess.project_name}' n'a pas été trouvé.")
                print("project directory not found")
                break
        return project_dir
    
    def get_api_key(provider):
        # Direction vers le fichier api_key.txt
        project_dir=utils_preprocess.setProjectpath()
        Dir =  project_dir / 'prevision' / 'pre_processing'
        chemin = Dir / 'api_key.txt'
        """Récupère la clé d'API dans le fichier api_key.txt pour le fournisseur spécifié"""
        with open(chemin, 'r') as f:
            api_keys = f.readlines()

        for api_key in api_keys:
            if api_key.startswith(provider):
                return api_key.split(':')[1].strip()
        # Si aucune clé n'a été trouvée pour le fournisseur spécifié
        return None

class api_predicthq():
    '''
    Cette classe permet de récupérer les données de PredictHQ
    '''
    def __init__(self):
        # Get API key
        self.access_token_PHQ = utils_preprocess.get_api_key('PredictHQ')
        # constantes
        self.selected_cat = constante.ATTENDANCE_BASE_CAT
        self.lieu = constante.ST_CATH_LOC
        self.attendancePath = constante.affluencePath
    
    def get_id_place(self,address):
        '''
        Cette fonction permet de récupérer l'id d'un lieu à partir de son adresse

        Parameters
        ----------
        address : str
            Adresse du lieu

        Returns
        -------
        id : str
            Id du lieu
        longitude : float
            Longitude du lieu
        latitude : float
            Latitude du lieu

        '''
        response = requests.get(
            url="https://api.predicthq.com/v1/places/",
            headers={
            "Authorization": f"Bearer {self.access_token_PHQ}",
            "Accept": "application/json"
            },
            params={
                "q": address,
                "limit": 2
            }
        )

        data = response.json()
        id = data['results'][0]['id']
        longitude = data['results'][0]['location'][0]
        latitude = data['results'][0]['location'][1]

        return id, longitude, latitude

    def get_df_attendance_90(self,start_date,end_date):
        '''
        Cette fonction permet de récupérer les données d'attendance pour une période de 90 jours ou moins

        Parameters
        ----------
        start_date : str
            Date de début de la période
        end_date : str
            Date de fin de la période
        location_geo : list
            Coordonnées géographiques du lieu

        Returns
        -------
        df : DataFrame
            DataFrame contenant les données d'attendance

        '''
        # Recuperer les noms des types d'evenements
        ATTENDANCE_BASE_CAT = self.selected_cat
        # Création du dictionnaire de données
        data = {
            "active": {
                "gte": start_date,
                "lte": end_date
            },
            "location": {
                "geo": self.lieu
            }
        }
        # Ajouter les événements au dictionnaire
        for event in ATTENDANCE_BASE_CAT:
            data[event] = True
        # Appel de l'API
        response = requests.post(
            url="https://api.predicthq.com/v1/features",
            headers={
            "Authorization": f"Bearer {self.access_token_PHQ}",
            "Accept": "application/json"
            },
            json=data
        )
        api_result = response.json()
        # Création d'un dictionnaire pour stocker les données finales
        list_attendance = {'date': []}
        # Ajouter les événements au dictionnaire
        for event in ATTENDANCE_BASE_CAT:
            list_attendance[event[4:]] = []
        try : 
            # Parcourir les résultats et extraire les données
            for result in api_result['results']:
                list_attendance['date'].append(result['date'])
                for event in ATTENDANCE_BASE_CAT:
                    list_attendance[event[4:]].append(result[event]['stats']['sum'])

            # Création du DataFrame à partir des données d'attendance
            df = pd.DataFrame(list_attendance)
            return df
        except:
            lg.error("Error in Api Call and results :{}".format(api_result['errors']))
            return pd.DataFrame()
            
    @staticmethod
    def batch_day(start_date,end_date):
        '''
        Cette fonction permet de créer des paquets de 90 jours pour récuperer les données d'attendance.
        '''
        # On crée une liste de date entre start_date et end_date
        date_list = pd.date_range(start_date, end_date).tolist()
        # On crée des paquets de 90 jours
        batch_list = []
        for i in range(0,len(date_list),90):
            batch_list.append(date_list[i:i+90])
        # convertir les dates en str et en format YYYY-MM-DD
        for i in range(len(batch_list)):
            batch_list[i] = [str(date)[:10] for date in batch_list[i]]
        return batch_list

    def get_df_attendance(self,start_date,end_date):
        '''
        L'API de PredictHQ ne permet pas de récupérer plus de 90 jours d'événements.
        Cette fonction permet de récupérer les données d'attendance sur une période plus longue.
        '''
        # On crée une liste de date entre start_date et end_date
        date_list = pd.date_range(start_date, end_date).tolist()
        # On verifie si la periode de temps demandé est inférieur à 90 jours
        if len(date_list) <= 90:
            # On récupère les données d'attendance sur la période demandée
            dataframe_attendance = api_predicthq.get_df_attendance_90(self,start_date, end_date)
        else:
            # On crée des paquets de 90 jours pour récuperer les données d'attendance
            batch_day_list = api_predicthq.batch_day(start_date,end_date)
            # On récupère les données d'attendance sur chaque paquet de 90 jours
            feature_list = []
            for batch in batch_day_list:
                dataframe_attendance_batch = api_predicthq.get_df_attendance_90(self,batch[0],batch[-1])
                feature_list.append(dataframe_attendance_batch)
            # On concatène les données d'attendance
            dataframe_attendance = pd.concat(feature_list, ignore_index=True)
        
        # On enregistre les données d'attendance
        dataframe_attendance.to_csv(self.attendancePath, index=False)
        return dataframe_attendance

    def update_df_attendance(self,df_attendance):
        '''
        Cette fonction permet de mettre à jour les données d'attendance.
        '''
        # On récupère la date de la dernière ligne du df_attendance
        start_date = df_attendance['date'].max()
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

        # On récupère les données d'attendance sur la période demandée
        df_attendance_new = api_predicthq.get_df_attendance(self,start_date, end_date)
        # On supprime les lignes qui sont déjà dans le df_attendance
        df_attendance_new = df_attendance_new[~df_attendance_new['date'].isin(df_attendance['date'])]
        # On ajoute les nouvelles données d'attendance au df_attendance
        df_attendance = pd.concat([df_attendance, df_attendance_new], ignore_index=True)
        # enregistrer le df_attendance
        df_attendance.to_csv(self.attendancePath, index=False)
        return df_attendance

    def get_today_df_attendance(self):
        # On récupère la date du jour
        today_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        
        if os.path.exists(self.attendancePath):
            # On récupère les données d'attendance
            df_attendance = pd.read_csv(self.attendancePath)
            if today_date != df_attendance['date'].max() : 
                df_attendance = api_predicthq.update_df_attendance(self,df_attendance)
            return df_attendance
        else:
            last_year_date = (pd.to_datetime('today') - pd.DateOffset(years=1) + pd.DateOffset(days=1) ).strftime('%Y-%m-%d')
            df_attendance = api_predicthq.get_df_attendance(self,last_year_date, today_date)
            return df_attendance


class api_weather():

    def __init__(self):
        self.latitude = constante.ST_CATH_LOC['lat']
        self.longitude = constante.ST_CATH_LOC['lon']
        self.meteoPath = constante.meteoPath

    def get_meteodata(self, start_date, end_date):
        # recupere les donnée méteo de https://api.open-meteo.com/v1/
        latitude = self.latitude
        longitude = self.longitude
        list_data_daily = ['apparent_temperature_max','apparent_temperature_min','sunset','uv_index_max','rain_sum','showers_sum','snowfall_sum','precipitation_hours']
        
        # Passer des liste à des str séparé par des virgules
        list_data_daily = ','.join(list_data_daily)

        lien =' https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&daily={}&start_date={}&end_date={}&timezone=America%2FNew_York'.format(latitude,longitude,list_data_daily,start_date,end_date)

        response = requests.get(lien)
        data = response.json()
        # convert to dataframe
        df_daily = pd.DataFrame(data['daily'])   

        # convertie la colonne sunset de 2022-06-17T20:47 en 20,78
        df_daily['sunset'] = round(df_daily['sunset'].str[11:13].astype(int) + df_daily['sunset'].str[14:16].astype(int)/60,2)
        # save to csv
        df_daily.to_csv(self.meteoPath)
        return df_daily
        


    def get_today_meteodata(self):
        '''
        Cette fonction permet de récupérer les données météo du jour.
        '''
        # On récupère la date de l'année passée (limite actuelle de notre abonnement à predicthq)
        start_date = (pd.to_datetime('today') - pd.DateOffset(years=1) + pd.DateOffset(days=1) ).strftime('%Y-%m-%d')
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        # verifie si le fichier existe
        if os.path.exists(self.meteoPath):
            # verifie si la derniere date est la date d'aujourd'hui
            df_meteo = pd.read_csv(self.meteoPath)
            if pd.to_datetime('today').strftime('%Y-%m-%d') != df_meteo['time'].max() :
                df_meteo = api_weather.get_meteodata(self,start_date, end_date)
        else:
           df_meteo = api_weather.get_meteodata(self,start_date, end_date)
        return df_meteo
    

class pre_process():
    '''
    Cette classe permet de préparer les données pour l'entrainement du modèle
    '''
    @staticmethod 
    def date2day(date_object):
        # On ajoute une colonne 'Jour' qui contient le jour de la semaine
        '''
        Cette fonction permet de convertire une date en jour de la semaine
        correspondant
            * Input :  (date_object) Date '%d-%m-%Y'
            * Output : (Str) Jour de la semaine
        '''
        # Obtention du jour de la semaine (0 = lundi, 1 = mardi, ..., 6 = dimanche)
        jour_semaine = date_object.weekday()
        return jour_semaine
    
    @staticmethod 
    def date2jourferie(date_obj):
        '''
        Cette fonction permet de convertire une date en jour férié correspondant
            * Input :  (Str) Date '%d-%m-%Y'
            * Output : (Str) Jour férié
        '''

        # convertie l'object date en str
        date_str = date_obj.strftime('%Y-%m-%d')

        # Liste des jours fériés au canada
        jours_feries = [
        '2022-09-05',  # Fête du travail (Labour Day)
        '2022-10-10',  # Action de grâce (Thanksgiving)
        '2022-12-25',  # Noël (Christmas)
        '2022-12-26',  # Lendemain de Noël (Boxing Day)
        '2023-01-01',  # Jour de l'An (New Year's Day)
        '2023-02-20',  # Fête de la famille (Family Day)
        '2023-04-14',  # Vendredi saint (Good Friday)
        '2023-05-22',  # Fête de la Reine (Victoria Day)
        '2023-07-01',  # Fête du Canada (Canada Day)
        '2023-09-04',  # Fête du travail (Labour Day)
        '2023-10-09',  # Action de grâce (Thanksgiving)
        '2023-12-25',  # Noël (Christmas)
        '2023-12-26'   # Lendemain de Noël (Boxing Day) 
        ]
        # Affichage de la saison correspondant à la date donnée
        if date_str in jours_feries:
            return 1
        else:
            return 0   
    @staticmethod   
    def date2vacances(date):
        '''
        Cette fonction permet de vérifier si une date se trouve pendant les vacances au Canada.
            * Input :  (str) Date au format '%d-%m-%Y'
            * Output : (str) 'oui' si c'est pendant les vacances, 'non' sinon
        '''
        # Liste des périodes de vacances au Canada
        vacances = [
            ('2022-12-17', '2023-01-02'),  # Vacances d'hiver
            ('2023-03-04', '2023-03-19'),  # Vacances de mars
            ('2023-06-24', '2023-08-27')   # Vacances d'été
        ]

        # Vérification si la date se trouve pendant les vacances
        for debut, fin in vacances:
            debut_vacances = datetime.strptime(debut, '%Y-%m-%d').date()
            fin_vacances = datetime.strptime(fin, '%Y-%m-%d').date()
            if debut_vacances <= date.date() <= fin_vacances:
                return 1

        return 0

    @staticmethod
    def remove_keys_not_in_list(df, feature_list):
        keys_to_remove = []
        for key in df.keys():
            if key not in feature_list:
                keys_to_remove.append(key)
        df.drop(columns=keys_to_remove, inplace=True)
        return df

    # TODO : Default feature should be in a macro
    @staticmethod 
    def get_data(feature = ['prevision','day_0','day_1','day_2','day_3','day_4','day_5','day_6','vacance','ferie','vente_day_1','vente_day_2','vente_day_3','vente_day_4','vente_day_5','vente_day_6','vente_day_7','apparent_temperature_max','apparent_temperature_min','sunset','uv_index_max','showers_sum','rain_sum','snowfall_sum','precipitation_hours','attendance_concerts','attendance_conferences','attendance_expos','attendance_festivals','attendance_performing_arts','attendance_sports']):
        '''
        Cette fonction permet de charger les données d'entrainement
            * Input :  (Str) Chemin vers le dossier contenant les données
            * Output : (DataFrame) Données d'entrainement
        '''
        # On récupère la date de l'année passée (limite actuelle de notre abonnement à predicthq)
        start_date = (pd.to_datetime('today') - pd.DateOffset(years=1) + pd.DateOffset(days=1) ).strftime('%Y-%m-%d')
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        # -----------------
        # Attendance : Nombre de spectateurs prévus
        # -----------------
        predicthq = api_predicthq()
        attendanceDf = predicthq.get_today_df_attendance()
        # Convertie la colonne date en datetime
        attendanceDf['date']=pd.to_datetime(attendanceDf['date'], format='%Y-%m-%d')
        # -----------------
        # Vente : Vente du restaurant
        # -----------------
        prevSellsPath=constante.dataVentePath
        # Fichier des prévisions de ventes
        prevSellsDf=pd.read_csv(prevSellsPath,sep=';')
        # Convertie la colonne date en datetime
        prevSellsDf['date']=pd.to_datetime(prevSellsDf['date'], format='%d-%m-%Y')
        
        # -----------------
        # Méteo
        # -----------------
        # Fichier des prévisions météo
        api_meteo = api_weather()
        meteoDf = api_meteo.get_today_meteodata()
        # Renommer la colonne time en date
        meteoDf = meteoDf.rename(columns={'time': 'date'})
        # Convertie la colonne date en datetime
        meteoDf['date']=pd.to_datetime(meteoDf['date'], format='%Y-%m-%d')

        # -----------------
        # Merge
        # -----------------
        # Concaténer les DataFrames en utilisant la colonne "date" comme clé de fusion
        df = pd.merge(pd.merge(attendanceDf, prevSellsDf, on='date'), meteoDf, on='date')
        # Réinitialiser les indices
        df = df.reset_index(drop=True)

        #Jours de la semaine
        # On ajoute une colonne date
        df['day'] = df['date'].apply(pre_process.date2day)
        # hot encode day
        df = pd.get_dummies(df, columns=['day'])

        # Vacances
        # On ajoute une colonne 
        df['vacance'] = df['date'].apply(pre_process.date2vacances)

        # Jours fériés
        # On ajoute une colonne
        df['ferie'] = df['date'].apply(pre_process.date2jourferie)
        
        # Ventes des 7 derniers jours
        for i in range(1, 8):
            try : 
                df[f'vente_day_{i}'] = df['vente'].shift(i)
            except : 
                df[f'vente_day_{i}'] = float('nan')

        # Supprimer les lignes avec des valeurs manquantes résultant du décalage
        df = df.dropna()
        X_temp = df.copy()
        # Supprimer les features qui ne sont pas dans la liste
        X = pre_process.remove_keys_not_in_list(X_temp, feature)
        y = df['vente']
        return X, y
    
    @staticmethod 
    def split(X, y,random_state = 7):
        # segmente les données d'entrainement et de validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=random_state)
        prevision_cage = X_test['prevision'].tolist()
        # Supprimer la colonne prevision
        X_train = X_train.drop(['prevision'], axis=1)
        X_test = X_test.drop(['prevision'], axis=1)

        return X_train, X_test, y_train, y_test, prevision_cage

if __name__ == '__main__':

    X,y = pre_process.get_data()

    print(X.columns)