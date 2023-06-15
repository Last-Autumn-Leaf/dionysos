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
# Os 
import os
from dotenv import load_dotenv
# predictHQ
from predicthq import Client

# request
import requests
import json

# Our Lib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pre_processing.constante as constante


class api_predicthq():
    '''
    Cette classe permet de récupérer les données de PredictHQ
    '''
    def __init__(self):
        self.access_token_PHQ = os.getenv("ACCESS_TOKEN_PREDICT_HQ")
        self.phq = Client(access_token=self.access_token_PHQ)
        self.selected_cat = constante.ATTENDANCE_BASE_CAT
        self.lieu = constante.ST_CATH_LOC
        self.affluencePath = constante.affluencePath
    
    def get_features_df(self, start_date, end_date,affluencePath):
        # Cette fonction permet de récupérer les features de PredictHQ pour un lieu et une liste de catégories données
        if not os.path.exists(affluencePath):
            lieu = self.lieu
            features_args = {
                "active__gte": start_date,
                "active__lte": end_date,
                "location__geo": lieu,
            }
            for cat in self.selected_cat:
                features_args[cat + "__stats"] = ["sum", "count"]
                features_args[cat + "__phq_rank"] = {"gt": 0}

            feature_list = []
            for feature in self.phq.features.obtain_features(**features_args):
                feature_dict = {"date": feature.date}
                for cat in self.selected_cat:
                    feature_dict[cat + "_sum"] = feature.data[cat]["sum"]
                    feature_dict[cat + "_count"] = feature.data[cat]["count"]
                    feature_dict[cat + "_rank"] = feature.data[cat]["phq_rank"]
                feature_list.append(feature_dict)
            
            df = pd.DataFrame(feature_list)
            df = df.set_index("date")
            df["phq_attendance_stats_sum"] = df.filter(regex=r'^phq_attendance.*sum$').sum(axis=1)
            df["phq_attendance_stats_count"] = df.filter(regex=r'^phq_attendance.*count').sum(axis=1)
            df.to_csv(affluencePath)
        else :
            df = pd.read_csv(affluencePath)
        return df

class api_weather():

    def __init__(self):
        self.latitude = constante.ST_CATH_LOC['lat']
        self.longitude = constante.ST_CATH_LOC['lon']
        self.meteoPath = constante.meteoPath

    def get_meteodata(self, start_date, end_date):
        # recupere les donnée méteo de https://api.open-meteo.com/v1/
        if not os.path.exists(self.meteoPath):
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
            # save to csv
            df_daily.to_csv(self.meteoPath)
            return df_daily 
        else :
            df_daily = pd.read_csv(self.meteoPath)
            return df_daily

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

    @staticmethod 
    def get_data(start_date, end_date,feature = ['prevision','day_0','day_1','day_2','day_3','day_4','day_5','day_6','vacance','ferie','vente_day_1','vente_day_2','vente_day_3','vente_day_4','vente_day_5','vente_day_6','vente_day_7','apparent_temperature_max','apparent_temperature_min','sunset','uv_index_max','precipitation_sum','showers_sum','rain_sum','snowfall_sum','precipitation_hours','spec_concert','spec_performing_arts','spec_conference','spec_festival','spec_expos','spec_sports'],big_chemin = 'prevision/data/'):
        '''
        Cette fonction permet de charger les données d'entrainement
            * Input :  (Str) Chemin vers le dossier contenant les données
            * Output : (DataFrame) Données d'entrainement
        '''
        # -----------------
        # Attendance : Nombre de spectateurs prévus
        # -----------------
        api_instance = api_predicthq() 
        attendancePath= constante.affluencePath
        attendanceDf = api_instance.get_features_df(start_date, end_date,attendancePath)
        # Fichier des prévisions d'attendance
        attendanceDf=attendanceDf[['date', 'phq_attendance_sports_sum','phq_attendance_conferences_sum','phq_attendance_expos_sum','phq_attendance_concerts_sum','phq_attendance_festivals_sum','phq_attendance_performing_arts_sum']]
        attendanceDf = attendanceDf.rename(columns= {
                                                    'phq_attendance_sports_sum': 'spec_sports',
                                                    'phq_attendance_conferences_sum': 'spec_conferences',
                                                    'phq_attendance_expos_sum': 'spec_expos',
                                                    'phq_attendance_concerts_sum': 'spec_concerts',
                                                    'phq_attendance_festivals_sum': 'spec_festivals',
                                                    'phq_attendance_performing_arts_sum': 'spec_performing_arts'
                                                    })
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
        # Chemin vers les fichiers
        meteoPath=constante.meteoPath
        # Fichier des prévisions météo
        api_meteo = api_weather()
        meteoDf = api_meteo.get_meteodata(start_date, end_date)
        meteoDf=pd.read_csv(meteoPath)
        # Ne pas prendre en compte les 3 premiere ligne du csv
        meteoDf = meteoDf.iloc[3:]
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
        df['vancance'] = df['date'].apply(pre_process.date2vacances)

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

    start_date = '2022-09-01'
    end_date = '2023-06-01'
    
    api_instance = api_predicthq() 
    attendancePath= constante.affluencePath
    attendanceDf = api_instance.get_features_df(start_date, end_date,attendancePath)
    
