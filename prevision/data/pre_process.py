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

# Class
class location:
    '''
    Cette classe permet de définir une localisation
    '''
    def __init__(self, name:str, lon:int, lat:int, radius:str):
        self.name = name
        self.lon=lon
        self.lat=lat
        self.radius=radius

    def get_location(self):
        return {
            "lon": self.lon,
            "lat": self.lat,
            "radius": self.radius
            }
    def get_location_origin(self):
        return str(round(self.lat,4))+','+str(round(self.lon,4))


# Constantes
class constante() : 
    '''
    Dans cette classe, on retrouve les constantes utilisées dans le pré-processing des données
    '''

    # Localisation du restaurant
    ST_CATH_LOC = location('laCageStCatherine', -73.56362829999999, 45.5077116, '0.5km')
    MONTREAL_TZ='Canada/Eastern'

    
    #File path
    dataDir =  "prevision/data/"
    affluencePath = dataDir + "affluence.csv"
    dataVentePath = dataDir + "data_vente.csv"
    meteoPath = dataDir + "archive.csv"

    # Variables à utiliser pour l'Api de prévision d'attendance PredictHQ
    ATTENDANCE_BASE_CAT = [
        "phq_attendance_community",
        "phq_attendance_concerts",
        "phq_attendance_conferences",
        "phq_attendance_expos",
        "phq_attendance_festivals",
        "phq_attendance_performing_arts",
        "phq_attendance_sports",
        "phq_attendance_academic_graduation",
        "phq_attendance_academic_social",
        # "phq_attendance_academic_academic_session",
        # "phq_attendance_academic_exam",
        # "phq_attendance_academic_holiday",
    ]
    ALL_VIEWERSHIP_CAT = [
        "phq_viewership_sports",
        "phq_viewership_sports_american_football",
        "phq_viewership_sports_american_football_ncaa_men",
        "phq_viewership_sports_american_football_nfl",
        "phq_viewership_sports_auto_racing",
        "phq_viewership_sports_auto_racing_indy_car",
        "phq_viewership_sports_auto_racing_nascar",
        "phq_viewership_sports_baseball",
        "phq_viewership_sports_baseball_mlb",
        "phq_viewership_sports_baseball_ncaa_men",
        "phq_viewership_sports_basketball",
        "phq_viewership_sports_basketball_nba",
        "phq_viewership_sports_basketball_ncaa_men",
        "phq_viewership_sports_basketball_ncaa_women",
        "phq_viewership_sports_boxing",
        "phq_viewership_sports_golf",
        "phq_viewership_sports_golf_masters",
        "phq_viewership_sports_golf_pga_championship",
        "phq_viewership_sports_golf_pga_tour",
        "phq_viewership_sports_golf_us_open",
        "phq_viewership_sports_horse_racing",
        "phq_viewership_sports_horse_racing_belmont_stakes",
        "phq_viewership_sports_horse_racing_kentucky_derby",
        "phq_viewership_sports_horse_racing_preakness_stakes",
        "phq_viewership_sports_ice_hockey",
        "phq_viewership_sports_ice_hockey_nhl",
        "phq_viewership_sports_mma",
        "phq_viewership_sports_mma_ufc",
        "phq_viewership_sports_soccer",
        "phq_viewership_sports_soccer_concacaf_champions_league",
        "phq_viewership_sports_soccer_concacaf_gold_cup",
        "phq_viewership_sports_soccer_copa_america_men",
        "phq_viewership_sports_soccer_fifa_world_cup_women",
        "phq_viewership_sports_soccer_fifa_world_cup_men",
        "phq_viewership_sports_soccer_mls",
        "phq_viewership_sports_soccer_uefa_champions_league_men",
        "phq_viewership_sports_softball",
        "phq_viewership_sports_softball_ncaa_women",
        "phq_viewership_sports_tennis",
        "phq_viewership_sports_tennis_us_open",
        "phq_viewership_sports_tennis_wimbledon"
    ]

    RANK_BASED_CAT = [
        "phq_rank_daylight_savings",
        "phq_rank_health_warnings",
        "phq_rank_observances",
        "phq_rank_public_holidays",
        "phq_rank_school_holidays",
        "phq_rank_politics"
    ]


    ALL_CAT = ATTENDANCE_BASE_CAT + ALL_VIEWERSHIP_CAT + RANK_BASED_CAT

    # Variables à utiliser pour l'Api de prévision d'attendance PredictHQ sur les locations
    MTL_ID={
                "id": "6077246",
                "type": "county",
                "name": "Montr\u00e9al",
                "county": "Montr\u00e9al",
                "region": "Quebec",
                "country": "Canada",
                "country_alpha2": "CA",
                "country_alpha3": "CAN",
                "location": [
                    -73.68248,
                    45.50008
                ]
            }

    LACAGE_ID={
                "id": "6137780",
                "type": "locality",
                "name": "Sainte-Catherine",
                "county": "Mont\u00e9r\u00e9gie",
                "region": "Quebec",
                "country": "Canada",
                "country_alpha2": "CA",
                "country_alpha3": "CAN",
                "location": [
                    -73.58248,
                    45.40008
                ]
            },

    CANADA_ID={
                "id": "6251999",
                "type": "country",
                "name": "Canada",
                "county": None,
                "region": None,
                "country": "Canada",
                "country_alpha2": "CA",
                "country_alpha3": "CAN",
                "location": [
                    -113.64258,
                    60.10867
                ]
            }

class api():
    
    def __init__(self):
        self.access_token = os.getenv("ACCESS_TOKEN_PREDICT_HQ")
        self.phq = Client(access_token=self.access_token)
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
                "location__geo": lieu.get_location(),
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
            if debut_vacances <= date <= fin_vacances:
                return 1

        return 0
    
    def get_data(start_date, end_date,big_chemin = 'prevision/data/'):
        '''
        Cette fonction permet de charger les données d'entrainement
            * Input :  (Str) Chemin vers le dossier contenant les données
            * Output : (DataFrame) Données d'entrainement
        '''
        
    
        '''
        Attendance : Nombre de spectateurs prévus
        '''
        api_instance = api() 
        attendancePath=big_chemin + "affluence.csv"
        attendanceDf = api_instance.get_features_df(start_date, end_date,attendancePath)
        # Fichier des prévisions d'attendance
        attendanceDf=attendanceDf[['date', 'phq_attendance_sports_sum','phq_attendance_conferences_sum','phq_attendance_expos_sum','phq_attendance_concerts_sum','phq_attendance_festivals_sum','phq_attendance_performing_arts_sum','phq_attendance_community_sum']]
        attendanceDf = attendanceDf.rename(columns= {
                                                    'phq_attendance_sports_sum': 'spec_sports',
                                                    'phq_attendance_conferences_sum': 'spec_conferences',
                                                    'phq_attendance_expos_sum': 'spec_expos',
                                                    'phq_attendance_concerts_sum': 'spec_concerts',
                                                    'phq_attendance_festivals_sum': 'spec_festivals',
                                                    'phq_attendance_performing_arts_sum': 'spec_performing_arts',
                                                    'phq_attendance_community_sum': 'spec_community'
                                                    })
        # Convertie la colonne date en datetime
        attendanceDf['date']=pd.to_datetime(attendanceDf['date'], format='%Y-%m-%d')

        '''
        Vente : Vente du restaurant
        '''
        prevSellsPath=big_chemin + "data_vente.csv"
        # Fichier des prévisions de ventes
        prevSellsDf=pd.read_csv(prevSellsPath,sep=';')
        # Convertie la colonne date en datetime
        prevSellsDf['date']=pd.to_datetime(prevSellsDf['date'], format='%d-%m-%Y')

        '''
        Méteo
        '''
        # Chemin vers les fichiers
        meteoPath=big_chemin + "archive.csv"
        # Fichier des prévisions météo
        meteoDf=pd.read_csv(meteoPath)
        # Ne pas prendre en compte les 3 premiere ligne du csv
        meteoDf = meteoDf.iloc[3:]
        # Renommer la colonne time en date
        meteoDf = meteoDf.rename(columns={'time': 'date',
                                          'apparent_temperature_mean (°C)': 'mean_temp',
                                          'rain_sum (mm)': 'rain',
                                          'snowfall_sum (cm)' : 'snow'})
        # Convertie la colonne date en datetime
        meteoDf['date']=pd.to_datetime(meteoDf['date'], format='%Y-%m-%d')

        """
        Merge
        """
        # Concaténer les DataFrames en utilisant la colonne "date" comme clé de fusion
        df = pd.merge(attendanceDf, prevSellsDf, on='date', how='outer')
        df = pd.merge(df, meteoDf, on='date', how='outer')
        # supprimer les lignes avec des valeurs manquantes 
        df = df.dropna()
        # Réinitialiser les indices
        df = df.reset_index(drop=True)

        '''
        Jours de la semaine
        '''
        # On ajoute une colonne date
        df['day'] = df['date'].apply(pre_process.date2day)
        # hot encode day
        df = pd.get_dummies(df, columns=['day'])

        '''
        Vacances
        '''
        # On ajoute une colonne 
        df['vancance'] = df['date'].apply(pre_process.date2vacances)

        '''
        Jours fériés
        '''
        # On ajoute une colonne
        df['ferie'] = df['date'].apply(pre_process.date2jourferie)

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
