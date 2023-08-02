import requests
from .utils import ACCESS_TOKEN_PREDICT_HQ, affluencePath, ATTENDANCE_BASE_CAT, ST_CATH_LOC, meteoPath
import pandas as pd


class api_predicthq():
    '''
    Cette classe permet de récupérer les données de PredictHQ
    '''

    def __init__(self, token=ACCESS_TOKEN_PREDICT_HQ):
        # Get API key
        self.access_token_PHQ = token
        # constantes
        self.selected_cat = ATTENDANCE_BASE_CAT
        self.lieu = ST_CATH_LOC
        self.attendancePath = affluencePath

    def get_id_place(self, address):
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

    def get_df_attendance_90(self, start_date, end_date):
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
        try:
            # Parcourir les résultats et extraire les données
            for result in api_result['results']:
                list_attendance['date'].append(result['date'])
                for event in ATTENDANCE_BASE_CAT:
                    list_attendance[event[4:]].append(result[event]['stats']['sum'])

            # Création du DataFrame à partir des données d'attendance
            df = pd.DataFrame(list_attendance)
            return df
        except:
            raise "Error in Api Call and results :{}".format(api_result['errors'])
            return pd.DataFrame()

    @staticmethod
    def batch_day(start_date, end_date):
        '''
        Cette fonction permet de créer des paquets de 90 jours pour récuperer les données d'attendance.
        '''
        # On crée une liste de date entre start_date et end_date
        date_list = pd.date_range(start_date, end_date).tolist()
        # On crée des paquets de 90 jours
        batch_list = []
        for i in range(0, len(date_list), 90):
            batch_list.append(date_list[i:i + 90])
        # convertir les dates en str et en format YYYY-MM-DD
        for i in range(len(batch_list)):
            batch_list[i] = [str(date)[:10] for date in batch_list[i]]
        return batch_list

    def get_df_attendance(self, start_date, end_date):
        '''
        L'API de PredictHQ ne permet pas de récupérer plus de 90 jours d'événements.
        Cette fonction permet de récupérer les données d'attendance sur une période plus longue.
        '''
        # On crée une liste de date entre start_date et end_date
        date_list = pd.date_range(start_date, end_date).tolist()
        # On verifie si la periode de temps demandé est inférieur à 90 jours
        if len(date_list) <= 90:
            # On récupère les données d'attendance sur la période demandée
            dataframe_attendance = self.get_df_attendance_90(start_date, end_date)
        else:
            # On crée des paquets de 90 jours pour récuperer les données d'attendance
            batch_day_list = api_predicthq.batch_day(start_date, end_date)
            # On récupère les données d'attendance sur chaque paquet de 90 jours
            feature_list = []
            for batch in batch_day_list:
                dataframe_attendance_batch = self.get_df_attendance_90(batch[0], batch[-1])
                feature_list.append(dataframe_attendance_batch)
            # On concatène les données d'attendance
            dataframe_attendance = pd.concat(feature_list, ignore_index=True)

        # On enregistre les données d'attendance
        dataframe_attendance.to_csv(self.attendancePath, index=False)
        return dataframe_attendance

    def update_df_attendance(self, df_attendance):
        '''
        Cette fonction permet de mettre à jour les données d'attendance.
        '''
        # On récupère la date de la dernière ligne du df_attendance
        start_date = df_attendance['date'].max()
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

        # On récupère les données d'attendance sur la période demandée
        df_attendance_new = self.get_df_attendance(start_date, end_date)
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

        if self.attendancePath.exists():
            # On récupère les données d'attendance
            df_attendance = pd.read_csv(self.attendancePath)
            if today_date != df_attendance['date'].max():
                df_attendance = self.update_df_attendance(df_attendance)
            return df_attendance
        else:
            last_year_date = (pd.to_datetime('today') - pd.DateOffset(years=1) + pd.DateOffset(days=1)).strftime(
                '%Y-%m-%d')
            df_attendance = self.get_df_attendance(last_year_date, today_date)
            return df_attendance


class api_weather():

    def __init__(self):
        self.meteoPath = meteoPath
        self.selected_cat = ATTENDANCE_BASE_CAT
        self.lieu = ST_CATH_LOC
        self.attendancePath = affluencePath

    def getLonLat(self):
        return self.lieu['lon'], self.lieu['lat']

    def get_meteodata(self, start_date, end_date):
        # recupere les donnée méteo de https://api.open-meteo.com/v1/
        longitude, latitude = self.getLonLat()
        # latitude = self.latitude
        # longitude = self.longitude
        list_data_daily = ['apparent_temperature_max', 'apparent_temperature_min', 'sunset', 'uv_index_max', 'rain_sum',
                           'showers_sum', 'snowfall_sum', 'precipitation_hours']

        # Passer des liste à des str séparé par des virgules
        list_data_daily = ','.join(list_data_daily)

        lien = 'https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&daily={}&start_date={}&end_date={}&timezone=America%2FNew_York'.format(
            latitude, longitude, list_data_daily, start_date, end_date)

        response = requests.get(lien)
        data = response.json()
        # convert to dataframe
        df_daily = pd.DataFrame(data['daily'])

        # convertie la colonne sunset de 2022-06-17T20:47 en 20,78
        df_daily['sunset'] = round(
            df_daily['sunset'].str[11:13].astype(int) + df_daily['sunset'].str[14:16].astype(int) / 60, 2)
        # save to csv
        df_daily.to_csv(self.meteoPath)
        return df_daily

    def get_today_meteodata(self):
        '''
        Cette fonction permet de récupérer les données météo du jour.
        '''
        # On récupère la date de l'année passée (limite actuelle de notre abonnement à predicthq)
        start_date = (pd.to_datetime('today') - pd.DateOffset(years=1) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        # verifie si le fichier existe
        if self.meteoPath.exists():
            # verifie si la derniere date est la date d'aujourd'hui
            df_meteo = pd.read_csv(self.meteoPath, index_col=0)
            if pd.to_datetime('today').strftime('%Y-%m-%d') != df_meteo['time'].max():
                df_meteo = self.get_meteodata(start_date, end_date)
        else:
            df_meteo = self.get_meteodata(start_date, end_date)
        return df_meteo


if __name__ == '__main__':
    predictHQ = api_predicthq()
    attendanceDf = predictHQ.get_today_df_attendance()
    rows, columns = attendanceDf.shape
    assert rows >= 365, f"We have less than a year of data from predictHQ"
    assert columns == 7, f"We have {columns} columns instead of {7}"
    print("attendanceDf validé")

    weather = api_weather()
    meteoDf = weather.get_today_meteodata()
    rows, columns = meteoDf.shape
    assert columns == 10, f"We have {columns} instead of {10}"
    print("meteoDf validé")
