from .api import api_predicthq, api_weather
from datetime import datetime
import pandas as pd
from .utils import jours_feries, vacances, ALL_FEATURES, dataVentePath, mma_path, nba_path, nfl_path, nhl_path, \
    dailySalesPath, meteoPath, affluencePath


def date2day(date_object):
    # On ajoute une colonne 'Jour' qui contient le jour de la semaine
    '''
    Cette fonction permet de convertire une date en jour de la semaine
    correspondant
        * Input :  (date_object) Date '%d-%m-%Y'
        * Output : (Str) Jour de la semaine
    '''
    # Obtention du jour de la semaine (0 = lundi, 1 = mardi, ..., 6 = dimanche)
    return date_object.weekday()


def date2jourferie(date_obj):
    '''
    Cette fonction permet de convertire une date en jour férié correspondant
        * Input :  (Str) Date '%d-%m-%Y'
        * Output : (bool) True if it's a Jour férié
    '''
    return date_obj.strftime('%Y-%m-%d') in jours_feries


def date2vacances(date):
    '''
    Cette fonction permet de vérifier si une date se trouve pendant les vacances au Canada.
        * Input :  (str) Date au format '%d-%m-%Y'
        * Output : (bool) True si c'est pendant les vacances, 'non' sinon
    '''
    # Liste des périodes de vacances au Canada

    # Vérification si la date se trouve pendant les vacances
    for debut, fin in vacances:
        debut_vacances = datetime.strptime(debut, '%Y-%m-%d').date()
        fin_vacances = datetime.strptime(fin, '%Y-%m-%d').date()
        if debut_vacances <= date.date() <= fin_vacances:
            return 1
    return 0


def remove_keys_not_in_list(df, feature_list):
    a = set(df.keys())
    b = set(feature_list)
    union = list(a.intersection(b))
    return df[union]


def get_all_data():
    '''
    Cette fonction permet de charger les données d'entrainement
        * Input :  (Str) Chemin vers le dossier contenant les données
        * Output : (DataFrame) Données d'entrainement
    '''
    # On récupère la date de l'année passée (limite actuelle de notre abonnement à predicthq)
    start_date = (pd.to_datetime('today') - pd.DateOffset(years=1) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    # -----------------
    # Attendance : Nombre de spectateurs prévus
    # -----------------
    predicthq = api_predicthq()
    attendanceDf = predicthq.get_today_df_attendance()
    # Convertie la colonne date en datetime
    attendanceDf['date'] = pd.to_datetime(attendanceDf['date'], format='%Y-%m-%d')
    attendanceDf.set_index('date', inplace=True)
    # attendanceDf.dropna()

    # -----------------
    # Vente : Vente du restaurant
    # -----------------
    # Todo : changement de path
    # prevSellsPath = dataVentePath
    # # Fichier des prévisions de ventes
    # prevSellsDf = pd.read_csv(prevSellsPath, sep=';')
    # # Convertie la colonne date en datetime
    # prevSellsDf['date'] = pd.to_datetime(prevSellsDf['date'], format='%d-%m-%Y')
    prevSellsDf = pd.read_csv(dailySalesPath, parse_dates=['Date'], index_col='Date').rename_axis('date')
    prevSellsDf.dropna()

    # -----------------
    # Méteo
    # -----------------
    # Fichier des prévisions météo
    api_meteo = api_weather()
    meteoDf = api_meteo.get_today_meteodata()
    # Renommer la colonne time en date
    meteoDf = meteoDf.rename(columns={'time': 'date'})
    # Convertie la colonne date en datetime
    meteoDf['date'] = pd.to_datetime(meteoDf['date'], format='%Y-%m-%d')
    meteoDf.set_index('date', inplace=True)
    # meteoDf.dropna()

    # -----------------
    # Merge
    # -----------------
    # Concaténer les DataFrames en utilisant la colonne "date" comme clé de fusion
    df = pd.merge(pd.merge(prevSellsDf, meteoDf, on='date'), attendanceDf, on='date').dropna()
    # Réinitialiser les indices
    # df = df.reset_index(drop=True)

    df = addDates(df)
    df = addSportBroadcast(df)
    df = df.dropna()
    X = df.drop('vente', axis=1)
    Y = df['vente']
    return X, Y


def addDates(df):
    # Jours de la semaine
    # On ajoute une colonne date
    df['day'] = df.index.map(date2day)
    # hot encode day
    df = pd.get_dummies(df, columns=['day'])

    # Vacances
    # On ajoute une colonne
    df['vacance'] = df.index.map(date2vacances)

    # Jours fériés
    # On ajoute une colonne
    df['ferie'] = df.index.map(date2jourferie)
    return df

def addSportBroadcast(df):

    csv_files = [
        (mma_path, 'match_mma'),
        (nba_path, 'match_nba'),
        (nfl_path, 'match_nfl'),
        (nhl_path, 'match_nhl')
    ]

    for file_info in csv_files:
        file_name, column_name = file_info
        schedule = pd.read_csv(file_name, sep=',', parse_dates=['date'], index_col='date')
        schedule = schedule.rename(columns={'Match': column_name})
        df = pd.merge(df, schedule, on='date', how='left')

    # Remplacez les valeurs NaN par 0
    df = df.fillna(0)

    return df


def get_data_filtered_data(features=ALL_FEATURES):
    X, Y = get_all_data()
    X = remove_keys_not_in_list(X, features)
    return X, Y


def getAllDataFromCSV():
    df_sales = pd.read_csv(dailySalesPath, parse_dates=['Date'], index_col='Date').rename_axis('date')
    df_meteo = pd.read_csv(meteoPath, parse_dates=['time'], index_col='time').rename_axis('date').drop(
        'Unnamed: 0', axis=1)
    df_predictHQ = pd.read_csv(affluencePath, parse_dates=['date'], index_col='date')
    df_all = pd.merge(pd.merge(df_sales, df_meteo, on='date'), df_predictHQ, on='date').dropna()
    df_all = addDates(df_all)
    df_all = addSportBroadcast(df_all)

    df = df_all.dropna()
    X = df.drop('vente', axis=1)
    Y = df['vente']
    return df, X, Y


# def split(X, y, test_size=0.2, random_state=7, shuffle=True):
#     # segmente les données d'entrainement et de validation
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
#                                                         shuffle=shuffle)
#     prevision_cage = X_test['prevision'].tolist()
#     # Supprimer la colonne prevision
#     X_train = X_train.drop(['prevision'], axis=1)
#     X_test = X_test.drop(['prevision'], axis=1)
#
#     return X_train, X_test, y_train, y_test, prevision_cage


if __name__ == '__main__':
    a = get_all_data()
    b = get_data_filtered_data()
    print(f'input size={len(a[0].columns)}')
    print("everything works")
