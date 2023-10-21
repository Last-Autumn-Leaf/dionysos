from .api import api_predicthq, api_weather
from datetime import datetime
import pandas as pd
import numpy as np
from .utils import jours_feries, vacances, ALL_FEATURES, dataVentePath, mma_path, nba_path, nfl_path, nhl_path, \
    dailySalesPath, meteoPath, affluencePath, hourlySalesPath, modeStr2i, MODE_DAILY_SALES, MODE_HOURLY_SALES, \
    MODE_HOURLY_CLIENT, isModeValid, meteoVCPath, DATE_COL


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


isDfHourly = lambda x: 'date_time' in x.columns and 'periode' in x.columns
isDfDaily = lambda x: 'Date' in x.columns and 'vente' in x.columns and len(x.columns) == 2


def getPrevSells(hourly: bool, dataPath=None) -> (pd.DataFrame, list):
    if dataPath is not None:
        df = pd.read_csv(dataPath)
        if isDfHourly(df):
            hourly = True
            print("Hourly format detected")
        elif isDfDaily(df):
            hourly = False
            print("Daily format detected")
        else:
            raise ValueError("Format non détecté")
    else:
        dataPath = hourlySalesPath if hourly else dailySalesPath

    if hourly:
        df_sales = pd.read_csv(dataPath).drop(['sources', 'periode', 'day'], axis=1)
        df_sales['date_time'] = pd.to_datetime(df_sales['date_time']).dt.date

        X = df_sales.values
        listHours = ["{:02d}:{:02d}".format(hour, minute) for hour in range(24) for minute in range(0, 60, 30)]
        steps = 24 * 2
        ndays = len(X) // (steps)
        newX = np.zeros((ndays, steps))
        dt = [0] * ndays
        for i in range(ndays):
            dt[i] = X[i * steps, 0]
            newX[i] = X[i * steps:steps * (i + 1), 1]
        start_i = 23
        encoding_h_str = 'h_'
        col_names = [f'{encoding_h_str}{x}' for x in listHours[start_i:]]
        prevSellsDf = pd.DataFrame(newX[:, start_i:], columns=col_names)
        prevSellsDf['date'] = dt
        prevSellsDf['date'] = prevSellsDf['date'].astype('datetime64[ns]')
        prevSellsDf.set_index('date', inplace=True)
    else:
        col_names = ['vente']
        prevSellsDf = pd.read_csv(dataPath, parse_dates=['Date'], index_col='Date').rename_axis('date')

    return prevSellsDf.dropna(), col_names


def getDataFromMode(mode, dataPath):
    if not isModeValid(mode):
        raise ValueError(f"invalid mode {mode}")
    if type(mode) == str:
        mode = modeStr2i[mode]

    if mode == MODE_DAILY_SALES:
        col_names = ['vente']
        prevSellsDf = pd.read_csv(dataPath, parse_dates=['Date']).rename(columns={'Date': DATE_COL})
    elif mode == MODE_HOURLY_SALES or mode == MODE_HOURLY_CLIENT:
        df_sales = pd.read_csv(dataPath).drop(['sources', 'periode', 'day'], axis=1)
        df_sales['date_time'] = pd.to_datetime(df_sales['date_time']).dt.date

        X = df_sales.values
        listHours = ["{:02d}:{:02d}".format(hour, minute) for hour in range(24) for minute in range(0, 60, 30)]
        steps = 24 * 2
        ndays = len(X) // (steps)
        newX = np.zeros((ndays, steps))
        dt = [0] * ndays
        for i in range(ndays):
            dt[i] = X[i * steps, 0]
            newX[i] = X[i * steps:steps * (i + 1), 1]
        start_i = 23
        encoding_h_str = 'h_'
        col_names = [f'{encoding_h_str}{x}' for x in listHours[start_i:]]
        prevSellsDf = pd.DataFrame(newX[:, start_i:], columns=col_names)
        prevSellsDf[DATE_COL] = dt
        prevSellsDf[DATE_COL] = prevSellsDf[DATE_COL].astype('datetime64[ns]')

    else:
        raise ValueError(f"mode {mode} not recognized !")

    return prevSellsDf.dropna(), col_names


def get_all_data(hourly=False):
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
    # prevSellsPath = dataVentePath
    # # Fichier des prévisions de ventes
    # prevSellsDf = pd.read_csv(prevSellsPath, sep=';')
    # # Convertie la colonne date en datetime
    # prevSellsDf['date'] = pd.to_datetime(prevSellsDf['date'], format='%d-%m-%Y')
    prevSellsDf, col_names = getPrevSells(hourly)

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
    # df = pd.merge(pd.merge(prevSellsDf, meteoDf, on='date'), attendanceDf, on='date').dropna()
    df = mergeDfList([prevSellsDf, meteoDf, attendanceDf])
    # Réinitialiser les indices
    # df = df.reset_index(drop=True)

    df = addDates(df)
    df = addSportBroadcast(df)
    df = df.dropna()

    X = df.drop(col_names, axis=1)
    Y = df[col_names]
    return X, Y


def mergeDfList(dfList, on=DATE_COL):
    assert dfList, f"dfList {dfList} is not a valid value"
    df = dfList[0]
    for i in range(1, len(dfList)):
        df = pd.merge(df, dfList[i], on=on).dropna()
    # df.set_index('date_column', inplace=True)
    return df


def addDates(df, datetime_col=DATE_COL):
    # Jours de la semaine
    # On ajoute une colonne date
    df['day'] = df[datetime_col].map(date2day)
    # hot encode day
    df = pd.get_dummies(df, columns=['day'])

    # Vacances
    # On ajoute une colonne
    df['vacance'] = df[datetime_col].map(date2vacances)

    # Jours fériés
    # On ajoute une colonne
    df['ferie'] = df[datetime_col].map(date2jourferie)
    return df


def generateSportBroadcast():
    res = []
    csv_files = [
        (mma_path, 'match_mma'),
        (nba_path, 'match_nba'),
        (nfl_path, 'match_nfl'),
        (nhl_path, 'match_nhl')
    ]

    for file_name, column_name in csv_files:
        schedule = pd.read_csv(file_name, sep=',', parse_dates=['date'])
        schedule = schedule.rename(columns={'Match': column_name, 'date': DATE_COL}).fillna(0)
        res.append(schedule)

    print("match data loaded")
    return res


def addSportBroadcast(df, on=DATE_COL):
    for matchDf in generateSportBroadcast():
        df = pd.merge(df, matchDf, on=on, how='left')

    return df.fillna(0)


def get_data_filtered_data(features=ALL_FEATURES, hourly=False):
    X, Y = get_all_data(hourly)
    X = remove_keys_not_in_list(X, features)
    return X, Y


def getAllDataFromCSV(hourly=False):
    df_sales, col_names = getPrevSells(hourly)
    df_meteo = pd.read_csv(meteoPath, parse_dates=['time'], index_col='time').rename_axis('date').drop(
        'Unnamed: 0', axis=1)
    df_predictHQ = pd.read_csv(affluencePath, parse_dates=['date'], index_col='date')
    df_all = pd.merge(pd.merge(df_sales, df_meteo, on='date'), df_predictHQ, on='date').dropna()
    df_all = addDates(df_all)
    df_all = addSportBroadcast(df_all)

    df = df_all.dropna()
    X = df.drop(col_names, axis=1)
    Y = df[col_names]
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
