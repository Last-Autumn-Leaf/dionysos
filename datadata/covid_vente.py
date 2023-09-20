import pandas as pd

# Charger les données de vente depuis le fichier CSV
vente_data = pd.read_csv("/mnt/data/vente_hours.csv")

# Conversion de la colonne 'date_time' en datetime
vente_data['date_time'] = pd.to_datetime(vente_data['date_time'])

# Création d'un dataframe pour les restrictions COVID-19 à Montréal
restrictions_data_montreal = {
    "start_date": [
        "2020-03-22", "2020-12-25", "2021-12-20", "2021-12-20", "2022-01-31", "2021-01-09", "2021-12-31"
    ],
    "end_date": [
        "2020-06-22", "2021-02-08", "2022-02-27", "2022-01-30", "2022-02-11", "2021-05-28", "2022-01-17"
    ],
    "restriction": [
        "Fermeture", "Fermeture", "50% capacité",
        "50% capacité, 4 personnes de 2 bulles familiales",
        "10 personnes de 3 adresses à la même table",
        "Couvre-feu 20h-5h", "Couvre-feu 22h-5h"
    ],
    "start_hour": [
        None, None, None, None, None, 20, 22
    ],
    "end_hour": [
        None, None, None, None, None, 5, 5
    ]
}

# Convertir les chaînes de caractères en dates
restrictions_df_montreal = pd.DataFrame(restrictions_data_montreal)
restrictions_df_montreal['start_date'] = pd.to_datetime(restrictions_df_montreal['start_date'])
restrictions_df_montreal['end_date'] = pd.to_datetime(restrictions_df_montreal['end_date'])

# Création d'une nouvelle colonne pour les ventes ajustées, initialement égale aux ventes réelles
vente_data['vente_ajustee'] = vente_data['vente']

# Appliquer les facteurs d'ajustement en fonction des restrictions
for index, row in restrictions_df_montreal.iterrows():
    mask = (vente_data['date_time'].dt.date >= row['start_date'].date()) & (
            vente_data['date_time'].dt.date <= row['end_date'].date())

    # Pour les couvre-feux, ajuster uniquement les ventes pendant les heures spécifiées
    if "Couvre-feu" in row['restriction']:
        mask &= (vente_data['date_time'].dt.hour >= row['start_hour']) | (
                vente_data['date_time'].dt.hour < row['end_hour'])
        vente_data.loc[mask, 'vente_ajustee'] = 0.0
    elif "Fermeture" in row['restriction']:
        vente_data.loc[mask, 'vente_ajustee'] = 0.0
    elif "50%" in row['restriction']:
        vente_data.loc[mask, 'vente_ajustee'] *= 2
    elif "10 personnes" in row['restriction']:
        vente_data.loc[mask, 'vente_ajustee'] *= 4 / 3

# Sauvegarder les données ajustées
vente_data.to_csv("vente_hours_adjusted.csv", index=False)
