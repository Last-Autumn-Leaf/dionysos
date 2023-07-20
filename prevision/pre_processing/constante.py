'''
Fichier de constante pour le projet de prévision de vente

Date: 2023-06-06
'''

# Localisation du restaurant
ST_CATH_LOC = {
            "lon": -73.56362829999999,
            "lat": 45.5077116,
            "radius": '0.5km'
            }

MONTREAL_TZ='Canada/Eastern'

from pathlib import Path
project_name="dionysos"
def setProjectpath():
    project_dir = Path.cwd()
    while project_dir.name != project_name:
        project_dir = project_dir.parent
        if project_dir.parent==project_dir:
            ValueError(f"Le dossier parent '{project_name}' n'a pas été trouvé.")
            print("project directory not found")
            break
    return project_dir


#File path
project_dir=setProjectpath()
dataDir =  project_dir / 'prevision' / 'pre_processing' / 'data'
affluencePath = dataDir / "affluence.csv"
dataVentePath = dataDir / "data_vente.csv"
meteoPath = dataDir / "meteo.csv"

# Variables à utiliser pour l'Api de prévision d'attendance PredictHQ
ATTENDANCE_BASE_CAT = [
    "phq_attendance_concerts",
    "phq_attendance_conferences",
    "phq_attendance_expos",
    "phq_attendance_festivals",
    "phq_attendance_performing_arts",
    "phq_attendance_sports" ]

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
        }

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

ALL_FEATURES=['prevision','day_0','day_1','day_2','day_3','day_4','day_5','day_6',
              'vacance','ferie','vente_day_1','vente_day_2','vente_day_3','vente_day_4',
              'vente_day_5','vente_day_6','vente_day_7','apparent_temperature_max',
              'apparent_temperature_min','sunset','uv_index_max','showers_sum','rain_sum',
              'snowfall_sum','precipitation_hours','attendance_concerts','attendance_conferences',
              'attendance_expos','attendance_festivals','attendance_performing_arts','attendance_sports']

# Model types (RNN cells types)
GRU = 'GRU'
LSTM = 'LSTM'
SimpleRNN = 'SimpleRNN'

# criterion
MSE='MSE'
MAE='MAE'
SMOOTH='SMOOHL1'

RNN_PARAMS_LSTM={
        "input_sequence_length": 21,
        "output_sequence_length": 7,
        "model_type" : LSTM,
        "hidden_size": 32,
        "num_layers": 4,
        "output_size": 1,
        "batch_size": 32,
        "epochs": 700,
        "learning_rate": 0.1,
        'dataset_split': 0.7,
        'optimizer':'SGD',
        'momentum' :0.9,
        'weight_decay':1e-5,
        'lossFunction':SMOOTH,
        'verbose_mod':100
    }