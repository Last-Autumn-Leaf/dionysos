from datetime import datetime

from ..utils import setProjectpath
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

ACCESS_TOKEN_PREDICT_HQ = os.getenv('ACCESS_TOKEN_PREDICT_HQ')
ACCESS_TOKEN_VISUAL_CROSSING = os.getenv('ACCESS_TOKEN_VISUAL_CROSSING')
# db variable
mysql_database = os.environ.get('MYSQL_DATABASE')
mysql_user = os.environ.get('MYSQL_USER')
mysql_password = os.environ.get('MYSQL_PASSWORD')
mysql_root_password = os.environ.get('MYSQL_ROOT_PASSWORD')
WeatherDataTable = 'weather_data'

# File path
project_dir = setProjectpath()
dataDir = project_dir / 'prevision' / 'data'
affluencePath = dataDir / "affluence.csv"
dataVentePath = dataDir / "data_vente.csv"
dailySalesPath = dataDir / "daily_sales.csv"
hourlySalesPath = dataDir / "vente_hours.csv"
meteoPath = dataDir / "meteo.csv"
mma_path = dataDir / "mma_schedule_day.csv"
nba_path = dataDir / "nba_schedule_day.csv"
nfl_path = dataDir / "nfl_schedule.csv"
nhl_path = dataDir / "nhl_mtl_schedule_day.csv"
hourly_client_CLS = dataDir / "client_CLS.csv"
daily_sales_CLS = dataDir / "daily_sales_CLS.csv"
hours_sales_CLS = dataDir / "hours_sales_CLS.csv"

# Variables à utiliser pour l'Api de prévision d'attendance PredictHQ
ATTENDANCE_BASE_CAT = [
    "phq_attendance_concerts",
    "phq_attendance_conferences",
    "phq_attendance_expos",
    "phq_attendance_festivals",
    "phq_attendance_performing_arts",
    "phq_attendance_sports"]

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
# Localisation du restaurant
ST_CATH_LOC = {
    "lon": -73.56362829999999,
    "lat": 45.5077116,
    "radius": '0.5km'
}

LA_SALLE = {
    "lon": -73.6123496,
    "lat": 45.4469922,
    "radius": '1mi'
}

MONTREAL_TZ = 'Canada/Eastern'

MTL_ID = {
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

LACAGE_ID = {
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

CANADA_ID = {
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

ALL_FEATURES = ['prevision', 'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
                'vacance', 'ferie', 'vente_day_1', 'vente_day_2', 'vente_day_3', 'vente_day_4',
                'vente_day_5', 'vente_day_6', 'vente_day_7', 'apparent_temperature_max',
                'apparent_temperature_min', 'sunset', 'uv_index_max', 'showers_sum', 'rain_sum',
                'snowfall_sum', 'precipitation_hours', 'attendance_concerts', 'attendance_conferences',
                'attendance_expos', 'attendance_festivals', 'attendance_performing_arts', 'attendance_sports']

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
    '2023-12-26'  # Lendemain de Noël (Boxing Day)
]

vacances = [
    ('2022-12-17', '2023-01-02'),  # Vacances d'hiver
    ('2023-03-04', '2023-03-19'),  # Vacances de mars
    ('2023-06-24', '2023-08-27')  # Vacances d'été
]

MODE_DAILY_SALES = 0
MODE_HOURLY_SALES = 1
MODE_HOURLY_CLIENT = 2
ALL_MODES = [MODE_DAILY_SALES, MODE_HOURLY_SALES, MODE_HOURLY_CLIENT]
modeStr2i = {'daily_sales': MODE_DAILY_SALES, 'hourly_sales': MODE_HOURLY_SALES, 'hourly_client': MODE_HOURLY_CLIENT}
modei2str = {MODE_DAILY_SALES: 'daily_sales', MODE_HOURLY_SALES: 'hourly_sales', MODE_HOURLY_CLIENT: 'hourly_client'}

modei2Path = {MODE_DAILY_SALES: daily_sales_CLS,
              MODE_HOURLY_SALES: hours_sales_CLS,
              MODE_HOURLY_CLIENT: hourly_client_CLS}
isModeValid = lambda x: x in ALL_MODES or x in modeStr2i
castStr2Datetime = lambda x: datetime.strptime(x, '%Y-%m-%d')
