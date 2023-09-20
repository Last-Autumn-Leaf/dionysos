from pathlib import Path
import pandas as pd
from IPython.display import HTML, display


def getDataFiles(folderPath=None, ignore_files=["labor_tache.csv", "vente_plu.csv"]):
    if not folderPath:
        folderPath = Path.cwd()
    dataFiles = [file for file in folderPath.iterdir() if
                 file.is_file() and file.suffix == '.csv' and file.name not in ignore_files]
    print("reading ...")
    for file in dataFiles:
        print("\t", file.name)
    return dataFiles


def importDf(dataFiles):
    # importing to df
    a = {file.stem: pd.read_csv(file) for file in dataFiles}

    # formating
    column_name_mapping = {
        'Date': 'date',
        'Datetime': 'date',
        'date_time': 'date',
        'datetime': 'date'
    }
    for df in a.values():
        df.rename(columns=lambda col: column_name_mapping.get(col, col), inplace=True)
    return a


AFFLUENCE = "affluence_generated"
POICON = "labor_poicon"
LABOR = "vente_jours_labor"
NFL = "nfl_schedule"
VENTE_HOURS = "vente_hours"
NHL_MTL = "nhl_mtl_schedule_day"
NHL = "nhl_schedule_day"
NBA = "nba_schedule_day"
VENTE_DAYS = "vente_jours"
MMA = "mma_schedule_day"
METEO = "meteo"
ALL_FILES = [AFFLUENCE, LABOR, NFL, VENTE_HOURS, NHL_MTL, NHL, NBA, VENTE_DAYS, MMA, METEO]

centered_text = lambda x: f"""
<div style="text-align:center">
    <p style="font-size: 24px; font-weight: bold;">{x}</p>
</div>
"""


def visualizeHead(a, nameTag=AFFLUENCE):
    display(HTML(centered_text(nameTag)))
    df = a[nameTag]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print(f"Date from {df['date'].max()} to {df['date'].min()}")
        display(df.head().style.set_properties(subset=['date'], **{'background-color': 'dodgerblue'}))
    else:
        display(df.head())


def visualizeEveryHead(a):
    for nameTag in ALL_FILES:
        visualizeHead(a, nameTag)
        print()


def initialize():
    dataFiles = getDataFiles()
    DFS = importDf(dataFiles)
    return DFS
