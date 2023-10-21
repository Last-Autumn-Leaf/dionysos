import pandas as pd

from .utils import dataDir
from .daily_sales import DailySales
from .hourly_sales import HourlySales
from .hourly_affluence import HourlyAffluence
import sys

hs = HourlySales()
ds = DailySales()
ha = HourlyAffluence()


def find_MD_file_type(filepath: str) -> str:
    """
    Determines the type of the given CSV file.
    
    Args:
    - filepath (str): Path to the CSV file.
    
    Returns:
    - str: The type of the CSV file.
    """
    # check if filepath is a folder or a file
    if filepath[-1] == '/':
        # open a file in the folder
        filepath = hs.list_files(filepath)[0]
    # Read the first row of the dataframe
    df = pd.read_csv(filepath, encoding="ISO-8859-1", nrows=1)
    parts = df.iloc[0, 0:5].tolist()
    if 'Ventes permanentes journalieres' in parts[0]:
        MD_file_type = 'daily_sales'
    elif 'clients' in parts[1]:
        MD_file_type = 'client'
    elif "ventes à l'heure" in parts[1]:
        MD_file_type = 'hours_sales'
    return MD_file_type


def get_type_csv_fromMD(filepath, outputSuffix):
    MD_file_type = find_MD_file_type(filepath)
    if MD_file_type == 'daily_sales':
        df = ds.process_daily_sales(filepath)
    else:
        # List all files in the provided directory
        files = hs.list_files(filepath)
        df = pd.DataFrame()
        for file in files:
            if MD_file_type == 'hours_sales':
                one_file = hs.process_hours_sales_data(file)
            elif MD_file_type == 'client':
                one_file = ha.process_affluence_data(file)
            # Merge with all_hourly, ensuring there are no duplicate dates, and sort by date
            df = pd.concat([one_file, df]).drop_duplicates(subset=['date_time']).sort_values(by=['date_time'])

    # Save the consolidated dataframe to a CSV file
    output_path = dataDir / '{}_{}.csv'.format(MD_file_type, outputSuffix)
    print("file created", output_path)
    df.to_csv(output_path, index=False)

    return MD_file_type, df, output_path


# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Execution logic for batch processing of CSV files and output generation
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) == 3:
        # si c'est un fichier mettre / a la fin
        filepath = str(sys.argv[1])
        file_name = str(sys.argv[2])

        get_type_csv_fromMD(filepath, file_name)
        print("hello")
