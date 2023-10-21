"""
Hourly Affluence Report Processing Script

Purpose:
    This script processes hourly affluence data from CSV files to provide insights 
    into customer footfall or visit patterns on an hourly basis. The extracted and 
    processed data allows for better understanding and prediction of high and low 
    traffic hours, which can be instrumental for businesses in making informed 
    operational decisions.

    Key functionalities of the script include:
    1. Detecting specific markers or sequences in the CSV to aid in data extraction.
    2. Parsing and cleaning the data to handle any specific formatting or inconsistencies.
    3. Consolidating data from multiple files and outputting a structured dataset.
    4. Listing all relevant CSV files in a specific directory for batch processing.

Usage:
    The script can be executed with specific command line arguments to define input 
    directories and desired output filenames. Upon execution, all relevant CSV files 
    in the specified directory will be processed, and the results will be consolidated 
    into a single output CSV file.

Author: Ilyas Baktache
Date: 17-11-2023
"""

# Librairies
# -----------------------------

import sys
import pandas as pd
import os
from datetime import datetime, timedelta

# Set locale to French for date processing
import locale

locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')


# Fonctions
# -----------------------------

class HourlyAffluence:
    def find_start_str(self, filepath: str) -> str:
        """
        Identifies and extracts the starting sequence from the provided CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            str: Detected starting sequence.
        """
        # Read the first row of the dataframe
        df = pd.read_csv(filepath, encoding="ISO-8859-1", nrows=1)
        parts = df.iloc[0, 1:5].tolist()
        parts = parts[1:]
        # Extract only the part after '\n' for the first part
        parts[0] = parts[0].split('\n')[-1]
        # Ensure the second part (number) has a "00" prefix
        parts[1] = f'00{parts[1]}'
        # Convert the entire list to a comma-separated string with quotes
        start_str = ','.join(['"' + str(part) + '"' for part in parts])
        return start_str[1:]

    def parse_line(self, line: str, n_cols: int) -> list:
        """
        Parses a single line from the CSV file into a list of values, handling potential
        data splits or inconsistencies.

        Args:
            line (str): A single line from the CSV file.
            n_cols (int): Expected number of columns in the parsed output.

        Returns:
            list: Parsed values from the line.
        """
        parts = line.split(',')[:-2]
        # If the line has more parts than expected, merge the last few parts
        while len(parts) > n_cols:
            parts[-2] = f'{parts[-2]},{parts[-1]}'
            parts = parts[:-1]
        return parts

    def process_affluence_data(self, filepath: str) -> pd.DataFrame:
        """
        Processes the provided CSV file to extract and structure hourly affluence data.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Processed and structured hourly affluence data.
        """
        # Find the starting string sequence
        start_str = self.find_start_str(filepath)

        # Read the file with ISO-8859-1 encoding
        with open(filepath, 'r', encoding='ISO-8859-1') as fichier:
            lines = fichier.readlines()

        start_date_str = lines[0].split(',')[0].split(' -')[0]
        start_date = datetime.strptime(start_date_str[1:], "%Y-%m-%d")

        # Filter the lines that start with the starting string sequence
        filtered_lines = [line for line in lines if line.startswith(start_str)]

        # Clean the lines
        clean_lines = []
        days = [(start_date + timedelta(days=i)).strftime('%d') for i in range(7)]
        for line in filtered_lines:
            only_data = line.replace(start_str + ',', '', 1)
            only_data = only_data.replace(
                f'"De     - a","Dim - {days[0]}","Lun - {days[1]}","Mar - {days[2]}","Mer - {days[3]}","Jeu - {days[4]}","Ven - {days[5]}","Sam - {days[6]}","Total",',
                '', 1)
            clean_lines.append(only_data)

        # Define column names
        column_names = ["sources", "intervale temps", "Dim", "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Total", "val",
                        "periode"]

        # Parse each line
        parsed_data = [self.parse_line(line, len(column_names)) for line in clean_lines]

        # Convert parsed data into a dataframe
        df = pd.DataFrame(parsed_data, columns=column_names)
        df['periode'] = df['periode'].apply(lambda x: x.split(',')[0])
        # Replace commas with dots in all columns that contain numbers
        cols_to_update = ["Dim", "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Total"]

        for col in cols_to_update:
            df[col] = df[col].apply(lambda x: x.replace(',', '.'))

        # Adjust the parse_interval function to use the end time of the interval
        def parse_interval(interval):
            start_time_str, end_time_str = interval.replace('"', '').split(' -')
            end_time = datetime.strptime(end_time_str, " %H:%M")
            end_timedelta = timedelta(hours=end_time.hour, minutes=end_time.minute)
            return end_timedelta

        # Apply the function to the 'intervale temps' column
        df['timedelta'] = df['intervale temps'].apply(parse_interval)

        # Add the timedelta to the start date to get the actual date and time
        df['date_time'] = df['timedelta'].apply(lambda x: start_date + x)

        # Map each day of the week to a timedelta representing the number of days since the start date
        day_to_timedelta = {
            'Dim': timedelta(days=0),
            'Lun': timedelta(days=1),
            'Mar': timedelta(days=2),
            'Mer': timedelta(days=3),
            'Jeu': timedelta(days=4),
            'Ven': timedelta(days=5),
            'Sam': timedelta(days=6),
        }

        # Melt the dataframe to have a row for each day of the week
        melted_df = pd.melt(df, id_vars=['sources', 'date_time', 'periode'],
                            value_vars=['Dim', 'Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam'], var_name='day',
                            value_name='vente')

        # Add the timedelta for the day of the week to the date_time
        melted_df['date_time'] = melted_df.apply(lambda row: row['date_time'] + day_to_timedelta[row['day']], axis=1)

        # Convert 'vente' to numeric
        melted_df['vente'] = pd.to_numeric(melted_df['vente'])

        # enleve les guillements de la colonne sources et periode
        melted_df['sources'] = melted_df['sources'].str.replace('"', '')
        melted_df['periode'] = melted_df['periode'].str.replace('"', '')

        return melted_df

    def list_files(self, directory: str) -> list:
        """
        Lists all CSV files present in the specified directory.

        Args:
            directory (str): Path to the directory.

        Returns:
            list: List of CSV file paths in the directory.
        """
        files = []
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                files.append(os.path.join(directory, file))
        return files


# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Execution logic for batch processing of CSV files and output generation
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) == 3:
        filepath = str(sys.argv[1])
        file_name = str(sys.argv[2])

        # List all files in the provided directory
        files = HourlyAffluence.list_files(filepath)
        all_sells = pd.DataFrame()
        for file in files:
            print("File : ", file)
            df = HourlyAffluence.process_affluence_data(file)
            # Merge with all_sells, ensuring there are no duplicate dates, and sort by date
            all_sells = pd.concat([all_sells, df]).drop_duplicates(subset=['date_time']).sort_values(by=['date_time'])

        # Save the consolidated dataframe to a CSV file
        all_sells.to_csv('affluence_hours_{}.csv'.format(file_name), index=False)
