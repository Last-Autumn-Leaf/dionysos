"""
Vente Hours Report Processing Script

Purpose:
    This script is designed to process sales report files, extracting and structuring 
    hourly sales data from a series of CSV reports. It's tailored to handle specific 
    formatting challenges in the report, such as split values, and to consolidate the 
    extracted data from multiple files into a single structured output.

    The main functionalities include:
    1. Detecting the start sequence from the CSV to aid in data extraction.
    2. Parsing each line of the report, considering potential data splits.
    3. Cleaning and transforming the report data into a structured DataFrame.
    4. Listing all relevant CSV files in a specified directory.
    5. Consolidating data from all listed files and saving it as a single CSV.

Usage:
    Run this script with the directory containing the report files as the first 
    command line argument and the desired name for the output CSV file as the second 
    command line argument. The script will then process all CSV files in the directory 
    and output the structured data in a single CSV.

Author: Ilyas Baktache
Date: 17-10-2023
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


# Fonction 
# -----------------------------
class HourlySales:

    def find_start_str(self, filepath: str) -> str:
        """
        Extracts the starting string sequence from the given CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            str: The starting string sequence.
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
        parsed_parts = []
        for part in parts:
            # Remove any spaces after a negative sign
            part = part.replace('- ', '-')
            # If the last parsed part and the current part are both numeric,
            # they are probably part of the same value, so merge them.
            # Except if the current part is negative, in which case it's the start of a new number.
            if parsed_parts and part.replace('-', '').replace(' ', '').replace('"', '').isdigit() and len(
                    part.replace('-', '').replace(' ', '').replace('"', '')) == 2 and ',' not in parsed_parts[-1] and \
                    parsed_parts[-1].replace('-', '').replace(' ', '').replace('"',
                                                                               '').isdigit() and not part.startswith(
                '-'):
                parsed_parts[-1] = f'{parsed_parts[-1]},{part}'
            else:
                parsed_parts.append(part)

        # If the line has more parts than expected, merge the last few parts
        while len(parsed_parts) > n_cols:
            parsed_parts[-2] = f'{parsed_parts[-2]},{parsed_parts[-1]}'
            parsed_parts = parsed_parts[:-1]

        return parsed_parts

    def process_hours_sales_data(self, filepath: str) -> pd.DataFrame:
        """
        Process the report file to extract sales data, and structure it in a clean dataframe format.

        Args:
            filepath (str): Path to the report file.

        Returns:
            pd.DataFrame: Cleaned sales data.
        """

        # Find the starting string sequence
        start_str = self.find_start_str(filepath)

        # Read the file with ISO-8859-1 encoding
        with open(filepath, 'r', encoding='ISO-8859-1') as fichier:
            lines = fichier.readlines()
        # The start date is on the second line, after "dimanche "
        start_date_str = lines[1].split("dimanche ")[1].strip()

        # The date is in the format "d MMMM yyyy  HH:mm", where the month is in French
        start_date = datetime.strptime(start_date_str, "%d %B %Y  %H:%M")
        start_date = start_date.replace(hour=0, minute=0)

        # Filter out lines that don't start with ","0094","LA CAGE 94"
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
        column_names = ["sources", "intervale temps", "Dim", "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Total",
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
            end_time = datetime.strptime(end_time_str, "%H:%M")
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
        Lists all the CSV files in a specified directory.

        Args:
            directory (str): Path to the directory.

        Returns:
            list: List of file paths.
        """
        files = []
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                files.append(os.path.join(directory, file))
        return files


# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) == 3:
        filepath = str(sys.argv[1])
        file_name = str(sys.argv[2])

        # List all files in the provided directory
        files = HourlySales.list_files(filepath)
        all_sells = pd.DataFrame()
        for file in files:
            print("File : ", file)
            df = HourlySales.process_hours_sales_data(file)
            # Merge with all_sells, ensuring there are no duplicate dates, and sort by date
            all_sells = pd.concat([all_sells, df]).drop_duplicates(subset=['date_time']).sort_values(by=['date_time'])

        # Save the consolidated dataframe to a CSV file
        all_sells.to_csv('vente_hours_{}.csv'.format(file_name), index=False)
