#####################################################################
#                                                                   #
# This Python script processes sales data from CSV files.           #
# It extracts relevant sales information, cleans it, and structures #
# the data into a pandas DataFrame for further analysis.            #
#                                                                   #
# The main functions included are:                                  #
# 1. find_start_str - Extracts the starting sequence from the CSV.  #
# 2. parse_line_vj - Parses a single line from the CSV.             #
# 3. clean_vente_journaliere - Cleans and processes the entire CSV. #
#                                                                   #
#####################################################################

# Librairies
# --------------------
import sys
import pandas as pd


class DailySales:
    # Fonctions:
    # --------------------
    def find_start_str(self, filepath: str) -> str:
        """
        Extracts the starting string sequence from the given CSV file.

        Args:
        - filepath (str): Path to the CSV file.

        Returns:
        - str: The starting string sequence.
        """
        # Read the first row of the dataframe
        df = pd.read_csv(filepath, encoding="ISO-8859-1", nrows=1)
        parts = df.iloc[0, 1:4].tolist()

        # Extract only the part after '\n' for the first part
        parts[0] = parts[0].split('\n')[-1]

        # Ensure the second part (number) has a "00" prefix
        parts[1] = f'00{parts[1]}'

        # Convert the entire list to a comma-separated string with quotes
        start_str = ','.join(['"' + str(part) + '"' for part in parts])

        return start_str[1:]

    def parse_line_vj(self, line: str, n_cols: int) -> list:
        """
        Parses a line from the CSV file into a list of values.

        Args:
        - line (str): The line from the CSV file.
        - n_cols (int): The expected number of columns.

        Returns:
        - list: Parsed values from the line.
        """
        parts = line.split(',')[:-2]
        parsed_parts = []

        for part in parts:
            # If the last parsed part and the current part are both numeric,
            # they are probably part of the same value, so merge them.
            if parsed_parts and part.replace(' ', '').replace('"', '').isdigit() and len(
                    part.replace(' ', '').replace('"', '')) == 2 and ',' not in parsed_parts[-1]:
                parsed_parts[-1] = f'{parsed_parts[-1]},{part}'
            else:
                parsed_parts.append(part)

        # If the line has more parts than expected, merge the last few parts
        while len(parsed_parts) > n_cols:
            parsed_parts[-2] = f'{parsed_parts[-2]},{parsed_parts[-1]}'
            parsed_parts = parsed_parts[:-1]

        return parsed_parts

    def process_daily_sales(self, filepath: str) -> pd.DataFrame:
        """
        Cleans and processes the given CSV file.

        Args:
        - filepath (str): Path to the CSV file.

        Returns:
        - pd.DataFrame: Processed dataframe with date and sales columns.
        """
        # Extract the starting string sequence from the file
        start_str = self.find_start_str(filepath)

        # Read the file with ISO-8859-1 encoding
        with open(filepath, 'r', encoding='ISO-8859-1') as file:
            lines = file.readlines()

        # Filter out lines starting with specific header
        filtered_lines = [line for line in lines if not line.startswith('"Ventes permanentes journalieres')]

        # Clean the lines
        clean_lines = []
        debut_ligne = start_str + ',"Ventes","Escomptes","Taxes","Annulees","Entrainement","Date","Total","Journalier","Total","Journalier","Total","Journalier","Total","Journalier","Total","Journalier",'
        for line in filtered_lines:
            if line.startswith('à'):
                clean_lines.append(line.replace(debut_ligne, '', 1))
            else:
                clean_lines.append(line)

        # Define column names
        column_names = ["Date", "Total_1", "Journalier_1", "Total_2", "Journalier_2", "Total_3", "Journalier_3",
                        "Total_4", "Journalier_4", "Total_5", "Journalier_5"]

        # Parse each line
        parsed_data = [self.parse_line_vj(line, len(column_names)) for line in clean_lines]

        # Convert parsed data into a dataframe
        df = pd.DataFrame(parsed_data, columns=column_names)

        # Convert date strings to datetime format
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='"%YM%mM%d"')
        except:
            df['Date'] = pd.to_datetime(df['Date'], format='"%Y-%m-%d"')

        # Replace commas with dots in all columns that contain numbers
        cols_to_update = ["Total_1", "Journalier_1", "Total_2", "Journalier_2", "Total_3", "Journalier_3", "Total_4",
                          "Journalier_4", "Total_5", "Journalier_5"]
        for col in cols_to_update:
            df[col] = df[col].apply(lambda x: x.replace(',', '.'))

        # Retain only the Date and Journalier_1 columns, and rename Journalier_1 to vente
        df = df[['Date', 'Journalier_1']]
        df.rename(columns={'Journalier_1': 'vente'}, inplace=True)

        df['vente'] = df['vente'].str.replace('\xa0', '').astype(float)

        return df


if __name__ == "__main__":
    if len(sys.argv) == 3:
        filepath = str(sys.argv[1])
        file_name = str(sys.argv[2])
    df = DailySales.process_daily_sales(filepath, )
    df.to_csv('daily_sales_{}.csv'.format(file_name), index=False)
