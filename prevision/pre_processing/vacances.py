import datetime
import pandas as pd

date_ranges = [
    ("22 mars 2020", "19 juin 2020"),
    ("25 décembre 2020", "28 février 2022"),
    ("20 DÉCEMBRE 2021", "12 février 2022")
]

# Define the start and end dates
start_date = datetime.date(2016, 1, 1)
end_date = datetime.date.today()

# Generate a list of dates
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Create a DataFrame with dates as index and filled with zeros
data = {'Value': [0] * len(date_range)}
df = pd.DataFrame(data, index=date_range)

# Print the DataFrame
print(df)
