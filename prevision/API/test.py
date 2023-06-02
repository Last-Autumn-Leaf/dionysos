import os
import pandas as pd


attendancePath="affluence.csv"
prevSellsPath="../data_vente.csv"

df1=pd.read_csv(attendancePath)[['date', 'phq_attendance_stats_sum']]
df2=pd.read_csv(prevSellsPath)
