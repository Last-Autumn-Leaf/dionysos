
import pandas as pd


attendancePath="affluence.csv"
prevSellsPath="../data_vente.csv"
meteoPath="../archive.csv"

attendanceDf=pd.read_csv(attendancePath)[['date', 'phq_attendance_stats_sum']]
prevSellsDf=pd.read_csv(prevSellsPath,sep=';')[['date', 'vente']]
meteoDf=pd.read_csv(meteoPath)

