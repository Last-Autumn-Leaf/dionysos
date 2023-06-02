import xgboost as xgb
import pandas as pd


attendancePath="affluence.csv"
prevSellsPath="../data_vente.csv"
meteoPath="../archive.csv"

attendanceDf=pd.read_csv(attendancePath)[['date', 'phq_attendance_stats_sum']]
prevSellsDf=pd.read_csv(prevSellsPath,sep=';')[['date', 'vente']]
meteoDf=pd.read_csv(meteoPath)


data = {
    'attendance': [],
    'apparent_temperature_mean': [],
    'rain_sum': [],
    'snowfall_sum': [],
    'vente':[]
}

df = pd.DataFrame(data)
X = df[['attendance', 'apparent_temperature_mean', 'rain_sum', 'snowfall_sum']]
y = df['vente']

model = xgb.XGBRegressor(
    n_estimators=100,  # Number of trees (boosting rounds)
    max_depth=3,  # Maximum depth of each tree
    learning_rate=0.1,  # Step size shrinkage
    subsample=0.8,  # Subsample ratio of the training instances
    colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
    objective='reg:squarederror'  # Loss function to be minimized
)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)