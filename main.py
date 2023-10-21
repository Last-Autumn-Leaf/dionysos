import pandas as pd

from prevision import Options, Model, DataLoader, XGBOOST_TYPE, getAllDataFromCSV, RNN_TYPE, hourlySalesPath, \
    dailySalesPath, getPrevSells, Api_VC, modei2Path, getDataFromMode, DataTable, api_predicthq, Api_PHQ, \
    generateSportBroadcast, mergeDfList, addDates, addSportBroadcast, DATE_COL, getDatesBetween, DATE_FORMAT
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def createXY(sellsDF, meteoDF, attendanceDf, outputColNames):
    df = mergeDfList([sellsDF, meteoDF, attendanceDf])
    df = addDates(df)
    df = addSportBroadcast(df)

    df.sort_values(by=DATE_COL, ascending=True, inplace=True)
    datetimeCol = [x.strftime("%Y-%m-%d") for x in df[DATE_COL].tolist()]
    object_columns = df.select_dtypes(include=['object'])  # remove object col
    X = df.drop(outputColNames, axis=1).drop(object_columns, axis=1)
    Y = df[outputColNames]
    return X, Y, datetimeCol


def main(mode, dataPath, localisation):
    vc = Api_VC()
    predicthq = Api_PHQ()

    sellsDF, outputColNames = getDataFromMode(mode, dataPath)
    min_date = sellsDF[DATE_COL].min().strftime('%Y-%m-%d')
    max_date = sellsDF[DATE_COL].max().strftime('%Y-%m-%d')
    # min_date = '2023-07-01'
    # max_date = '2023-09-01'
    print("From", min_date, " to ", max_date)
    min_date = predicthq.phqCapStartDate(min_date)

    meteoDF = vc.getMeteoData(localisation, min_date, max_date)
    attendanceDf = predicthq.getAttendanceData(localisation, min_date, max_date, vc.resolvedAddress)
    X, Y, TrainingDateTimeCol = createXY(sellsDF, meteoDF, attendanceDf, outputColNames)

    # Setting up training and fine Tuning !

    TrainingData = (X, Y)
    recursif = True
    hourly = mode
    input_size = len(X.columns)
    output_size = len(Y.columns)
    options = Options(model_type=XGBOOST_TYPE, input_sequence_length=14,
                      output_sequence_length=1 if recursif or hourly else 7, verbose_mod=100, input_size=input_size
                      , output_size=output_size if hourly else 1, recursif=recursif, verbose=0, hourly=hourly)

    modelInstance = Model(options)
    dataLoaderTraining = DataLoader(options=options, customData=TrainingData)
    modelInstance.train(dataLoaderTraining, test=True, plot=True)
    # modelInstance.fineTuneXGBoostRay(dataLoaderTraining, None, 100)

    # -------- Deploying the model

    # getting the lasts recorded day we need at least input_sequence_length + 1 days
    daysToPredict = 14
    offset = datetime.strptime(max_date, DATE_FORMAT) - timedelta(days=options.input_sequence_length - 1)
    sellsDFDeploy = sellsDF[offset <= sellsDF[DATE_COL]]
    min_date = datetime.strptime(max_date, DATE_FORMAT) + timedelta(days=1)
    max_date = datetime.strptime(max_date, DATE_FORMAT) + timedelta(days=daysToPredict)

    offset = offset.strftime(DATE_FORMAT)
    min_date = min_date.strftime(DATE_FORMAT)
    max_date = max_date.strftime(DATE_FORMAT)

    series_to_add = getDatesBetween(min_date, max_date)
    series_to_add = {k: (series_to_add if k == DATE_COL else [0] * len(series_to_add)) for k in sellsDFDeploy.columns}
    sellsDFDeploy = pd.concat([sellsDFDeploy, pd.DataFrame(series_to_add)], ignore_index=True)
    sellsDFDeploy[DATE_COL] = pd.to_datetime(sellsDFDeploy[DATE_COL])
    meteoDFDeploy = vc.getMeteoData(localisation, offset, max_date)
    attendanceDfDeploy = predicthq.getAttendanceData(localisation, offset, max_date, vc.resolvedAddress)
    XDeploy, YDeploy, DeployDateTimeCol = createXY(sellsDFDeploy, meteoDFDeploy, attendanceDfDeploy, outputColNames)
    deployData = (XDeploy, YDeploy)

    options.fullTraining = True
    dataLoaderDeploy = DataLoader(options=options, customData=deployData)

    res = modelInstance.deploy(dataLoaderDeploy)
    plt.plot(res['predicted_sequence'])
    plt.xticks([x for x in range(daysToPredict)], getDatesBetween(min_date, max_date), rotation=90)
    plt.title('Predicted sequence')
    plt.show()
    modelInstance.featureImportance(dataLoaderDeploy.getFeatureNames())

    return X, Y, outputColNames


if __name__ == '__main__':
    mode = 1

    localisation = "7077 Bd Newman, LaSalle, QC H8N 1N1"
    dataPath = modei2Path[mode]
    main(mode, dataPath, localisation)

    # recursif = True
    # options = Options(model_type=XGBOOST_TYPE, input_sequence_length=14,
    #                   output_sequence_length=1 if recursif or hourly else 7, verbose_mod=100, input_size=27
    #                   , output_size=25 if hourly else 1, verbose=0, recursif=recursif, hourly=hourly)
    #
    # a = Model(options)
    # b = DataLoader(options)
    # a.train(b, 1, 1)
    # res = a.deploy(b)
    # print(res)
    # a.fineTuneXGBoostRay(b, None, 100)
    # a.featureImportance(b.getFeatureNames())
