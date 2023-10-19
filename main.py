from prevision import Options, Model, DataLoader, XGBOOST_TYPE, getAllDataFromCSV, RNN_TYPE, hourlySalesPath, \
    dailySalesPath, getPrevSells, visual_Crossing, modei2Path, getDataFromMode, DataTable
import numpy as np


def main(mode, dataPath, localisation):
    sellsDF, col_names = getDataFromMode(mode, dataPath)

    min_date = sellsDF.index.min().strftime('%Y-%m-%d')
    max_date = sellsDF.index.max().strftime('%Y-%m-%d')
    # min_date = '2023-10-14'
    # max_date = '2023-11-01'
    print("From", min_date, " to ", max_date)
    vc = visual_Crossing()
    # vc.getNext2Weeks(localisation)
    print(vc.getMeteoDate(localisation, min_date, max_date))


if __name__ == '__main__':
    mode = 0

    localisation = "7077 Bd Newman, LaSalle, QC H8N 1N1"
    dataPath = modei2Path[mode]
    main(mode, dataPath, localisation)
    # print(dt.getMeteoDate())
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
