import os
from dotenv import load_dotenv
from predicthq import Client
from constants import ST_CATH_LOC, ATTENDANCE_BASE_CAT, ALL_VIEWERSHIP_CAT, affluencePath
import pandas as pd
import matplotlib.pyplot as plt


if not os.path.isfile(affluencePath):
    load_dotenv()
    access_token = os.getenv("ACCESS_TOKEN_PREDICT_HQ")

    phq = Client(access_token=access_token)
    start_date = "2023-03-08"
    end_date = "2023-06-05"
    selected_cat=ATTENDANCE_BASE_CAT
    features_args = {
        "active__gte": start_date,
        "active__lte": end_date,
        "location__geo": ST_CATH_LOC.get_location(),
    }
    for cat in selected_cat:
        features_args[cat + "__stats"] = ["sum", "count"]
        features_args[cat + "__phq_rank"] = {"gt": 0}

    feature_list = []
    for feature in phq.features.obtain_features(**features_args):
        feature_dict = {"date": feature.date}
        for cat in selected_cat:
            feature_dict[cat + "_sum"]= getattr(getattr(getattr(feature, cat), 'stats'), 'sum')
            feature_dict[cat + "_count"]= getattr(getattr(getattr(feature, cat), 'stats'), 'count')

        feature_list.append(feature_dict)

    df = pd.DataFrame(feature_list)
    df = df.set_index("date")
    df["phq_attendance_stats_sum"] = df.filter(regex=r'^phq_attendance.*sum$').sum(axis=1)
    df["phq_attendance_stats_count"] = df.filter(regex=r'^phq_attendance.*count').sum(axis=1)
    df.to_csv(affluencePath)
else:
    df = pd.read_csv(affluencePath)

plt.plot(df["phq_attendance_stats_sum"].tolist())
plt.show()
