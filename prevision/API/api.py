import os
from dotenv import load_dotenv
from predicthq import Client
from constants import ST_CATH_LOC
import pandas as pd
import matplotlib.pyplot as plt

data_path= "affluence.csv"
if not os.path.isfile(data_path):
    load_dotenv()
    H=os.getenv("ACCESS_TOKEN_PREDICT_HQ")
    access_token = H


    phq = Client(access_token=access_token)
    start_date="2023-03-03"
    end_date="2023-05-31"
    # for feature in phq.features.obtain_features(
    #         active__gte=start_date,
    #         active__lte=end_date,
    #         location__geo=ST_CATH_LOC.get_location(),
    #         phq_attendance_sports__stats=["count", "median"]
    # ):
    #     print(feature.date, feature.phq_attendance_sports.stats.count)

    feature_list=[]
    for feature in phq.features.obtain_features(
            active__gte=start_date,
            active__lte=end_date,
            location__geo=ST_CATH_LOC.get_location(),
            phq_rank_public_holidays=True,
            phq_attendance_sports__stats=["sum", "count"],
            phq_attendance_sports__phq_rank={"gt": 0},
            phq_attendance_conferences__stats=["sum", "count"],
            phq_attendance_conferences__phq_rank={"gt": 0},
            phq_attendance_expos__stats=["sum", "count"],
            phq_attendance_expos__phq_rank={"gt": 0},
            phq_attendance_concerts__stats=["sum", "count"],
            phq_attendance_concerts__phq_rank={"gt": 0},
            phq_attendance_festivals__stats=["sum", "count"],
            phq_attendance_festivals__phq_rank={"gt": 0},
            phq_attendance_performing_arts__stats=["sum", "count"],
            phq_attendance_performing_arts__phq_rank={"gt": 0},
            phq_attendance_community__stats=["sum", "count"],
            phq_attendance_community__phq_rank={"gt": 0},
        ):
            feature_dict = {
                "date": feature.date,
                "phq_attendance_sports_sum": feature.phq_attendance_sports.stats.sum,
                "phq_attendance_sports_count": feature.phq_attendance_sports.stats.count,
                "phq_attendance_conferences_sum": feature.phq_attendance_conferences.stats.sum,
                "phq_attendance_conferences_count": feature.phq_attendance_conferences.stats.count,
                "phq_attendance_expos_sum": feature.phq_attendance_expos.stats.sum,
                "phq_attendance_expos_count": feature.phq_attendance_expos.stats.count,
                "phq_attendance_concerts_sum": feature.phq_attendance_concerts.stats.sum,
                "phq_attendance_concerts_count": feature.phq_attendance_concerts.stats.count,
                "phq_attendance_festivals_sum": feature.phq_attendance_festivals.stats.sum,
                "phq_attendance_festivals_count": feature.phq_attendance_festivals.stats.count,
                "phq_attendance_performing_arts_sum": feature.phq_attendance_performing_arts.stats.sum,
                "phq_attendance_performing_arts_count": feature.phq_attendance_performing_arts.stats.count,
                "phq_attendance_community_sum": feature.phq_attendance_community.stats.sum,
                "phq_attendance_community_count": feature.phq_attendance_community.stats.count,
            }
            feature_list.append(feature_dict)

    df = pd.DataFrame(feature_list)
    df = df.set_index("date")
    df["phq_attendance_stats_sum"] = df.filter(regex="sum").sum(axis=1)
    df["phq_attendance_stats_count"] = df.filter(regex="count").sum(axis=1)
    df.to_csv(data_path)
else :
    df=pd.read_csv(data_path)

plt.plot(df["phq_attendance_stats_sum"].tolist())
plt.show()