import os
from dotenv import load_dotenv
from predicthq import Client
from constants import ST_CATH_LOC

load_dotenv()
H=os.getenv("ACCESS_TOKEN_PREDICT_HQ")
access_token = H

phq = Client(access_token=access_token)
start_date="2023-03-02"
end_date="2023-05-30"
for feature in phq.features.obtain_features(
        active__gte=start_date,
        active__lte=end_date,
        location__geo=ST_CATH_LOC.get_location(),
        phq_attendance_sports__stats=["count", "median"]
):
    print(feature.date, feature.phq_attendance_sports.stats.count)


