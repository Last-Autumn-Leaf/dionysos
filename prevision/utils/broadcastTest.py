import os
from dotenv import load_dotenv
from predicthq import Client

from prevision.api.constants import MONTREAL_TZ,

load_dotenv()
access_token = os.getenv("ACCESS_TOKEN_PREDICT_HQ")
phq = Client(access_token=access_token)

start_date = "2023-03-08"
end_date = "2023-06-05"
place_id=[] # <- insert LaCage ID
for b in phq.broadcasts.search(start__gte=start_date, start__lte=end_date, start__tz=MONTREAL_TZ,
                               broadcast_status=['scheduled', 'predicted'],location__place_id=place_id):
    print(b.event.event_id,b.event.title, b.event.category, b.broadcast_status, b.phq_viewership, b.dates.start_local.strftime('%Y-%m-%d'))

