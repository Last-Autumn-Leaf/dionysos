import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()
access_token = os.getenv("ACCESS_TOKEN_PREDICT_HQ")

response = requests.get(
    url="https://api.predicthq.com/v1/places/",
    headers={
      "Authorization": f"Bearer {access_token}",
      "Accept": "application/json"
    },
    params={
        "q": "Canada",
        "limit": 5
    }
)

# print(response.json())
print(json.dumps(response.json(), indent=4))