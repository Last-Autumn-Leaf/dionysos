import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = "vwPFA-mbl2iAArouiAYW1m7qJNsa0Z25gUQoOF-yyGIT0O1iCebC74MWSl2-ZY6ciWz4pC3dtQX90XtvMUHnlQ=="
org = "dionysas"
url = "http://localhost:8086"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
bucket = "data"

write_api = client.write_api(write_options=SYNCHRONOUS)

for value in range(5):
    point = (
        Point("measurement1")
        .tag("tagname1", "tagvalue1")
        .field("field1", value)
    )
    write_api.write(bucket=bucket, org="dionysas", record=point)
    time.sleep(1)  # separate points by 1 second

query_api = client.query_api()

query = """from(bucket: "data")
  |> range(start: -10m)
  |> filter(fn: (r) => r._measurement == "measurement1")
  |> mean()"""
tables = query_api.query(query, org="dionysas")

for table in tables:
    for record in table.records:
        print(record)
