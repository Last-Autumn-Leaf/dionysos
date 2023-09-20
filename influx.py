from dotenv import load_dotenv
from prevision import get_all_data, timeThis
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Load environment variables from .env file
load_dotenv()
TOKEN = os.environ['DOCKER_INFLUXDB_INIT_ADMIN_TOKEN']
ORG = os.environ['DOCKER_INFLUXDB_INIT_ORG']
PORT = os.environ['DOCKER_INFLUXDB_INIT_PORT']
HOST = "localhost"
URL = f"http://{HOST}:{PORT}"
default_bucket = os.environ['DOCKER_INFLUXDB_INIT_BUCKET']
assert TOKEN, "Influx token not found"
assert ORG, "Influx org not found"
assert URL, "Influx url not found"
assert default_bucket, "default_bucket not found"


class influxdb:
    @timeThis("Connection à influxDB")
    def __init__(self):
        self.client = influxdb_client.InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    def write_point(self, measurementName, tagName, tagValue, fieldName, fieldValue, bucket=default_bucket):
        point = (
            Point(measurementName)
            .tag(tagName, tagValue)
            .field(fieldName, fieldValue)
        )
        self.write_api.write(bucket=bucket, org=ORG, record=point)

    def get_all_data(self, bucket=default_bucket):
        query = f"""
        from(bucket: "{bucket}")
        |> range(start: 0)
        |> filter(fn: (r) => r._measurement == "measurement1")
        """
        tables = self.query_api.query(query, org=ORG)
        return tables

    def getDfFromQuery(self, ):
        ...

    def write_df(self, df, measurementName, timestampName='date', bucket=default_bucket):
        assert 'date' in df, 'No date column'
        self.write_api.write(bucket=bucket, record=df, data_frame_measurement_name=measurementName,
                             data_frame_timestamp_column=timestampName)

    def close(self):
        self.client.close()

    def reopen(self):
        self.client = influxdb_client.InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()


if __name__ == '__main__':
    X, Y = get_all_data()
    inf = influxdb()
    # simple test
    # for v in range(5):
    #     inf.write_point("measurement1", "tagname1", "tagvalue1", "field1", v)
    #     time.sleep(1)
    # print(inf.get_all_data())
    print(X.head())
    # inf.write_df(X,"measurement2")
