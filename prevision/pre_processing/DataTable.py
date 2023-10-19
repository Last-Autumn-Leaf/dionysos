import pandas as pd

from .utils import mysql_user, mysql_password, mysql_database, WeatherDataTable
from sqlalchemy import create_engine, Column, Integer, DateTime, Float, Text, String, and_, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse

Base = declarative_base()

encoded_url = lambda url_string: urllib.parse.quote(url_string)


class WeatherData(Base):
    __tablename__ = WeatherDataTable

    stationID = Column(String(50), primary_key=True)
    datetime = Column(DateTime, primary_key=True)
    datetimeEpoch = Column(Integer)
    tempmax = Column(Float)
    tempmin = Column(Float)
    temp = Column(Float)
    feelslikemax = Column(Float)
    feelslikemin = Column(Float)
    feelslike = Column(Float)
    dew = Column(Float)
    humidity = Column(Float)
    precip = Column(Float)
    precipprob = Column(Float)
    precipcover = Column(Float)
    snow = Column(Float)
    snowdepth = Column(Float)
    windgust = Column(Float)
    windspeed = Column(Float)
    winddir = Column(Float)
    pressure = Column(Float)
    cloudcover = Column(Float)
    visibility = Column(Float)
    solarradiation = Column(Float)
    solarenergy = Column(Float)
    uvindex = Column(Float)
    severerisk = Column(Float)
    sunrise = Column(Text)
    sunriseEpoch = Column(Integer)
    sunset = Column(Text)
    sunsetEpoch = Column(Integer)
    moonphase = Column(Float)
    conditions = Column(Text)
    description = Column(Text)
    icon = Column(Text)
    source = Column(Text)

    def __repr__(self):
        return f"<WeatherData(stationID={self.stationID}, datetime={self.datetime}, " \
               f"tempmax={self.tempmax}, tempmin={self.tempmin})>"

    @staticmethod
    def getName():
        return WeatherData.__tablename__


class DataTable:
    def __init__(self):
        self.engine = create_engine(f'mysql+pymysql://{encoded_url(mysql_user)}:{encoded_url(mysql_password)}'
                                    f'@localhost:3306/{encoded_url(mysql_database)}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("select 1"))
                print("MySQL connection successful")
        except Exception as e:
            print("Error:", str(e))

    def insert_data_from_dataframe(self, df, dt=WeatherData):
        df.to_sql(dt.getName(), con=self.engine, if_exists='append', index=False)

    def get_col_names_from_table(self, dt=WeatherData):
        return dt.__table__.columns.keys()

    def get_weather_data_by_id_and_datetime(self, station_id, datetime):
        session = self.Session()
        weather_data = session.query(WeatherData).filter(
            and_(WeatherData.stationID == station_id, WeatherData.datetime == datetime)
        ).first()
        session.close()
        return weather_data

    def get_weather_date_between(self, station_id, start_datetime, end_datetime):
        query = f"""
            SELECT * FROM {WeatherData.getName()}
            WHERE datetime >= '{start_datetime}' AND
            datetime <= '{end_datetime}' AND stationID='{station_id}'
            ORDER BY datetime ASC
        """

        # Execute the SQL query and retrieve the result into a Pandas DataFrame
        df = pd.read_sql_query(query, self.engine)
        return df

# class DataTable():
#
#     create_table_meteo = f'''
#         CREATE TABLE IF NOT EXISTS {WeatherDataTable} (
#             stationID INTEGER,
#             datetime DATETIME,
#             datetimeEpoch INTEGER,
#             tempmax REAL,
#             tempmin REAL,
#             temp REAL,
#             feelslikemax REAL,
#             feelslikemin REAL,
#             feelslike REAL,
#             dew REAL,
#             humidity REAL,
#             precip REAL,
#             precipprob REAL,
#             precipcover REAL,
#             snow REAL,
#             snowdepth REAL,
#             windgust REAL,
#             windspeed REAL,
#             winddir REAL,
#             pressure REAL,
#             cloudcover REAL,
#             visibility REAL,
#             solarradiation REAL,
#             solarenergy REAL,
#             uvindex REAL,
#             severerisk REAL,
#             sunrise TEXT,
#             sunriseEpoch INTEGER,
#             sunset TEXT,
#             sunsetEpoch INTEGER,
#             moonphase REAL,
#             conditions TEXT,
#             description TEXT,
#             icon TEXT,
#             source TEXT,
#             PRIMARY KEY (stationID, datetime)
#     );
#     '''
#     def __init__(self):
#         self.connectMeteo()
#
#     def getConnector(self):
#         return mysql.connector.connect(
#             host='localhost',
#             port=3306,
#             user=mysql_user,
#             password=mysql_password,
#             database=mysql_database
#         )
#     def connectMeteo(self):
#         with self.getConnector() as conn :
#             cursor = conn.cursor()
#             cursor.execute(self.create_table_meteo)
#
#     def getColNamesFromTable(self,dbName=WeatherDataTable):
#         with self.getConnector() as conn :
#             cursor = conn.cursor()
#             cursor.execute(f"SELECT * FROM {dbName} LIMIT 1")
#         return [desc[0] for desc in cursor.description]
#
#     def sendDf(self,df,dbName=WeatherDataTable):
#         with self.getConnector() as conn :
#             df.to_sql(name=dbName, con=conn, if_exists='replace', index=False)
