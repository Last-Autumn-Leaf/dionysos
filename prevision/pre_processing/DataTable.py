import pandas as pd

from .utils import mysql_user, mysql_password, mysql_database, WeatherDataTableName, EventAttendanceDataTableName
from sqlalchemy import create_engine, Column, Integer, DateTime, Float, Text, String, and_, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
import urllib.parse

Base = declarative_base()

encoded_url = lambda url_string: urllib.parse.quote(url_string)


class DataTable:
    def __init__(self, database):
        self.engine = create_engine(f'mysql+pymysql://{encoded_url(mysql_user)}:{encoded_url(mysql_password)}'
                                    f'@localhost:3306/{encoded_url(mysql_database)}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.database = database()
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("select 1"))
                print("MySQL connection successful")
        except Exception as e:
            print("Error:", str(e))

    def get_all_data(self):
        query = f"select * from {self.database.getName()}"
        df = pd.read_sql(query, self.engine)
        return df

    def insert_df(self, df):
        df.to_sql(self.database.getName(), con=self.engine, if_exists='append', index=False)

    def get_col_names(self):
        return self.database.__table__.columns.keys()

    def get_data_by_id_and_datetime(self, id, datetime):
        session = self.Session()
        weather_data = session.query(self.database).filter(
            and_(self.database.id == id, self.database.datetime == datetime)
        ).first()
        session.close()
        return weather_data

    def get_date_between(self, start_datetime, end_datetime, id=None):
        query = f"""
            SELECT * FROM {self.database.getName()}
            WHERE datetime >= '{start_datetime}' AND
            datetime <= '{end_datetime}'{f" AND id='{id}'" if id else ""}
            ORDER BY datetime ASC
        """

        # Execute the SQL query and retrieve the result into a Pandas DataFrame
        df = pd.read_sql_query(query, self.engine)
        return df


class WeatherDatabase(Base):
    __tablename__ = WeatherDataTableName

    id = Column(String(50), primary_key=True)
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
        return f"<WeatherDatabase(stationID={self.id}, datetime={self.datetime}, " \
               f"tempmax={self.tempmax}, tempmin={self.tempmin})>"

    def getName(self):
        return self.__tablename__


class WeatherDataTable(DataTable):
    def __init__(self):
        super().__init__(WeatherDatabase)


class EventAttendance(Base):
    __tablename__ = EventAttendanceDataTableName

    id = Column(String(50), primary_key=True)
    datetime = Column(DateTime, primary_key=True)
    attendance_concerts = Column(Integer)
    attendance_conferences = Column(Integer)
    attendance_expos = Column(Integer)
    attendance_festivals = Column(Integer)
    attendance_performing_arts = Column(Integer)
    attendance_sports = Column(Integer)

    def __repr__(self):
        return f"EventAttendance(id={self.id}, date={self.date}, " \
               f"concerts={self.attendance_concerts}, conferences={self.attendance_conferences}, " \
               f"expos={self.attendance_expos}, festivals={self.attendance_festivals}, " \
               f"performing_arts={self.attendance_performing_arts}, sports={self.attendance_sports})"

    def getName(self):
        return self.__tablename__


class EventAttendanceDataTable(DataTable):
    def __init__(self):
        super().__init__(EventAttendance)
