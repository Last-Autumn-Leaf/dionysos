import math


class Location:
    def __init__(self, name:str, lon:int, lat:int, radius:str):
        self.name = name
        self.lon=lon
        self.lat=lat
        self.radius=radius

    def get_location(self):
        return {
            "lon": self.lon,
            "lat": self.lat,
            "radius": self.radius
            }
    def get_location_origin(self):
        return str(round(self.lat,4))+','+str(round(self.lon,4))