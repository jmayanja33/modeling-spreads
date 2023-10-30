"""
Helper script to extract stadium data
"""
import math

import pandas as pd
from root_path import ROOT_PATH
import json
from stadiums import surface_types, weather_types, roof_types


def extract_column_value(value):
    """Function to clean null values"""
    if type(value) != str and math.isnan(value):
        return "Missing"
    elif value is None:
        return "Missing"
    else:
        return value
    

class StadiumDataExtractor:

    def __init__(self):
        self.data = pd.read_csv(f"{ROOT_PATH}/Data/nfl_stadiums.csv", encoding='utf-8',
                                encoding_errors='ignore')
        self.cities = {}
        self.stadiums = {}

    def extract_data(self):
        """Function to iterate through stadiums and extract data"""
        for i in range(len(self.data)):
            name = self.data["stadium_name"][i]
            print(f"Gathering data for stadium: {name}; PROGRESS: {i+1}/{len(self.data)}")
            azimuth_angle = extract_column_value(self.data["stadium_azimuthangle"][i])
            open_date = extract_column_value(self.data["stadium_open"][i])
            latitude = extract_column_value(self.data["stadium_latitude"][i])
            longitude = extract_column_value(self.data["stadium_longitude"][i])
            elevation = extract_column_value(self.data["stadium_elevation"][i])

            # Assign id to city:
            if self.data["stadium_location"][i] not in self.cities.keys():
                self.cities[self.data["stadium_location"][i]] = i
            city = self.cities[self.data["stadium_location"][i]]

            # Find capacity
            capacity = extract_column_value(self.data["stadium_capacity"][i])
            if type(capacity) == str:
                capacity = capacity.replace(",", "")
            elif math.isnan(capacity):
                capacity = 0
            if capacity != "Missing":
                capacity = int(capacity)

            # Find roof type
            stadium_type = extract_column_value(self.data["stadium_type"][i])
            if stadium_type.lower() in roof_types.keys():
                stadium_type = roof_types[stadium_type.lower()]

            # Find surface type
            surface = extract_column_value(self.data["stadium_surface"][i])
            if surface.lower() in surface_types.keys():
                surface = surface_types[surface.lower()]

            # Find weather type
            stadium_weather_type = extract_column_value(self.data["stadium_weather_type"][i])
            if stadium_weather_type.lower() in weather_types.keys():
                stadium_weather_type = weather_types[stadium_weather_type.lower()]

            # Save info to dictionary
            self.stadiums[name] = {
                "id": i,
                "city": city,
                "open_date": open_date,
                "roof_type": stadium_type,
                "weather_type": stadium_weather_type,
                "capacity": capacity,
                "surface": surface,
                "latitude": latitude,
                "longitude": longitude,
                "azimuth_angle": azimuth_angle,
                "elevation": elevation
            }

    def save_data(self):
        print("Saving data to file")
        with open("stadiums.py", "a") as file:
            file.write("\ncities = " + json.dumps(self.cities))
            file.write("\nstadiums = " + json.dumps(self.stadiums) + "\n")

    def get_stadium_data(self):
        self.extract_data()
        self.save_data()


if __name__ == "__main__":
    stadium_extractor = StadiumDataExtractor()
    stadium_extractor.get_stadium_data()


