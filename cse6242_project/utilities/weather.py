from typing import Dict, List, Any

import requests


def get_wx_forcast(lat: float, lon: float) -> Dict[str, List[Any]]:
    """Get 7 day weather forecast"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    r = requests.get(url)
    json_data = r.json()
    return json_data["hourly"]
