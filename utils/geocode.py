from geopy.geocoders import Nominatim

def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="disaster_predictor")
    location = geolocator.geocode(location_name)
    if not location:
        raise ValueError(f"Location '{location_name}' not found.")
    return round(location.latitude, 2), round(location.longitude, 2)
