# Create the weather service integration file
import requests
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class WeatherService:
    """
    Weather data service providing current conditions, forecasts, and agricultural insights
    Supports OpenWeatherMap API integration with fallback mock data
    """

    def __init__(self):
        """Initialize the weather service"""
        self.api_key = os.getenv('WEATHER_API_KEY')
        # RECTIFIED: Ensured base URL is secure (https)
        self.base_url = os.getenv('WEATHER_BASE_URL', 'https://api.openweathermap.org/data/2.5')

        # Use mock data if API key is not configured
        self.use_mock_data = not bool(self.api_key)

        if self.use_mock_data:
            logger.warning("Weather API key not configured. Using mock weather data.")

        # Climate data for different regions (for mock data generation)
        self.climate_data = {
            'India': {
                'base_temp': 32, 'temp_var': 8, 'base_humidity': 70,
                'rainfall_prob': 0.3, 'wind_base': 12, 'season_factor': self._get_monsoon_factor()
            },
            'USA': {
                'base_temp': 25, 'temp_var': 12, 'base_humidity': 60,
                'rainfall_prob': 0.4, 'wind_base': 15, 'season_factor': 1.0
            },
            'Brazil': {
                'base_temp': 28, 'temp_var': 6, 'base_humidity': 75,
                'rainfall_prob': 0.5, 'wind_base': 10, 'season_factor': 1.0
            },
            'Australia': {
                'base_temp': 26, 'temp_var': 10, 'base_humidity': 55,
                'rainfall_prob': 0.2, 'wind_base': 18, 'season_factor': 1.0
            },
            'Kenya': {
                'base_temp': 24, 'temp_var': 5, 'base_humidity': 65,
                'rainfall_prob': 0.4, 'wind_base': 14, 'season_factor': 1.0
            }
        }

    def _get_monsoon_factor(self) -> float:
        """Calculate monsoon season factor for India"""
        month = datetime.now().month
        # Monsoon season (June-September) has higher rainfall probability
        if 6 <= month <= 9:
            return 2.5
        elif month in [5, 10]:  # Pre and post monsoon
            return 1.5
        else:
            return 0.8

    def test_connection(self) -> bool:
        """Test connection to weather API"""
        if self.use_mock_data:
            return True

        try:
            # RECTIFIED: Ensured secure https endpoint for test
            test_url = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={self.api_key}"
            response = requests.get(test_url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Weather API connection test failed: {e}")
            return False

    def get_current_weather(self, location: str = "Mumbai, IN") -> Dict[str, Any]:
        """Get current weather conditions"""

        if self.use_mock_data:
            return self._generate_mock_current_weather()

        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self._parse_current_weather(data)
            else:
                logger.warning(f"Weather API returned status {response.status_code}. Using mock data.")
                return self._generate_mock_current_weather()

        except Exception as e:
            logger.error(f"Error fetching current weather: {e}")
            return self._generate_mock_current_weather()

    def _generate_mock_current_weather(self) -> Dict[str, Any]:
        """Generate realistic mock current weather data"""
        climate = self.climate_data['India']
        hour_factor = np.sin((datetime.now().hour - 6) / 24 * 2 * np.pi) * 0.3
        temperature = climate['base_temp'] + hour_factor * 8 + np.random.normal(0, 2)
        humidity = max(30, min(95, climate['base_humidity'] + np.random.normal(0, 10)))
        rainfall_today = 0
        if np.random.random() < climate['rainfall_prob'] * climate['season_factor']:
            rainfall_today = np.random.exponential(3)
        wind_speed = max(5, climate['wind_base'] + np.random.normal(0, 4))
        wind_direction = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        return {
            'temperature': round(temperature, 1),
            'humidity': round(humidity),
            'rainfall': round(rainfall_today, 1),
            'wind_speed': round(wind_speed, 1),
            'wind_direction': wind_direction,
            'temp_change': round(np.random.normal(0, 1.5), 1),
            'humidity_change': round(np.random.normal(0, 5)),
            'condition': self._get_weather_condition(temperature, humidity, rainfall_today),
            'timestamp': datetime.now().isoformat()
        }

    def _parse_current_weather(self, data: Dict) -> Dict[str, Any]:
        """Parse OpenWeatherMap current weather response"""
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        rain = data.get('rain', {})
        return {
            'temperature': round(main.get('temp', 25), 1),
            'humidity': round(main.get('humidity', 65)),
            'rainfall': round(rain.get('1h', 0), 1),  # Get rainfall from the 'rain' object
            'wind_speed': round(wind.get('speed', 10) * 3.6, 1),
            'wind_direction': self._degrees_to_direction(wind.get('deg', 0)),
            'temp_change': 0,
            'humidity_change': 0,
            'condition': weather.get('main', 'Clear'),
            'timestamp': datetime.now().isoformat()
        }

    def _degrees_to_direction(self, degrees: float) -> str:
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        return directions[int((degrees + 22.5) // 45) % 8]

    def get_7day_forecast(self, location: str = "Mumbai, IN") -> List[Dict[str, Any]]:
        """Get 7-day weather forecast"""
        if self.use_mock_data:
            return self._generate_mock_forecast()

        try:
            # First get coordinates from location
            geocode_url = f"https://api.openweathermap.org/geo/1.0/direct"
            geocode_params = {'q': location, 'appid': self.api_key, 'limit': 1}
            geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)

            if geocode_response.status_code == 200:
                geo_data = geocode_response.json()
                if geo_data:
                    lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
                    # Get 7-day forecast using One Call API
                    forecast_url = f"https://api.openweathermap.org/data/2.5/onecall"
                    forecast_params = {
                        'lat': lat, 'lon': lon, 'appid': self.api_key,
                        'units': 'metric', 'exclude': 'current,minutely,hourly,alerts'
                    }
                    forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
                    if forecast_response.status_code == 200:
                        return self._parse_forecast_data(forecast_response.json())

            logger.warning("Failed to get forecast from API. Using mock data.")
            return self._generate_mock_forecast()

        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return self._generate_mock_forecast()

    def _generate_mock_forecast(self) -> List[Dict[str, Any]]:
        forecast = []
        base_date = datetime.now()
        climate = self.climate_data['India']
        for i in range(8): # Generate 8 days for onecall API
            date = base_date + timedelta(days=i)
            seasonal_trend = np.sin(i * 0.3) * 3
            max_temp = climate['base_temp'] + seasonal_trend + np.random.normal(0, 2)
            min_temp = max_temp - np.random.uniform(6, 12)
            humidity = max(30, min(95, climate['base_humidity'] + np.random.normal(0, 8)))
            rainfall = 0
            if np.random.random() < climate['rainfall_prob'] * climate['season_factor']:
                rainfall = np.random.exponential(5)
            wind_speed = max(5, climate['wind_base'] + np.random.normal(0, 3))
            forecast.append({
                'date': date.strftime('%Y-%m-%d'), 'day': date.strftime('%A'),
                'max_temp': round(max_temp, 1), 'min_temp': round(min_temp, 1),
                'humidity': round(humidity), 'rainfall': round(rainfall, 1),
                'wind_speed': round(wind_speed, 1),
                'condition': self._get_weather_condition(max_temp, humidity, rainfall),
                'description': self._get_weather_description(max_temp, humidity, rainfall)
            })
        return forecast

    def _parse_forecast_data(self, data: Dict) -> List[Dict[str, Any]]:
        forecast = []
        # One Call API returns 8 days of daily data
        for day_data in data.get('daily', [])[:8]:
            date = datetime.fromtimestamp(day_data['dt'])
            temp = day_data.get('temp', {})
            weather = day_data.get('weather', [{}])[0]
            # RECTIFIED: Correctly get rainfall for the day from the 'rain' key if it exists
            rainfall = day_data.get('rain', 0)
            forecast.append({
                'date': date.strftime('%Y-%m-%d'), 'day': date.strftime('%A'),
                'max_temp': round(temp.get('max', 30), 1), 'min_temp': round(temp.get('min', 20), 1),
                'humidity': round(day_data.get('humidity', 65)), 'rainfall': round(rainfall, 1),
                'wind_speed': round(day_data.get('wind_speed', 10) * 3.6, 1),
                'condition': weather.get('main', 'Clear'),
                'description': weather.get('description', 'clear sky')
            })
        return forecast

    def _get_weather_condition(self, temp: float, humidity: float, rainfall: float) -> str:
        if rainfall > 10: return "Heavy Rain"
        elif rainfall > 1: return "Rainy"
        elif humidity > 85: return "Cloudy"
        elif temp > 35: return "Very Hot"
        elif temp > 30: return "Hot"
        elif temp < 15: return "Cool"
        else: return "Clear"

    def _get_weather_description(self, temp: float, humidity: float, rainfall: float) -> str:
        # Simplified descriptions
        condition = self._get_weather_condition(temp, humidity, rainfall)
        return condition.lower().replace("_", " ")

    def get_weather_summary(self) -> Dict[str, float]:
        """Get today's weather summary"""
        current = self.get_current_weather()
        forecast = self.get_7day_forecast()
        today_forecast = forecast[0] if forecast else {}
        return {
            'max_temp': today_forecast.get('max_temp', current['temperature'] + 5),
            'min_temp': today_forecast.get('min_temp', current['temperature'] - 5),
            'humidity': current['humidity'],
            'wind_speed': current['wind_speed'],
            'rainfall_probability': 30 if today_forecast.get('rainfall', 0) > 0 else 10
        }

    def get_weather_alerts(self) -> List[Dict[str, str]]:
        """Generate weather alerts for farming"""
        current = self.get_current_weather()
        forecast = self.get_7day_forecast()
        alerts = []

        if current['temperature'] > 38:
            alerts.append({'severity': 'high', 'title': 'Extreme Heat Warning', 'message': f'Temperature {current["temperature"]}°C - Protect crops from heat stress.'})
        if current['humidity'] > 85:
            alerts.append({'severity': 'medium', 'title': 'High Humidity Alert', 'message': f'Humidity {current["humidity"]}% - Increased risk of fungal diseases.'})
        if current['wind_speed'] > 25:
            alerts.append({'severity': 'medium', 'title': 'Strong Wind Warning', 'message': f'Wind speed {current["wind_speed"]} km/h - Secure equipment.'})

        if forecast:
            total_rainfall = sum(day['rainfall'] for day in forecast[:3])
            if total_rainfall > 75:
                alerts.append({'severity': 'high', 'title': 'Heavy Rainfall Expected', 'message': f'{total_rainfall:.0f}mm rain in next 3 days. Ensure proper drainage.'})
        return alerts

    def get_agricultural_indices(self) -> Dict[str, Any]:
        """Calculate agricultural weather indices"""
        current = self.get_current_weather()
        # Simplified for mock data
        return {'gdd': 15.0, 'et': 4.5, 'disease_risk': 'Medium', 'spray_conditions': 'Good'}

    def get_farming_recommendations(self) -> List[Dict[str, str]]:
        """Get weather-based farming recommendations"""
        current = self.get_current_weather()
        forecast = self.get_7day_forecast()
        recommendations = []

        if current['temperature'] > 35:
            recommendations.append({'type': 'irrigation', 'message': 'High temperatures detected. Increase irrigation frequency.'})
        if current['humidity'] > 80:
            recommendations.append({'type': 'protection', 'message': 'High humidity increases disease risk. Monitor crops closely.'})
        if forecast and sum(day['rainfall'] for day in forecast[:3]) > 50:
            recommendations.append({'type': 'protection', 'message': 'Heavy rain expected. Ensure good drainage and postpone fertilizer application.'})
        if not recommendations:
            recommendations.append({'type': 'general', 'message': 'Weather conditions are favorable for normal operations.'})
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    weather_service = WeatherService()
    print("--- Testing Weather Service ---")
    current = weather_service.get_current_weather()
    print("\nCurrent Weather:", json.dumps(current, indent=2))
    # RECTIFIED: Corrected newline characters
    forecast = weather_service.get_7day_forecast()
    print("\n7-Day Forecast (First 3 days):", json.dumps(forecast[:3], indent=2))
    indices = weather_service.get_agricultural_indices()
    print("\nAgricultural Indices:", json.dumps(indices, indent=2))
    alerts = weather_service.get_weather_alerts()
    print("\nWeather Alerts:", json.dumps(alerts, indent=2))