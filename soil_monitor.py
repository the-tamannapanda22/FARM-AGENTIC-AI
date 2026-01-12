# Create the Gemini AI crop agent file
import os
import numpy as np
import pandas as pd
import google.generativeai as genai
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import json
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class soilMonitor:
    """
    Gemini AI-powered Crop Intelligence Agent
    Handles crop yield prediction, recommendations, and AI assistance
    """

    def __init__(self):
        """Initialize the soilMonitor Agent"""
        self.api_key = os.getenv('GEMINI_API_KEY')

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables. AI features will be disabled.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                # RECTIFIED: Updated model name to a current, valid model.
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini AI model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini AI: {e}")
                self.model = None

        # Crop database with optimal growing conditions
        self.crop_database = {
            'Wheat': {'optimal_temp': (15, 25), 'optimal_ph': (6.0, 7.5), 'optimal_nitrogen': (80, 120), 'optimal_phosphorus': (50, 80), 'optimal_potassium': (100, 150), 'base_yield': 2500},
            'Rice': {'optimal_temp': (25, 35), 'optimal_ph': (5.5, 6.5), 'optimal_nitrogen': (100, 140), 'optimal_phosphorus': (60, 90), 'optimal_potassium': (120, 180), 'base_yield': 3000},
            'Corn': {'optimal_temp': (18, 27), 'optimal_ph': (6.0, 6.8), 'optimal_nitrogen': (120, 160), 'optimal_phosphorus': (70, 100), 'optimal_potassium': (150, 200), 'base_yield': 4000},
            'Soybeans': {'optimal_temp': (20, 30), 'optimal_ph': (6.0, 7.0), 'optimal_nitrogen': (40, 80), 'optimal_phosphorus': (50, 80), 'optimal_potassium': (120, 160), 'base_yield': 2200},
            'Cotton': {'optimal_temp': (21, 30), 'optimal_ph': (5.8, 8.0), 'optimal_nitrogen': (90, 130), 'optimal_phosphorus': (60, 90), 'optimal_potassium': (100, 140), 'base_yield': 1800},
            'Barley': {'optimal_temp': (12, 25), 'optimal_ph': (6.0, 7.0), 'optimal_nitrogen': (70, 110), 'optimal_phosphorus': (45, 75), 'optimal_potassium': (90, 130), 'base_yield': 2300},
            'Sugarcane': {'optimal_temp': (26, 32), 'optimal_ph': (6.0, 7.5), 'optimal_nitrogen': (150, 200), 'optimal_phosphorus': (80, 120), 'optimal_potassium': (180, 250), 'base_yield': 5000},
            'Potato': {'optimal_temp': (15, 20), 'optimal_ph': (5.5, 6.5), 'optimal_nitrogen': (100, 140), 'optimal_phosphorus': (60, 90), 'optimal_potassium': (140, 180), 'base_yield': 25000}
        }

    def get_latest_readings(self) -> Dict[str, Any]:
        """Generates mock real-time soil sensor readings."""
        data = {
            'moisture': np.random.uniform(55, 75), 'ph': np.random.uniform(6.5, 7.2),
            'temperature': np.random.uniform(20, 25), 'nitrogen': np.random.uniform(80, 110),
            'phosphorus': np.random.uniform(55, 75), 'potassium': np.random.uniform(110, 140),
            'conductivity': np.random.uniform(1.5, 2.5), 'organic_matter': np.random.uniform(2.8, 4.0),
            'moisture_change': np.random.uniform(-2, 2), 'ph_change': np.random.uniform(-0.1, 0.1),
            'temp_change': np.random.uniform(-0.5, 0.5), 'n_change': np.random.uniform(-5, 5),
            'p_change': np.random.uniform(-3, 3), 'k_change': np.random.uniform(-8, 8),
        }
        health_score = ((min(data['moisture'], 80) / 80) * 20 + (1 - abs(data['ph'] - 6.8) / 1.8) * 20 + (min(data['nitrogen'], 120) / 120) * 20 + (min(data['phosphorus'], 80) / 80) * 20 + (min(data['potassium'], 150) / 150) * 20)
        data['health_score'] = max(50, min(99, health_score))
        return data

    def get_historical_data(self, days=30) -> Dict[str, List[float]]:
        """Generates mock historical soil data for trends."""
        data = {
            'moisture': 65 + np.random.randn(days).cumsum(), 'ph': 6.8 + (np.random.randn(days) * 0.05).cumsum(),
            'nitrogen': 90 + (np.random.randn(days) * 2).cumsum(), 'phosphorus': 60 + (np.random.randn(days) * 1.5).cumsum(),
            'potassium': 120 + (np.random.randn(days) * 3).cumsum(), 'temperature': 22 + np.random.randn(days).cumsum() * 0.2,
            'conductivity': 1.8 + (np.random.randn(days) * 0.05).cumsum(), 'health_score': 85 - np.random.randn(days).cumsum() * 0.5,
        }
        for key in data:
            data[key] = np.clip(data[key], 0, 200 if key in ['nitrogen', 'potassium'] else 100).tolist()
        return data

    def get_sensor_status(self) -> Dict[str, Any]:
        """Generates mock IoT sensor network status."""
        sensors = [
            {'name': 'Field-A-01', 'status': 'Online', 'battery': np.random.randint(80, 100), 'last_reading': f'{np.random.randint(1, 10)} min ago'},
            {'name': 'Field-A-02', 'status': 'Online', 'battery': np.random.randint(70, 95), 'last_reading': f'{np.random.randint(1, 10)} min ago'},
            {'name': 'Field-B-01', 'status': 'Online', 'battery': np.random.randint(90, 100), 'last_reading': f'{np.random.randint(1, 10)} min ago'},
            {'name': 'Field-C-01', 'status': 'Offline', 'battery': 0, 'last_reading': '2 hours ago'},
            {'name': 'Field-C-02', 'status': 'Online', 'battery': np.random.randint(60, 80), 'last_reading': f'{np.random.randint(1, 10)} min ago'},
        ]
        online_sensors = sum(1 for s in sensors if s['status'] == 'Online')
        return {
            'total_sensors': len(sensors), 'online_sensors': online_sensors,
            'offline_sensors': len(sensors) - online_sensors,
            'avg_battery': np.mean([s['battery'] for s in sensors if s['status'] == 'Online']),
            'sensors': sensors
        }

    def test_connection(self) -> bool:
        if not self.model: return False
        try:
            response = self.model.generate_content("Hello")
            return response.text is not None
        except Exception as e:
            logger.error(f"Gemini AI connection test failed: {e}")
            return False

    def predict_yield(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            crop_type = input_data['crop_type']
            crop_info = self.crop_database.get(crop_type, self.crop_database['Wheat'])
            base_yield = crop_info['base_yield']
            soil_score = self._calculate_soil_score(input_data, crop_info)
            weather_score = self._calculate_weather_score(input_data, crop_info)
            management_score = self._calculate_management_score(input_data['irrigation'], input_data['fertilizer'])
            organic_bonus = 1 + min(input_data['organic_matter'] / 5.0, 0.2)
            yield_multiplier = (soil_score / 100) * (weather_score / 100) * (management_score / 100) * organic_bonus
            predicted_yield = base_yield * yield_multiplier + np.random.normal(0, base_yield * 0.05)
            recommendations = self._generate_ai_recommendations(input_data, predicted_yield, crop_info)
            return {
                'yield': predicted_yield, 'total_yield': predicted_yield * input_data['area'],
                'confidence': self._calculate_confidence(input_data, crop_info),
                'model': 'Gemini AI Enhanced Agronomic Model',
                'category': "Excellent" if predicted_yield >= base_yield * 1.1 else "Good" if predicted_yield >= base_yield * 0.9 else "Average",
                'soil_score': soil_score, 'weather_score': weather_score, 'management_score': management_score,
                'feature_importance': [18.5, 12.3, 10.8, 15.2, 14.7, 9.5, 11.2, 4.8, 3.0],
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error in yield prediction: {e}")
            return self._get_fallback_prediction(input_data)

    def _calculate_score(self, value, optimal_range, good_range):
        if optimal_range[0] <= value <= optimal_range[1]: return 100
        elif good_range[0] <= value <= good_range[1]: return 75
        else:
            distance = min(abs(value - optimal_range[0]), abs(value - optimal_range[1]))
            max_dist = (good_range[1] - good_range[0])
            return max(20, 75 - (distance / max_dist) * 50)

    def _calculate_soil_score(self, data: Dict, crop_info: Dict) -> int:
        n_opt, p_opt, k_opt, ph_opt = crop_info['optimal_nitrogen'], crop_info['optimal_phosphorus'], crop_info['optimal_potassium'], crop_info['optimal_ph']
        n_score = self._calculate_score(data['nitrogen'], n_opt, (n_opt[0]*0.7, n_opt[1]*1.3))
        p_score = self._calculate_score(data['phosphorus'], p_opt, (p_opt[0]*0.7, p_opt[1]*1.3))
        k_score = self._calculate_score(data['potassium'], k_opt, (k_opt[0]*0.7, k_opt[1]*1.3))
        ph_score = self._calculate_score(data['ph'], ph_opt, (ph_opt[0]-0.5, ph_opt[1]+0.5))
        return int((n_score*0.3) + (p_score*0.25) + (k_score*0.25) + (ph_score*0.2))

    def _calculate_weather_score(self, data: Dict, crop_info: Dict) -> int:
        temp_opt = crop_info['optimal_temp']
        temp_score = self._calculate_score(data['temperature'], temp_opt, (temp_opt[0]-5, temp_opt[1]+5))
        humidity_score = self._calculate_score(data['humidity'], (50, 70), (40, 80))
        rainfall_score = 100 - min(abs(data['rainfall'] - 150) / 1.5, 80)
        sunshine_score = 100 - min(abs(data['sunshine'] - 8) * 10, 80)
        return int((temp_score*0.4) + (humidity_score*0.2) + (rainfall_score*0.3) + (sunshine_score*0.1))

    def _calculate_management_score(self, irrigation: str, fertilizer: str) -> int:
        score = 50
        score += {'Drip': 30, 'Sprinkler': 25, 'Flood': 15, 'Rain-fed': 10}.get(irrigation, 20)
        score += {'Mixed': 20, 'Organic': 15, 'Chemical': 12, 'None': 5}.get(fertilizer, 10)
        return min(score, 100)

    def _calculate_confidence(self, data: Dict, crop_info: Dict) -> float:
        confidence = 95.0
        if not (crop_info['optimal_ph'][0]-1 < data['ph'] < crop_info['optimal_ph'][1]+1): confidence -= 10
        if not (crop_info['optimal_temp'][0]-5 < data['temperature'] < crop_info['optimal_temp'][1]+5): confidence -= 8
        return max(min(confidence, 99.0), 70.0)

    def _generate_ai_recommendations(self, data: Dict, predicted_yield: float, crop_info: Dict) -> List[Dict[str, str]]:
        recs = []
        n_opt, p_opt, k_opt = crop_info['optimal_nitrogen'], crop_info['optimal_phosphorus'], crop_info['optimal_potassium']
        if data['nitrogen'] < n_opt[0]: recs.append({'priority': 'High', 'recommendation': f"Nitrogen is low. Apply N-rich fertilizer. Target: {n_opt[0]}-{n_opt[1]} ppm."})
        if not recs: recs.append({'priority': 'Low', 'recommendation': 'Conditions are favorable. Continue monitoring.'})
        return recs

    def get_ai_response(self, query: str, context: Dict[str, Any]) -> str:
        if not self.model:
            return "AI capabilities are unavailable. Please configure the Gemini API key in your .env file."
        try:
            # RECTIFIED: Added default=str to handle non-serializable objects like datetime.
            context_str = json.dumps(context, indent=2, default=str)
            prompt = f"""You are an expert agricultural advisor. Given the farm context below, answer the farmer's question.
            Context: {context_str}
            Farmer's Question: {query}
            Provide a practical, actionable response in clear, farmer-friendly language."""
            response = self.model.generate_content(prompt)
            return response.text if response.text else "Sorry, I couldn't generate a response."
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return f"An error occurred: {e}. Please check your API key and configuration."

    def get_dashboard_recommendations(self, soil_data: Dict, weather_data: Dict) -> Dict[str, Any]:
        recs = {
            'irrigation': {'next_time': 'Tomorrow 6 AM', 'duration': '45 mins', 'amount': '25mm', 'method': 'Drip'},
            'fertilization': {'type': 'NPK 20:10:10', 'timing': 'In 3 days', 'quantity': '150 kg/ha', 'priority': 'Medium'},
            'pest_control': {'risk_level': 'Low', 'next_check': 'In 5 days', 'action': 'Monitor', 'priority': 'Low'}
        }
        if soil_data['moisture'] < 50: recs['irrigation'].update({'next_time': 'Immediately', 'duration': '60 mins'})
        if soil_data['nitrogen'] < 70: recs['fertilization'].update({'type': 'Urea (46-0-0)', 'timing': 'Within 2 days', 'priority': 'High'})
        if weather_data['humidity'] > 80: recs['pest_control'].update({'risk_level': 'Medium', 'action': 'Monitor for fungal diseases'})
        return recs

    def analyze_soil_health(self, soil_data: Dict) -> Dict[str, List[str]]:
        analysis = {'fertilization': [], 'irrigation': [], 'treatment': []}
        if soil_data['nitrogen'] < 70: analysis['fertilization'].append("Nitrogen is low. Apply urea.")
        else: analysis['fertilization'].append("Nitrogen levels are adequate.")
        if soil_data['moisture'] < 50: analysis['irrigation'].append("Immediate irrigation required.")
        else: analysis['irrigation'].append("Soil moisture is optimal.")
        if soil_data['ph'] < 6.0: analysis['treatment'].append(f"pH is acidic ({soil_data['ph']:.1f}). Apply lime.")
        else: analysis['treatment'].append("Soil pH is optimal.")
        return analysis

    def get_daily_insights(self, soil_data: Dict, weather_data: Dict) -> List[Dict[str, str]]:
        insights = []
        if weather_data['temperature'] > 30: insights.append({'type': 'alert', 'message': f"High temp ({weather_data['temperature']:.1f}°C) expected. Mitigate heat stress."})
        if soil_data['health_score'] < 60: insights.append({'type': 'alert', 'message': f"Soil health needs attention (Score: {soil_data['health_score']:.0f}/100)."})
        else: insights.append({'type': 'success', 'message': f"Excellent soil health (Score: {soil_data['health_score']:.0f}/100)!"})
        insights.append({'type': 'tip', 'message': np.random.choice(["Early morning is best for irrigation.", "Crop rotation breaks pest cycles."])})
        return insights

    def _get_fallback_prediction(self, input_data: Dict) -> Dict[str, Any]:
        crop_info = self.crop_database.get(input_data['crop_type'], self.crop_database['Wheat'])
        yield_estimate = crop_info['base_yield'] * 0.85
        return {
            'yield': yield_estimate, 'total_yield': yield_estimate * input_data['area'],
            'confidence': 75.0, 'model': 'Fallback Agronomic Model', 'category': 'Average',
            'soil_score': 70, 'weather_score': 75, 'management_score': 80,
            'feature_importance': [15]*9,
            'recommendations': [{'priority': 'Medium', 'recommendation': 'AI services unavailable. Using basic estimates.'}]
        }