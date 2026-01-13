"""
utils.py - Utility functions and classes for Krishi Sahyog
Image processing, weather, and crop analysis utilities
"""

import cv2
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
import random
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing for plant disease detection"""
    
    @staticmethod
    def allowed_file(filename):
        """Check if file extension is allowed"""
        try:
            if not filename or '.' not in filename:
                return False
            ext = filename.rsplit('.', 1)[1].lower()
            return ext in Config.ALLOWED_EXTENSIONS
        except Exception as e:
            logger.error(f"Error checking file: {e}")
            return False
    
    @staticmethod
    def preprocess_for_ml(image_path, target_size=(224, 224)):
        """Preprocess image for ML model inference"""
        try:
            if not image_path:
                logger.error("Invalid image path")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size)
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    @staticmethod
    def enhance_image_quality(image_path):
        """Enhance image quality using CLAHE"""
        try:
            if not image_path:
                return image_path
            
            image = cv2.imread(image_path)
            if image is None:
                return image_path
            
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            enhanced_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(enhanced_path, enhanced)
            
            return enhanced_path
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image_path


class WeatherUtils:
    """Weather data retrieval and analysis"""
    
    @staticmethod
    def get_real_weather(lat, lon):
        """Get weather from Open-Meteo API"""
        try:
            weather_url = "https://api.open-meteo.com/v1/forecast"
            
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation,pressure_msl',
                'daily': 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,uv_index_max',
                'timezone': 'auto'
            }
            
            weather_response = requests.get(weather_url, params=params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            location = WeatherUtils._get_location_name(lat, lon)
            
            weather_codes = {
                0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
                45: 'Fog', 48: 'Depositing rime fog', 
                51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
                61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
                80: 'Slight rain showers', 81: 'Moderate rain showers', 82: 'Violent rain showers',
                95: 'Thunderstorm', 96: 'Thunderstorm with hail', 99: 'Thunderstorm with heavy hail'
            }
            
            icon_map = {
                'Clear sky': '01d', 'Partly cloudy': '02d', 'Overcast': '04d', 'Rain': '09d',
                'Drizzle': '09d', 'Thunderstorm': '11d', 'Fog': '50d', 'Snow': '13d'
            }
            
            current = weather_data.get('current', {})
            daily = weather_data.get('daily', {})
            
            description = weather_codes.get(current.get('weather_code', 0), 'Unknown')
            
            weather_result = {
                'temperature': round(current.get('temperature_2m', 0), 1),
                'humidity': current.get('relative_humidity_2m', 0),
                'description': description,
                'wind_speed': round(current.get('wind_speed_10m', 0), 1),
                'pressure': current.get('pressure_msl', 1013),
                'location': location,
                'icon': icon_map.get(description, '01d'),
                'uv_index': round(daily.get('uv_index_max', [1])[0], 1) if daily.get('uv_index_max') else 1,
                'rainfall': round(daily.get('precipitation_sum', [0])[0], 1) if daily.get('precipitation_sum') else 0,
                'visibility': round(random.uniform(8, 15), 1),
                'alerts': WeatherUtils._generate_agricultural_alerts_openmeteo(weather_data),
                'forecast': WeatherUtils._process_forecast_openmeteo(weather_data)
            }
            
            logger.info("Weather data fetched successfully")
            return weather_result
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return WeatherUtils._get_mock_weather()
    
    @staticmethod
    def _get_location_name(lat, lon):
        """Get location name from coordinates"""
        try:
            location_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
            location_response = requests.get(
                location_url,
                headers={'User-Agent': 'KrishiSahyog-App'},
                timeout=5
            )
            location_response.raise_for_status()
            location_data = location_response.json()
            
            address = location_data.get('address', {})
            location = (address.get('city') or address.get('town') or 
                       address.get('county') or address.get('state') or 'Your Location')
            
            return location
        except Exception as e:
            logger.warning(f"Error fetching location: {e}")
            return Config.DEFAULT_LOCATION
    
    @staticmethod
    def _get_mock_weather():
        """Generate mock weather data"""
        month = datetime.now().month
        season_data = WeatherUtils._get_seasonal_data(month)
        
        base_temp = season_data['base_temp'] + random.uniform(-3, 3)
        humidity = max(40, min(95, season_data['base_humidity'] + random.randint(-15, 15)))
        
        return {
            'temperature': round(base_temp, 1),
            'humidity': int(humidity),
            'description': random.choice(season_data['descriptions']),
            'wind_speed': round(random.uniform(2, 8), 1),
            'pressure': 1013 + random.randint(-10, 10),
            'location': Config.DEFAULT_LOCATION,
            'icon': random.choice(['01d', '02d', '03d', '04d', '09d', '10d', '11d']),
            'visibility': round(random.uniform(8, 15), 1),
            'uv_index': random.randint(3, 9),
            'rainfall': round(random.uniform(0, season_data['max_rainfall']), 1),
            'alerts': WeatherUtils._generate_mock_alerts(season_data),
            'forecast': WeatherUtils._generate_mock_forecast()
        }
    
    @staticmethod
    def _get_seasonal_data(month):
        """Get seasonal weather data"""
        if month in [12, 1, 2]:
            return {
                'base_temp': 20, 'base_humidity': 65,
                'descriptions': ['Clear Sky', 'Sunny', 'Partly Cloudy', 'Cool'],
                'max_rainfall': 2, 'season': 'Winter'
            }
        elif month in [3, 4, 5]:
            return {
                'base_temp': 32, 'base_humidity': 70,
                'descriptions': ['Hot', 'Sunny', 'Partly Cloudy', 'Warm'],
                'max_rainfall': 5, 'season': 'Summer'
            }
        elif month in [6, 7, 8, 9]:
            return {
                'base_temp': 28, 'base_humidity': 85,
                'descriptions': ['Heavy Rain', 'Moderate Rain', 'Light Rain', 'Cloudy', 'Overcast'],
                'max_rainfall': 25, 'season': 'Monsoon'
            }
        else:
            return {
                'base_temp': 26, 'base_humidity': 75,
                'descriptions': ['Pleasant', 'Partly Cloudy', 'Clear Sky', 'Mild'],
                'max_rainfall': 8, 'season': 'Post-Monsoon'
            }
    
    @staticmethod
    def _generate_agricultural_alerts_openmeteo(weather_data):
        """Generate agricultural alerts"""
        alerts = []
        try:
            current = weather_data.get('current', {})
            current_temp = current.get('temperature_2m', 0)
            current_humidity = current.get('relative_humidity_2m', 0)
            current_precipitation = current.get('precipitation', 0)
            
            if current_temp > 35:
                alerts.append("High temperature alert - Provide shade to crops")
            if current_temp < 10:
                alerts.append("Low temperature alert - Protect sensitive crops")
            if current_humidity > 85:
                alerts.append("High humidity - Monitor for fungal diseases")
            if current_precipitation > 5:
                alerts.append("Heavy rainfall - Ensure proper drainage")
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
        
        return alerts
    
    @staticmethod
    def _generate_mock_alerts(season_data):
        """Generate mock agricultural alerts"""
        alerts = []
        if season_data['season'] == 'Monsoon':
            alerts.append("Heavy rainfall expected - Ensure proper drainage")
            alerts.append("High humidity - Monitor crops for disease")
        elif season_data['season'] == 'Summer':
            alerts.append("High temperature - Increase irrigation frequency")
        return alerts
    
    @staticmethod
    def _generate_mock_forecast():
        """Generate mock weather forecast"""
        forecast = []
        for i in range(5):
            forecast.append({
                'date': (datetime.now() + timedelta(days=i)).date().strftime('%Y-%m-%d'),
                'temp_max': round(random.uniform(25, 35), 1),
                'temp_min': round(random.uniform(18, 25), 1),
                'humidity': random.randint(60, 90),
                'description': random.choice(['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy'])
            })
        return forecast
    
    @staticmethod
    def _process_forecast_openmeteo(data):
        """Process forecast from Open-Meteo API"""
        processed = []
        try:
            if 'daily' in data:
                daily_data = data['daily']
                times = daily_data.get('time', [])
                temps_max = daily_data.get('temperature_2m_max', [])
                temps_min = daily_data.get('temperature_2m_min', [])
                
                for i in range(min(5, len(times))):
                    processed.append({
                        'date': times[i],
                        'temp_max': round(temps_max[i], 1) if i < len(temps_max) else 0,
                        'temp_min': round(temps_min[i], 1) if i < len(temps_min) else 0,
                        'humidity': random.randint(60, 90),
                        'description': 'Daily forecast'
                    })
        except Exception as e:
            logger.error(f"Error processing forecast: {e}")
        
        return processed


class CropDataAnalyzer:
    """Analyzes soil conditions and provides crop recommendations"""
    
    @staticmethod
    def analyze_soil_conditions(ph, moisture, temperature, npk=None):
        """Analyze soil conditions"""
        analysis = {
            'overall_health': 'Good',
            'recommendations': [],
            'suitable_crops': [],
            'warnings': []
        }
        
        try:
            if ph < 5.5:
                analysis['warnings'].append('Soil is too acidic')
                analysis['recommendations'].append('Add lime to increase pH')
            elif ph > 8.0:
                analysis['warnings'].append('Soil is too alkaline')
                analysis['recommendations'].append('Add organic matter to lower pH')
            
            if moisture < 30:
                analysis['warnings'].append('Low soil moisture')
                analysis['recommendations'].append('Increase irrigation frequency')
            elif moisture > 85:
                analysis['warnings'].append('Excessive soil moisture')
                analysis['recommendations'].append('Improve drainage')
            
            if temperature < 15:
                analysis['warnings'].append('Low soil temperature')
                analysis['recommendations'].append('Consider season-appropriate crops')
            elif temperature > 35:
                analysis['warnings'].append('High soil temperature')
                analysis['recommendations'].append('Provide shade or mulching')
            
            if 5.5 <= ph <= 7.0 and 50 <= moisture <= 80:
                analysis['suitable_crops'].extend(['Rice', 'Wheat', 'Potato'])
            if 6.0 <= ph <= 7.5 and 40 <= moisture <= 70:
                analysis['suitable_crops'].extend(['Tomato', 'Corn', 'Soybean'])
            if 5.8 <= ph <= 6.8 and 45 <= moisture <= 75:
                analysis['suitable_crops'].extend(['Onion', 'Cabbage'])
            
            analysis['suitable_crops'] = list(set(analysis['suitable_crops']))
            
            warning_count = len(analysis['warnings'])
            if warning_count == 0:
                analysis['overall_health'] = 'Excellent'
            elif warning_count <= 2:
                analysis['overall_health'] = 'Good'
            elif warning_count <= 4:
                analysis['overall_health'] = 'Fair'
            else:
                analysis['overall_health'] = 'Poor'
        except Exception as e:
            logger.error(f"Error analyzing soil: {e}")
        
        return analysis
    
    @staticmethod
    def get_seasonal_recommendations(month=None):
        """Get seasonal crop recommendations"""
        try:
            if month is None:
                month = datetime.now().month
            
            if not isinstance(month, int) or month < 1 or month > 12:
                month = datetime.now().month
            
            seasons = {
                'Kharif': [4, 5, 6, 7, 8, 9],
                'Rabi': [10, 11, 12, 1, 2, 3],
            }
            
            current_season = 'Kharif' if month in seasons['Kharif'] else 'Rabi'
            
            recommendations = {
                'Kharif': {
                    'crops': ['Rice', 'Cotton', 'Corn', 'Jute', 'Sugarcane'],
                    'activities': [
                        'Prepare fields for monsoon crops',
                        'Ensure proper drainage systems',
                        'Stock up on fertilizers',
                        'Monitor weather for planting time'
                    ]
                },
                'Rabi': {
                    'crops': ['Wheat', 'Barley', 'Mustard', 'Peas', 'Potato'],
                    'activities': [
                        'Prepare winter crop fields',
                        'Plan irrigation schedule',
                        'Apply base fertilizers',
                        'Select disease-resistant varieties'
                    ]
                }
            }
            
            return {
                'current_season': current_season,
                'recommended_crops': recommendations[current_season]['crops'],
                'seasonal_activities': recommendations[current_season]['activities']
            }
        except Exception as e:
            logger.error(f"Error getting seasonal recommendations: {e}")
            return {
                'current_season': 'Unknown',
                'recommended_crops': [],
                'seasonal_activities': []
            }


class TranslationUtils:
    """Provides translations for the application"""
    
    @staticmethod
    def get_translations():
        """Return comprehensive translations"""
        return {
            "en": {
                "app-name": "Krishi Sahyog",
                "nav-logout": "Logout",
                "nav-home": "Home",
                "nav-diagnosis": "Plant Diagnosis",
                "nav-features": "Features",
                "nav-advisory": "Advisory",
                "nav-contact": "Contact",
                "nav-chatbot": "Smart Assistant",
                "nav-soil": "Soil Analysis",
                "nav-dashboard": "Dashboard",
                "hero-title": "Smart Agricultural Advisory System",
                "hero-subtitle": "Empowering farmers with AI-powered insights and real-time data",
                "feature-plant-health": "Plant Health Diagnosis",
                "feature-soil-monitoring": "Soil Monitoring",
                "feature-weather": "Weather Insights",
                "feature-market-prices": "Market Prices",
                "feature-crop-advisory": "Crop Advisory",
                "feature-ai-assistant": "AI Assistant",
                "btn-diagnose": "Diagnose Plant",
                "btn-chat": "Start Chat",
                "success": "Success",
                "error": "Error",
                "loading": "Loading..."
            },
            "hi": {
                "app-name": "कृषि सहायोग",
                "nav-logout": "लॉग आउट",
                "nav-home": "होम",
                "nav-chatbot": "स्मार्ट सहायक",
                "nav-diagnosis": "पादप रोग निदान",
                "nav-features": "विशेषताएं",
                "nav-soil": "मृदा विश्लेषण",
                "nav-advisory": "सलाह",
                "nav-contact": "संपर्क",
                "nav-dashboard": "डैशबोर्ड",
                "hero-title": "स्मार्ट कृषि परामर्श प्रणाली",
                "hero-subtitle": "AI और डेटा से किसानों को सशक्त बनाना",
                "feature-plant-health": "पौधों का स्वास्थ्य निदान",
                "feature-soil-monitoring": "मिट्टी की निगरानी",
                "feature-weather": "मौसम जानकारी",
                "feature-market-prices": "बाजार भाव",
                "feature-crop-advisory": "फसल परामर्श",
                "feature-ai-assistant": "AI सहायक",
                "btn-diagnose": "निदान करें",
                "btn-chat": "चैट शुरू करें",
                "success": "सफलता",
                "error": "त्रुटि",
                "loading": "लोड हो रहा है..."
            },
            "bn": {
                "app-name": "কৃষিমিত্র",
                "nav-diagnosis": "উদ্ভিদ নির্ণয়",
                "nav-logout": "লগআউট",
                "nav-home": "হোম",
                "nav-soil": "মাটি বিশ্লেষণ",
                "nav-chatbot": "স্মার্ট সহকারী",
                "nav-features": "বৈশিষ্ট্য",
                "nav-advisory": "পরামর্শ",
                "nav-contact": "যোগাযোগ",
                "nav-dashboard": "ড্যাশবোর্ড",
                "hero-title": "স্মার্ট কৃষি পরামর্শ ব্যবস্থা",
                "hero-subtitle": "AI এবং তথ্যের মাধ্যমে কৃষকদের ক্ষমতায়ন",
                "feature-plant-health": "গাছের স্বাস্থ্য নির্ণয়",
                "feature-soil-monitoring": "মাটির পর্যবেক্ষণ",
                "feature-weather": "আবহাওয়ার তথ্য",
                "feature-market-prices": "বাজার মূল্য",
                "feature-crop-advisory": "ফসল পরামর্শ",
                "feature-ai-assistant": "AI সহকারী",
                "btn-diagnose": "নির্ণয় করুন",
                "btn-chat": "চ্যাট শুরু করুন",
                "success": "সফল",
                "error": "ত্রুটি",
                "loading": "লোড হচ্ছে..."
            }
        }
    
    @staticmethod
    def get_translation(lang, key):
        """Get a specific translation"""
        try:
            translations = TranslationUtils.get_translations()
            lang_code = lang if lang in translations else 'en'
            return translations[lang_code].get(key, f"[{key}]")
        except Exception as e:
            logger.error(f"Error getting translation: {e}")
            return f"[{key}]"
