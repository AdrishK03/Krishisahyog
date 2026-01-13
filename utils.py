"""
utils.py - Utility functions and classes for Krishi Sahyog
Image processing, database operations, weather, and crop analysis utilities
"""

import cv2
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from PIL import Image
import random
from config import Config

# Configure logging
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
        """Preprocess image for TensorFlow Lite model inference"""
        try:
            if not image_path or not isinstance(image_path, str):
                logger.error("Invalid image path provided")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image from path: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, target_size)
            
            # Normalize to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            logger.debug(f"Image preprocessed successfully: shape {image.shape}")
            return image
            
        except cv2.error as e:
            logger.error(f"OpenCV error during image preprocessing: {e}")
            return None
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    @staticmethod
    def enhance_image_quality(image_path):
        """Enhance image quality using CLAHE"""
        try:
            if not image_path or not isinstance(image_path, str):
                logger.error("Invalid image path provided")
                return image_path
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return image_path
            
            # Convert BGR to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels back
            enhanced = cv2.merge([l, a, b])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Save enhanced image
            enhanced_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(enhanced_path, enhanced)
            
            logger.info(f"Image enhanced successfully: {enhanced_path}")
            return enhanced_path
            
        except cv2.error as e:
            logger.error(f"OpenCV error during image enhancement: {e}")
            return image_path
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image_path


class WeatherUtils:
    """Weather data retrieval and analysis"""
    
    @staticmethod
    def get_real_weather(lat, lon):
        """Get weather from Open-Meteo API and Nominatim for location"""
        try:
            weather_url = "https://api.open-meteo.com/v1/forecast"
            
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation,pressure_msl',
                'daily': 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,uv_index_max',
                'timezone': 'auto'
            }
            
            logger.info(f"Fetching weather for lat={lat}, lon={lon}")
            weather_response = requests.get(weather_url, params=params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Get location name from Nominatim
            location = WeatherUtils._get_location_name(lat, lon)
            
            # Weather code mapping
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
            
        except requests.exceptions.Timeout:
            logger.error("Weather API request timed out")
            return WeatherUtils._get_mock_weather()
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API error: {e}")
            return WeatherUtils._get_mock_weather()
        except Exception as e:
            logger.error(f"Unexpected error in weather retrieval: {e}")
            return WeatherUtils._get_mock_weather()
    
    @staticmethod
    def _get_location_name(lat, lon):
        """Get location name from coordinates using Nominatim"""
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
            logger.warning(f"Error fetching location name: {e}")
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
        """Generate agricultural alerts based on Open-Meteo weather data"""
        alerts = []
        try:
            current_temp = weather_data.get('current', {}).get('temperature_2m', 0)
            current_humidity = weather_data.get('current', {}).get('relative_humidity_2m', 0)
            current_precipitation = weather_data.get('current', {}).get('precipitation', 0)
            
            if current_temp > 35:
                alerts.append("High temperature alert - Provide shade to crops")
            if current_temp < 10:
                alerts.append("Low temperature alert - Protect sensitive crops")
            if current_humidity > 85:
                alerts.append("High humidity - Monitor for fungal diseases")
            if current_precipitation > 5:
                alerts.append("Heavy rainfall detected - Ensure proper drainage")
        except Exception as e:
            logger.error(f"Error generating agricultural alerts: {e}")
        
        return alerts
    
    @staticmethod
    def _generate_mock_alerts(season_data):
        """Generate mock agricultural alerts"""
        alerts = []
        try:
            if season_data['season'] == 'Monsoon':
                alerts.append("Heavy rainfall expected - Ensure proper drainage")
                alerts.append("High humidity - Monitor crops for disease")
            elif season_data['season'] == 'Summer':
                alerts.append("High temperature - Increase irrigation frequency")
        except Exception as e:
            logger.error(f"Error generating mock alerts: {e}")
        
        return alerts
    
    @staticmethod
    def _generate_mock_forecast():
        """Generate mock weather forecast"""
        forecast = []
        try:
            for i in range(5):
                forecast.append({
                    'date': (datetime.now() + timedelta(days=i)).date().strftime('%Y-%m-%d'),
                    'temp_max': round(random.uniform(25, 35), 1),
                    'temp_min': round(random.uniform(18, 25), 1),
                    'humidity': random.randint(60, 90),
                    'description': random.choice(['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy'])
                })
        except Exception as e:
            logger.error(f"Error generating mock forecast: {e}")
        
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
        """Analyze soil conditions and provide recommendations"""
        analysis = {
            'overall_health': 'Good',
            'recommendations': [],
            'suitable_crops': [],
            'warnings': []
        }
        
        try:
            # pH Analysis
            if ph < 5.5:
                analysis['warnings'].append('Soil is too acidic')
                analysis['recommendations'].append('Add lime to increase pH')
            elif ph > 8.0:
                analysis['warnings'].append('Soil is too alkaline')
                analysis['recommendations'].append('Add organic matter to lower pH')
            
            # Moisture Analysis
            if moisture < 30:
                analysis['warnings'].append('Low soil moisture')
                analysis['recommendations'].append('Increase irrigation frequency')
            elif moisture > 85:
                analysis['warnings'].append('Excessive soil moisture')
                analysis['recommendations'].append('Improve drainage')
            
            # Temperature Analysis
            if temperature < 15:
                analysis['warnings'].append('Low soil temperature')
                analysis['recommendations'].append('Consider season-appropriate crops')
            elif temperature > 35:
                analysis['warnings'].append('High soil temperature')
                analysis['recommendations'].append('Provide shade or mulching')
            
            # Crop Recommendations based on soil conditions
            if 5.5 <= ph <= 7.0 and 50 <= moisture <= 80:
                analysis['suitable_crops'].extend(['Rice', 'Wheat', 'Potato'])
            if 6.0 <= ph <= 7.5 and 40 <= moisture <= 70:
                analysis['suitable_crops'].extend(['Tomato', 'Corn', 'Soybean'])
            if 5.8 <= ph <= 6.8 and 45 <= moisture <= 75:
                analysis['suitable_crops'].extend(['Onion', 'Cabbage'])
            
            # Remove duplicates
            analysis['suitable_crops'] = list(set(analysis['suitable_crops']))
            
            # Overall health assessment
            warning_count = len(analysis['warnings'])
            if warning_count == 0:
                analysis['overall_health'] = 'Excellent'
            elif warning_count <= 2:
                analysis['overall_health'] = 'Good'
            elif warning_count <= 4:
                analysis['overall_health'] = 'Fair'
            else:
                analysis['overall_health'] = 'Poor'
            
            logger.info(f"Soil analysis completed: {analysis['overall_health']}")
            
        except Exception as e:
            logger.error(f"Error analyzing soil conditions: {e}")
            analysis['warnings'].append('Error during analysis')
        
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
            
            result = {
                'current_season': current_season,
                'recommended_crops': recommendations[current_season]['crops'],
                'seasonal_activities': recommendations[current_season]['activities']
            }
            
            logger.info(f"Seasonal recommendations generated: {current_season}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating seasonal recommendations: {e}")
            return {
                'current_season': 'Unknown',
                'recommended_crops': [],
                'seasonal_activities': []
            }


class TranslationUtils:
    """Provides translations for the application"""
    
    @staticmethod
    def get_translations():
        """Return comprehensive translations for all supported languages"""
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
                "hero-subtitle": "Empowering farmers with AI-powered insights, real-time data, and personalized recommendations",
                "feature-plant-health": "Plant Health Diagnosis",
                "feature-plant-health-desc": "Upload plant images for AI-powered disease and pest detection using advanced computer vision.",
                "upload-plant-image": "Click to upload plant image",
                "btn-diagnose": "Diagnose Plant",
                "feature-soil-monitoring": "মাটির পর্যবেক্ষণ",
                "feature-soil-monitoring-desc": "IoT সেন্সরের মাধ্যমে pH, আর্দ্রতা এবং তাপমাত্রার রিয়েল-টাইম পর্যবেক্ষণ।",
                "sensor-ph": "pH স্তর",
                "sensor-moisture": "আর্দ্রতা",
                "sensor-temperature": "তাপমাত্রা",
                "btn-refresh": "ডেটা রিফ্রেশ করুন",
                "feature-weather": "আবহাওয়ার তথ্য",
                "feature-weather-desc": "স্থানীয় আবহাওয়ার পূর্বাভাস এবং কৃষি সতর্কতা।",
                "btn-update-weather": "আবহাওয়া আপডেট করুন",
                "feature-market-prices": "বাজার মূল্য",
                "feature-market-prices-desc": "ফসলের রিয়েল-টাইম দাম এবং প্রবণতা।",
                "btn-update-prices": "মূল্য আপডেট করুন",
                "feature-crop-advisory": "ফসল পরামর্শ",
                "feature-crop-advisory-desc": "আপনার কৃষিক্ষেত্রের শর্ত অনুযায়ী ব্যক্তিগত পরামর্শ পান।",
                "label-crop-type": "ফসলের ধরন:",
                "label-farm-size": "ক্ষেত্রের আকার (একর):",
                "btn-get-advice": "পরামর্শ পান",
                "feature-ai-assistant": "AI সহকারী",
                "feature-ai-assistant-desc": "তাৎক্ষণিক কৃষি পরামর্শ এবং সহায়তার জন্য চ্যাট করুন।",
                "btn-chat": "চ্যাট শুরু করুন",
                "btn-voice": "ভয়েস সহকারী",
                "results-title": "বিশ্লেষণের ফলাফল",
                "chat-title": "কৃষিমিত্র সহকারী",
                "chat-welcome": "স্বাগতম! আমি কিভাবে আপনার সাহায্য করতে পারি?",
                "crop-rice": "ধান",
                "crop-wheat": "গম",
                "crop-potato": "আলু",
                "crop-tomato": "টমেটো",
                "crop-onion": "পেঁয়াজ",
                "crop-corn": "ভুট্টা",
                "crop-soybean": "সয়াবিন",
                "contact-name": "নাম",
                "contact-email": "ইমেইল",
                "contact-subject": "বিষয়",
                "contact-message": "বার্তা",
                "contact-submit": "বার্তা পাঠান",
                "card-title-sensor": "লাইভ সেন্সর ডেটা",
                "card-desc-sensor": "আপনার IoT সেন্সর থেকে রিয়েল-টাইম রিডিং।",
                "card-title-diagnoses": "সাম্প্রতিক গাছের নির্ণয়",
                "card-desc-diagnoses": "AI-চালিত রোগ শনাক্তকরণের ইতিহাস।",
                "card-title-market": "বর্তমান বাজার মূল্য",
                "card-desc-market": "স্থানীয় বাজার থেকে আপ-টু-ডেট ফসলের দাম।",
                "btn-fetch-diagnoses": "নির্ণয় আনুন",
                "diagnoses-placeholder": "কোনো সাম্প্রতিক নির্ণয় উপলব্ধ নেই।",
                "weather-location": "অবস্থান",
                "soil-analysis": "মাটি বিশ্লেষণ",
                "crop-recommendations": "ফসল সুপারিশ",
                "seasonal-advice": "মৌসুমি পরামর্শ",
                "market-trends": "বাজার প্রবণতা",
                "success": "সফল",
                "error": "ত্রুটি",
                "loading": "লোড হচ্ছে...",
                "no-data": "কোনো ডেটা উপলব্ধ নেই",
                "try-again": "আবার চেষ্টা করুন"
            }
        }
    
    @staticmethod
    def get_translation(lang, key):
        """Get a specific translation by language and key"""
        try:
            translations = TranslationUtils.get_translations()
            lang_code = lang if lang in translations else 'en'
            return translations[lang_code].get(key, f"[{key}]")
        except Exception as e:
            logger.error(f"Error getting translation: {e}")
            return f"[{key}]" "Soil Monitoring",
                "feature-soil-monitoring-desc": "Real-time soil pH, moisture, and temperature monitoring through IoT sensors.",
                "sensor-ph": "pH Level",
                "sensor-moisture": "Moisture",
                "sensor-temperature": "Temperature",
                "btn-refresh": "Refresh Data",
                "feature-weather": "Weather Insights",
                "feature-weather-desc": "Get localized weather forecasts and agricultural alerts.",
                "btn-update-weather": "Update Weather",
                "feature-market-prices": "Market Prices",
                "feature-market-prices-desc": "Real-time crop prices and market trends.",
                "btn-update-prices": "Update Prices",
                "feature-crop-advisory": "Crop Advisory",
                "feature-crop-advisory-desc": "Get personalized crop recommendations based on your farm conditions.",
                "label-crop-type": "Crop Type:",
                "label-farm-size": "Farm Size (acres):",
                "btn-get-advice": "Get Advice",
                "feature-ai-assistant": "AI Assistant",
                "feature-ai-assistant-desc": "Chat with our AI assistant for instant farming advice and support.",
                "btn-chat": "Start Chat",
                "btn-voice": "Voice Assistant",
                "results-title": "Analysis Results",
                "chat-title": "Krishi Sahyog Assistant",
                "chat-welcome": "Welcome! How can I help you today?",
                "crop-rice": "Rice",
                "crop-wheat": "Wheat",
                "crop-potato": "Potato",
                "crop-tomato": "Tomato",
                "crop-onion": "Onion",
                "crop-corn": "Corn",
                "crop-soybean": "Soybean",
                "contact-name": "Name",
                "contact-email": "Email",
                "contact-subject": "Subject",
                "contact-message": "Message",
                "contact-submit": "Send Message",
                "card-title-sensor": "Live Sensor Data",
                "card-desc-sensor": "Real-time readings from your IoT sensors.",
                "card-title-diagnoses": "Recent Plant Diagnoses",
                "card-desc-diagnoses": "AI-powered disease detection history.",
                "card-title-market": "Current Market Prices",
                "card-desc-market": "Up-to-date crop prices from local markets.",
                "btn-fetch-diagnoses": "Fetch Diagnoses",
                "diagnoses-placeholder": "No recent diagnoses available.",
                "weather-location": "Location",
                "soil-analysis": "Soil Analysis",
                "crop-recommendations": "Crop Recommendations",
                "seasonal-advice": "Seasonal Advice",
                "market-trends": "Market Trends",
                "success": "Success",
                "error": "Error",
                "loading": "Loading...",
                "no-data": "No data available",
                "try-again": "Try Again"
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
                "hero-subtitle": "AI, डेटा और व्यक्तिगत सलाह से किसानों को सशक्त बनाना",
                "feature-plant-health": "पौधों का स्वास्थ्य निदान",
                "feature-plant-health-desc": "रोग और कीट पहचान के लिए पौधों की छवियां अपलोड करें।",
                "upload-plant-image": "पौधे की छवि अपलोड करें",
                "btn-diagnose": "निदान करें",
                "feature-soil-monitoring": "मिट्टी की निगरानी",
                "feature-soil-monitoring-desc": "IoT सेंसर से pH, नमी और तापमान की वास्तविक समय निगरानी।",
                "sensor-ph": "पीएच स्तर",
                "sensor-moisture": "नमी",
                "sensor-temperature": "तापमान",
                "btn-refresh": "डेटा रीफ्रेश करें",
                "feature-weather": "मौसम जानकारी",
                "feature-weather-desc": "स्थानीय मौसम पूर्वानुमान और कृषि अलर्ट।",
                "btn-update-weather": "मौसम अपडेट करें",
                "feature-market-prices": "बाजार भाव",
                "feature-market-prices-desc": "फसलों की वास्तविक समय कीमतें और रुझान।",
                "btn-update-prices": "कीमतें अपडेट करें",
                "feature-crop-advisory": "फसल परामर्श",
                "feature-crop-advisory-desc": "आपकी खेती की स्थिति के अनुसार सलाह प्राप्त करें।",
                "label-crop-type": "फसल का प्रकार:",
                "label-farm-size": "खेती का आकार (एकड़):",
                "btn-get-advice": "सलाह प्राप्त करें",
                "feature-ai-assistant": "AI सहायक",
                "feature-ai-assistant-desc": "तुरंत कृषि सलाह और समर्थन के लिए चैट करें।",
                "btn-chat": "चैट शुरू करें",
                "btn-voice": "वॉइस असिस्टेंट",
                "results-title": "विश्लेषण परिणाम",
                "chat-title": "कृषि सहायोग सहायक",
                "chat-welcome": "स्वागत है! मैं आपकी कैसे मदद कर सकता हूं?",
                "crop-rice": "चावल",
                "crop-wheat": "गेहूं",
                "crop-potato": "आलू",
                "crop-tomato": "टमाटर",
                "crop-onion": "प्याज",
                "crop-corn": "मक्का",
                "crop-soybean": "सोयाबीन",
                "contact-name": "नाम",
                "contact-email": "ईमेल",
                "contact-subject": "विषय",
                "contact-message": "संदेश",
                "contact-submit": "संदेश भेजें",
                "card-title-sensor": "लाइव सेंसर डेटा",
                "card-desc-sensor": "आपके IoT सेंसर से वास्तविक समय रीडिंग।",
                "card-title-diagnoses": "हाल के पौधों के निदान",
                "card-desc-diagnoses": "AI-संचालित रोग पहचान इतिहास।",
                "card-title-market": "वर्तमान बाजार भाव",
                "card-desc-market": "स्थानीय बाजारों से अद्यतित फसल की कीमतें।",
                "btn-fetch-diagnoses": "निदान लाएं",
                "diagnoses-placeholder": "कोई हालिया निदान उपलब्ध नहीं।",
                "weather-location": "स्थान",
                "soil-analysis": "मिट्टी विश्लेषण",
                "crop-recommendations": "फसल सिफारिशें",
                "seasonal-advice": "मौसमी सलाह",
                "market-trends": "बाजार रुझान",
                "success": "सफलता",
                "error": "त्रुटि",
                "loading": "लोड हो रहा है...",
                "no-data": "कोई डेटा उपलब्ध नहीं",
                "try-again": "फिर कोशिश करें"
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
                "hero-subtitle": "AI, তথ্য এবং ব্যক্তিগত পরামর্শের মাধ্যমে কৃষকদের ক্ষমতায়ন",
                "feature-plant-health": "গাছের স্বাস্থ্য নির্ণয়",
                "feature-plant-health-desc": "রোগ এবং কীট শনাক্তকরণের জন্য গাছের ছবি আপলোড করুন।",
                "upload-plant-image": "গাছের ছবি আপলোড করুন",
                "btn-diagnose": "নির্ণয় করুন",
                "feature-soil-monitoring": "মাটির পর্যবেক্ষণ",
                "feature-soil-monitoring-desc": "IoT সেন্সরের মাধ্যমে pH, আর্দ্রতা এবং তাপমাত্রার রিয়েল-টাইম পর্যবেক্ষণ।",
                "sensor-ph": "pH স্তর",
                "sensor-moisture": "আর্দ্রতা",
                "sensor-temperature": "তাপমাত্রা",
                "btn-refresh": "ডেটা রিফ্রেশ করুন",
                "feature-weather": "আবহাওয়ার তথ্য",
                "feature-weather-desc": "স্থানীয় আবহাওয়ার পূর্বাভাস এবং কৃষি সতর্কতা।",
                "btn-update-weather": "আবহাওয়া আপডেট করুন",
                "feature-market-prices": "বাজার মূল্য",
                "feature-market-prices-desc": "ফসলের রিয়েল-টাইম দাম এবং প্রবণতা।",
                "btn-update-prices": "মূল্য আপডেট করুন",
                "feature-crop-advisory": "ফসল পরামর্শ",
                "feature-crop-advisory-desc": "আপনার কৃষিক্ষেত্রের শর্ত অনুযায়ী ব্যক্তিগত পরামর্শ পান।",
                "label-crop-type": "ফসলের ধরন:",
                "label-farm-size": "ক্ষেত্রের আকার (একর):",
                "btn-get-advice": "পরামর্শ পান",
                "feature-ai-assistant": "AI সহকারী",
                "feature-ai-assistant-desc": "তাৎক্ষণিক কৃষি পরামর্শ এবং সহায়তার জন্য চ্যাট করুন।",
                "btn-chat": "চ্যাট শুরু করুন",
                "btn-voice": "ভয়েস সহকারী",
                "results-title": "বিশ্লেষণের ফলাফল",
                "chat-title": "কৃষিমিত্র সহকারী",
                "chat-welcome": "স্বাগতম! আমি কিভাবে আপনার সাহায্য করতে পারি?",
                "crop-rice": "ধান",
                "crop-wheat": "গম",
                "crop-potato": "আলু",
                "crop-tomato": "টমেটো",
                "crop-onion": "পেঁয়াজ",
                "crop-corn": "ভুট্টা",
                "crop-soybean": "সয়াবিন",
                "contact-name": "নাম",
                "contact-email": "ইমেইল",
                "contact-subject": "বিষয়",
                "contact-message": "বার্তা",
                "contact-submit": "বার্তা পাঠান",
                "card-title-sensor": "লাইভ সেন্সর ডেটা",
                "card-desc-sensor": "আপনার IoT সেন্সর থেকে রিয়েল-টাইম রিডিং।",
                "card-title-diagnoses": "সাম্প্রতিক গাছের নির্ণয়",
                "card-desc-diagnoses": "AI-চালিত রোগ শনাক্তকরণের ইতিহাস।",
                "card-title-market": "বর্তমান বাজার মূল্য",
                "card-desc-market": "স্থানীয় বাজার থেকে আপ-টু-ডেট ফসলের দাম।",
                "btn-fetch-diagnoses": "নির্ণয় আনুন",
                "diagnoses-placeholder": "কোনো সাম্প্রতিক নির্ণয় উপলব্ধ নেই।",
                "weather-location": "অবস্থান",
                "soil-analysis": "মাটি বিশ্লেষণ",
                "crop-recommendations": "ফসল সুপারিশ",
                "seasonal-advice": "মৌসুমি পরামর্শ",
                "market-trends": "বাজার প্রবণতা",
                "success": "সফল",
                "error": "ত্রুটি",
                "loading": "লোড হচ্ছে...",
                "no-data": "কোনো ডেটা উপলব্ধ নেই",
                "try-again": "আবার চেষ্টা করুন"
            }
        }
