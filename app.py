"""
app.py - Main Backend Application for Krishi Sahyog
Flask application with all routes, API endpoints, and services
"""

from functools import wraps
import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import requests
import random
import time
from threading import Thread
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
import tensorflow as tf
import numpy as np
import re
import traceback
import joblib
from googletrans import Translator

from config import Config
from utils import ImageProcessor, WeatherUtils, CropDataAnalyzer, TranslationUtils
from database import DatabaseManager, initialize_database

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)

# Check for API key at startup
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logging.warning("GOOGLE_API_KEY not set in .env - some features may be limited")

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enhanced session configuration
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'krishi_sahyog_secret_key_2024_change_in_production')
app.permanent_session_lifetime = timedelta(days=7)
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_TYPE'] = 'filesystem'

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize translator
translator = Translator()

# Gemini API URL
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in') or not session.get('user_id'):
            app.logger.info(f"Unauthorized access attempt to {f.__name__}")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# GLOBAL SENSOR DATA
# ============================================================================

sensor_data = {
    'soil_ph': 6.5,
    'soil_moisture': 65,
    'soil_temperature': 25,
    'nitrogen': 45,
    'phosphorus': 35,
    'potassium': 50,
    'last_updated': datetime.now().isoformat()
}

# ============================================================================
# FERTILIZER MODEL
# ============================================================================

try:
    fertilizer_model = joblib.load(os.path.join(Config.MODEL_DIR, 'fertilizer_model.pkl'))
    print("✓ Fertilizer recommendation model loaded successfully.")
except:
    print("⚠ Fertilizer model file not found. Using mock predictions.")
    fertilizer_model = None

# ============================================================================
# PLANT DISEASE DETECTION
# ============================================================================

class EnhancedPlantDiseaseDetector:

    def __init__(self):
        # Keras models
        self.models = {}

        # TFLite models
        self.tflite_models = {}
        self.tflite_details = {}

        self.classes = {
            'wheat': ['HealthyLeaf', 'BlackPoint', 'LeafBlight', 'FusariumFootRot', 'WheatBlast'],
            'tomato': [
                'healthy', 'bacterial_spot', 'early_blight', 'late_blight',
                'leaf_mold', 'septoria_leaf_spot', 'spider_mites',
                'target_spot', 'mosaic_virus', 'yellow_leaf_curl'
            ],
            'potato': ['Potato___healthy', 'Potato___Early_blight', 'Potato___late_blight'],
            'rice': ['healthy', 'bacterial_blight', 'brown_spot', 'leaf_smut']
        }

        self.treatments = {
            'HealthyLeaf': 'No pesticide needed',
            'BlackPoint': 'Use Mancozeb or Chlorothalonil',
            'LeafBlight': 'Copper-based fungicides',
            'FusariumFootRot': 'Use Prothioconazole',
            'WheatBlast': 'Use Tricyclazole',

            'tomato_healthy': 'Continue preventive care',
            'tomato_bacterial_spot': 'Use copper-based bactericides',
            'tomato_early_blight': 'Apply chlorothalonil',
            'tomato_late_blight': 'Apply metalaxyl',
            'tomato_leaf_mold': 'Improve ventilation',
            'tomato_septoria_leaf_spot': 'Use copper fungicides',
            'tomato_spider_mites': 'Apply neem oil',
            'tomato_target_spot': 'Apply chlorothalonil',
            'tomato_mosaic_virus': 'Remove infected plants',
            'tomato_yellow_leaf_curl': 'Control whiteflies',

            'Potato___healthy': 'No treatment required',
            'Potato___Early_blight': 'Use mancozeb',
            'Potato___late_blight': 'Use metalaxyl',

            'rice_healthy': 'Maintain nutrients',
            'rice_bacterial_blight': 'Use copper oxychloride',
            'rice_brown_spot': 'Apply propiconazole',
            'rice_leaf_smut': 'Apply tricyclazole'
        }

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    def load_models(self):
        models_dir = "models"

        model_files = {
            'wheat': 'Wheat_best_final_model.keras',
            'tomato': 'tomato_model.tflite',
            'potato': 'potato_best_final_model.keras',
            'rice': 'rice_best_final_model_fixed.keras'
        }

        loaded = 0

        for plant, file in model_files.items():
            path = os.path.join(models_dir, file)

            if not os.path.exists(path):
                print(f"⚠ {plant} model not found")
                continue

            try:
                # ---- TFLite ----
                if file.endswith(".tflite"):
                    interpreter = tf.lite.Interpreter(model_path=path)
                    interpreter.allocate_tensors()

                    self.tflite_models[plant] = interpreter
                    self.tflite_details[plant] = {
                        "input": interpreter.get_input_details(),
                        "output": interpreter.get_output_details()
                    }

                    print(f"✓ {plant.capitalize()} TFLite model loaded")

                # ---- Keras ----
                else:
                    model = tf.keras.models.load_model(path, compile=False)
                    self.models[plant] = model
                    print(f"✓ {plant.capitalize()} Keras model loaded")

                loaded += 1

            except Exception as e:
                print(f"❌ Failed to load {plant}: {e}")

        return loaded > 0

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict_disease(self, image_path, plant_type=None):

        if plant_type is None:
            plant_type = self._detect_plant_type(image_path)

        if plant_type not in self.classes:
            return self._get_mock_prediction('tomato')

        processed_image = ImageProcessor.preprocess_for_ml(image_path)
        if processed_image is None:
            return self._get_mock_prediction(plant_type)

        try:
            # -------- TFLite --------
            if plant_type in self.tflite_models:
                interpreter = self.tflite_models[plant_type]
                details = self.tflite_details[plant_type]

                interpreter.set_tensor(
                    details["input"][0]["index"],
                    processed_image.astype(np.float32)
                )
                interpreter.invoke()

                predictions = interpreter.get_tensor(
                    details["output"][0]["index"]
                )

            # -------- Keras --------
            else:
                predictions = self.models[plant_type].predict(processed_image)

            idx = np.argmax(predictions[0])
            confidence = float(predictions[0][idx])

            disease = self.classes[plant_type][idx]
            key = f"{plant_type}_{disease}"

            return {
                "plant_type": plant_type.capitalize(),
                "disease": disease.replace("_", " ").title(),
                "confidence": round(confidence * 100, 2),
                "treatment": self.treatments.get(key, "Consult expert"),
                "severity": self._determine_severity(confidence),
                "recommendations": self._get_detailed_recommendations(key)
            }

        except Exception as e:
            print("Prediction error:", e)
            return self._get_mock_prediction(plant_type)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _detect_plant_type(self, image_path):
        name = image_path.lower()
        for plant in self.classes:
            if plant in name:
                return plant
        return random.choice(list(self.classes.keys()))

    def _determine_severity(self, confidence):
        if confidence < 0.3:
            return "low"
        elif confidence < 0.7:
            return "medium"
        return "high"

    def _get_detailed_recommendations(self, disease_key):
        recs = [
            "Monitor plant regularly",
            "Maintain field hygiene",
            "Use resistant varieties"
        ]
        if "healthy" not in disease_key:
            recs.append("Apply recommended treatment promptly")
        return recs

    def _get_mock_prediction(self, plant_type):
        disease = random.choice(self.classes[plant_type])
        conf = random.uniform(70, 95)
        return {
            "plant_type": plant_type.capitalize(),
            "disease": disease.replace("_", " ").title(),
            "confidence": round(conf, 2),
            "treatment": "Demo mode",
            "severity": "medium",
            "recommendations": ["Demo prediction"]
        }


# Initialize plant detector
plant_detector = EnhancedPlantDiseaseDetector()

# ============================================================================
# WEATHER SERVICE
# ============================================================================

class RealTimeWeatherService:
    @staticmethod
    def get_comprehensive_weather(lat=None, lon=None):
        if lat is None:
            lat = Config.DEFAULT_LAT
        if lon is None:
            lon = Config.DEFAULT_LON
            
        try:
            if Config.WEATHER_API_KEY and Config.WEATHER_API_KEY != 'your_openweather_api_key_here':
                return RealTimeWeatherService._get_real_weather(lat, lon)
            else:
                return RealTimeWeatherService._get_enhanced_mock_weather()
        except Exception as e:
            print(f"Weather service error: {e}")
            return RealTimeWeatherService._get_enhanced_mock_weather()
    
    @staticmethod
    def _get_real_weather(lat, lon):
        url = f"{Config.WEATHER_BASE_URL}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': Config.WEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            forecast_url = f"{Config.WEATHER_BASE_URL}/forecast"
            forecast_response = requests.get(forecast_url, params=params, timeout=10)
            forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
            
            weather_result = {
                'temperature': round(data['main']['temp'], 1),
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'].title(),
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'pressure': data['main']['pressure'],
                'location': data.get('name', Config.DEFAULT_LOCATION),
                'icon': data['weather'][0]['icon'],
                'visibility': data.get('visibility', 10000) / 1000,
                'uv_index': random.randint(1, 8),
                'rainfall': data.get('rain', {}).get('1h', 0),
                'alerts': RealTimeWeatherService._generate_agricultural_alerts(data)
            }
            
            if forecast_data:
                weather_result['forecast'] = RealTimeWeatherService._process_forecast(forecast_data)
            
            return weather_result
        else:
            return RealTimeWeatherService._get_enhanced_mock_weather()
    
    @staticmethod
    def _get_enhanced_mock_weather():
        month = datetime.now().month
        season_data = RealTimeWeatherService._get_seasonal_data(month)
        
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
            'alerts': RealTimeWeatherService._generate_mock_alerts(season_data),
            'forecast': RealTimeWeatherService._generate_mock_forecast()
        }
    
    @staticmethod
    def _get_seasonal_data(month):
        if month in [12, 1, 2]:
            return {'base_temp': 20, 'base_humidity': 65, 'descriptions': ['Clear Sky', 'Sunny', 'Partly Cloudy', 'Cool'], 'max_rainfall': 2, 'season': 'Winter'}
        elif month in [3, 4, 5]:
            return {'base_temp': 32, 'base_humidity': 70, 'descriptions': ['Hot', 'Sunny', 'Partly Cloudy', 'Warm'], 'max_rainfall': 5, 'season': 'Summer'}
        elif month in [6, 7, 8, 9]:
            return {'base_temp': 28, 'base_humidity': 85, 'descriptions': ['Heavy Rain', 'Moderate Rain', 'Light Rain', 'Cloudy', 'Overcast'], 'max_rainfall': 25, 'season': 'Monsoon'}
        else:
            return {'base_temp': 26, 'base_humidity': 75, 'descriptions': ['Pleasant', 'Partly Cloudy', 'Clear Sky', 'Mild'], 'max_rainfall': 8, 'season': 'Post-Monsoon'}
    
    @staticmethod
    def _generate_agricultural_alerts(weather_data):
        alerts = []
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        
        if temp > 35:
            alerts.append("High temperature alert - Provide shade to crops")
        if temp < 10:
            alerts.append("Low temperature alert - Protect sensitive crops")
        if humidity > 85:
            alerts.append("High humidity - Monitor for fungal diseases")
        if humidity < 40:
            alerts.append("Low humidity - Increase irrigation")
            
        return alerts
    
    @staticmethod
    def _generate_mock_alerts(season_data):
        alerts = []
        if season_data['season'] == 'Monsoon':
            alerts.append("Heavy rainfall expected - Ensure proper drainage")
            alerts.append("High humidity - Monitor crops for disease")
        elif season_data['season'] == 'Summer':
            alerts.append("High temperature - Increase irrigation frequency")
        return alerts
    
    @staticmethod
    def _generate_mock_forecast():
        forecast = []
        for i in range(5):
            forecast.append({
                'date': (datetime.now().date().strftime('%Y-%m-%d')),
                'temp_max': round(random.uniform(25, 35), 1),
                'temp_min': round(random.uniform(18, 25), 1),
                'humidity': random.randint(60, 90),
                'description': random.choice(['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy'])
            })
        return forecast
    
    @staticmethod
    def _process_forecast(forecast_data):
        processed = []
        for item in forecast_data['list'][:5]:
            processed.append({
                'date': item['dt_txt'][:10],
                'temp_max': round(item['main']['temp_max'], 1),
                'temp_min': round(item['main']['temp_min'], 1),
                'humidity': item['main']['humidity'],
                'description': item['weather'][0]['description'].title()
            })
        return processed

# ============================================================================
# MARKET SERVICE
# ============================================================================

class RealTimeMarketService:
    @staticmethod
    def get_comprehensive_market_data():
        try:
            real_data = RealTimeMarketService._get_government_market_data()
            if real_data:
                return real_data
        except Exception as e:
            print(f"Real market data error: {e}")
        
        return RealTimeMarketService._get_enhanced_mock_data()
    
    @staticmethod
    def _get_government_market_data():
        return None
    
    @staticmethod
    def _get_enhanced_mock_data():
        base_prices = {
            'rice': {'base': 28, 'trend': 'stable'},
            'wheat': {'base': 32, 'trend': 'up'},
            'potato': {'base': 18, 'trend': 'down'},
            'onion': {'base': 15, 'trend': 'up'},
            'tomato': {'base': 25, 'trend': 'stable'},
            'corn': {'base': 22, 'trend': 'up'},
            'soybean': {'base': 45, 'trend': 'stable'}
        }
        
        market_data = {}
        for crop, info in base_prices.items():
            base_price = info['base']
            trend = info['trend']
            
            if trend == 'up':
                variation = random.uniform(0.02, 0.08)
            elif trend == 'down':
                variation = random.uniform(-0.08, -0.02)
            else:
                variation = random.uniform(-0.03, 0.03)
            
            current_price = round(base_price * (1 + variation), 2)
            
            market_data[crop] = {
                'price': current_price,
                'trend': trend,
                'change_percent': round(variation * 100, 1),
                'quality_grades': {
                    'A': current_price,
                    'B': round(current_price * 0.85, 2),
                    'C': round(current_price * 0.7, 2)
                },
                'market_locations': ['Kolkata Market', 'Howrah Market', 'Durgapur Market']
            }
        
        return market_data

# ============================================================================
# IOT SIMULATOR
# ============================================================================

class AdvancedIoTSimulator:
    def __init__(self):
        self.running = False
        self.thread = None
        self.sensor_locations = ['Field A', 'Field B', 'Greenhouse 1']
    
    def start_simulation(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._simulate_sensors)
            self.thread.daemon = True
            self.thread.start()
            print("IoT Sensor simulation started")
    
    def stop_simulation(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _simulate_sensors(self):
        global sensor_data
        while self.running:
            try:
                sensor_data['soil_ph'] += random.uniform(-0.05, 0.05)
                sensor_data['soil_ph'] = max(5.0, min(8.5, sensor_data['soil_ph']))
                
                sensor_data['soil_moisture'] += random.randint(-2, 2)
                sensor_data['soil_moisture'] = max(15, min(95, sensor_data['soil_moisture']))
                
                sensor_data['soil_temperature'] += random.uniform(-0.5, 0.5)
                sensor_data['soil_temperature'] = max(10, min(40, sensor_data['soil_temperature']))
                
                sensor_data['nitrogen'] += random.randint(-1, 1)
                sensor_data['nitrogen'] = max(20, min(80, sensor_data['nitrogen']))
                
                sensor_data['phosphorus'] += random.randint(-1, 1)
                sensor_data['phosphorus'] = max(15, min(60, sensor_data['phosphorus']))
                
                sensor_data['potassium'] += random.randint(-1, 1)
                sensor_data['potassium'] = max(25, min(70, sensor_data['potassium']))
                
                sensor_data['last_updated'] = datetime.now().isoformat()
                
                DatabaseManager.save_sensor_reading(
                    sensor_data['soil_ph'],
                    sensor_data['soil_moisture'],
                    sensor_data['soil_temperature'],
                    sensor_data['nitrogen'],
                    sensor_data['phosphorus'],
                    sensor_data['potassium']
                )
                
                socketio.emit('sensor_update', sensor_data)
                time.sleep(Config.IOT_UPDATE_INTERVAL)
                
            except Exception as e:
                print(f"IoT Simulation error: {e}")
                time.sleep(5)

iot_simulator = AdvancedIoTSimulator()

# ============================================================================
# ROUTES - PAGES
# ============================================================================

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/diagnosis')
@login_required
def diagnosis_page():
    return render_template('diagnosis.html')

@app.route('/soil')
@login_required
def soil_page():
    return render_template('soil.html')

@app.route('/features')
@login_required
def features():
    return render_template('features.html')

@app.route('/advisory')
@login_required
def advisory():
    return render_template('advisory.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('Chatbot.html')

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            
            app.logger.info(f"Registration attempt: username={username}, email={email}")
            
            if not username or not email or not password:
                app.logger.warning("Missing required fields")
                return render_template('register.html', error="All fields are required.")
            
            if len(password) < 6:
                app.logger.warning("Password too short")
                return render_template('register.html', error="Password must be at least 6 characters long.")
            
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                app.logger.warning("Invalid email format")
                return render_template('register.html', error="Please enter a valid email address.")
            
            user_id = DatabaseManager.create_user(username, email, password)
            
            if user_id is None:
                app.logger.warning(f"User already exists: {username} or {email}")
                return render_template('register.html', error="Username or email already exists.")
            
            app.logger.info(f"User created successfully with ID: {user_id}")
            
            # Redirect to login page instead of rendering
            return redirect(url_for('login'))
                
        except Exception as e:
            app.logger.error(f"General registration error: {e}")
            app.logger.error(traceback.format_exc())
            return render_template('register.html', error="Registration failed. Please try again.")
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            
            app.logger.info(f"Login attempt for email: {email}")
            
            if not email or not password:
                app.logger.warning("Missing email or password")
                return render_template('login.html', error="Email and password are required.")
            
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return render_template('login.html', error="Please enter a valid email address.")
            
            # Get user from database
            user = DatabaseManager.get_user_by_email(email)
            
            if user is None:
                app.logger.warning(f"User not found for email: {email}")
                return render_template('login.html', error="Invalid email or password.")
            
            # Check password
            if check_password_hash(user[3], password):
                # Clear any existing session
                session.clear()
                
                # Set new session
                session['logged_in'] = True
                session['user_id'] = user[0]
                session['username'] = user[1]
                session['email'] = user[2]
                session.permanent = True
                
                # Update last login
                DatabaseManager.update_last_login(user[0])
                
                app.logger.info(f"Successful login for user: {user[1]} (ID: {user[0]})")
                app.logger.info(f"Session data: {dict(session)}")
                
                return redirect(url_for('dashboard'))
            else:
                app.logger.warning(f"Invalid password for email: {email}")
                return render_template('login.html', error="Invalid email or password.")
                
        except Exception as e:
            app.logger.error(f"General login error: {e}")
            app.logger.error(traceback.format_exc())
            return render_template('login.html', error="An error occurred during login. Please try again.")
    
    # GET request - show login form
    return render_template('login.html')

@app.route('/logout')
def logout():
    try:
        # Clear all session data
        session.pop('logged_in', None)
        session.pop('user_id', None)
        session.pop('username', None)
        session.pop('email', None)
        session.clear()
        app.logger.info("User logged out successfully")
    except Exception as e:
        app.logger.error(f"Logout error: {e}")
    
    return redirect(url_for('landing'))

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/upload-image', methods=['POST'])
@login_required
def upload_image_for_diagnosis():
    image_file = request.files.get('image')
    plant_type = request.form.get('plant_type')

    if not image_file or image_file.filename == '':
        return jsonify({'success': False, 'error': 'No image file provided.'}), 400

    filename = secure_filename(image_file.filename)
    upload_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    image_file.save(upload_path)

    result = plant_detector.predict_disease(upload_path, plant_type)

    # Save diagnosis to database
    if result and 'user_id' in session:
        DatabaseManager.save_diagnosis(
            session['user_id'],
            result['plant_type'],
            result['disease'],
            result['confidence'],
            upload_path
        )

    os.remove(upload_path)

    if result:
        return jsonify({'success': True, 'result': result})
    else:
        return jsonify({'success': False, 'error': 'Analysis failed. Please try again.'}), 500

@app.route('/api/sensor-data')
@login_required
def get_sensor_data():
    return jsonify(sensor_data)

@app.route('/api/weather-data')
@login_required
def get_weather_data():
    weather = RealTimeWeatherService.get_comprehensive_weather()
    return jsonify(weather)

@app.route('/api/market-data')
@login_required
def get_market_data():
    market = RealTimeMarketService.get_comprehensive_market_data()
    return jsonify(market)

@app.route('/api/soil-analysis')
@login_required
def get_soil_analysis():
    analysis = CropDataAnalyzer.analyze_soil_conditions(
        sensor_data['soil_ph'],
        sensor_data['soil_moisture'],
        sensor_data['soil_temperature']
    )
    return jsonify(analysis)

# ============================================================================
# CHATBOT API
# ============================================================================

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    user_message = data.get('message')
    voice_support = data.get('voice_support', False)
    lang_code = data.get('lang', 'en-IN')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    lang_map = {
        'en-IN': 'English',
        'hi-IN': 'Hindi',
        'bn-IN': 'Bengali'
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"You are Krishi Sahyog, a friendly and knowledgeable Indian agricultural assistant. You are an expert on farming, crops, weather, and government schemes for Indian farmers. Always respond in {lang_map.get(lang_code, 'English')}. Your tone is helpful and empathetic.\n\nUser: {user_message}"
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                text_response = candidate['content']['parts'][0].get('text', 'Sorry, I could not generate a response.')
            else:
                text_response = 'Sorry, I could not generate a response.'
        else:
            text_response = 'Sorry, I could not generate a response.'
        
        # Save chat to database
        if 'user_id' in session:
            DatabaseManager.save_chat_message(
                session['user_id'],
                user_message,
                text_response,
                lang_code
            )
        
        return jsonify({
            "text": text_response,
            "audio": ""
        })
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return jsonify({"error": "Failed to get response from AI"}), 500
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('sensor_update', sensor_data)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_data')
def handle_data_request(data):
    data_type = data.get('type', 'sensor')
    if data_type == 'sensor':
        emit('sensor_update', sensor_data)
    elif data_type == 'weather':
        weather = RealTimeWeatherService.get_comprehensive_weather()
        emit('weather_update', weather)
    elif data_type == 'market':
        market = RealTimeMarketService.get_comprehensive_market_data()
        emit('market_update', market)

# ============================================================================
# DEBUG ROUTES
# ============================================================================

@app.route('/debug/session')
def debug_session():
    if app.debug:
        return jsonify({
            'session': dict(session),
            'logged_in': session.get('logged_in', False),
            'user_id': session.get('user_id'),
            'username': session.get('username')
        })
    return "Debug mode only", 403

@app.route('/debug/create-user/<username>/<email>/<password>')
def debug_create_user(username, email, password):
    if app.debug:
        try:
            user_id = DatabaseManager.create_user(username, email, password)
            if user_id:
                return f"User created: {username} / {email} / {password} (ID: {user_id})"
            else:
                return "User already exists or creation failed"
        except Exception as e:
            return f"Error: {str(e)}"
    return "Debug mode only", 403

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_chatbot_response(message, language):
    """Enhanced chatbot with context-aware responses"""
    
    responses = {
        'en': {
            'weather': f"Current weather: {get_current_weather_summary()}. Perfect conditions for most crops!",
            'price': f"Latest market prices: {get_market_summary()}. Prices are generally stable.",
            'sensor': f"Current soil conditions: {get_sensor_summary()}. Your soil health looks good!",
            'fertilizer': "For optimal growth, use NPK fertilizer (10:26:26) for flowering crops, or urea for leafy vegetables. Always test soil first.",
            'disease': "Please upload a clear image of the affected plant leaves for accurate disease diagnosis. Include the whole leaf in the photo.",
            'irrigation': "Water early morning or evening. Check soil moisture at 2-3 inch depth. Most crops need 1-2 inches of water per week.",
            'pest': "Common pests in West Bengal: aphids, thrips, bollworms. Use neem oil spray or integrated pest management techniques.",
            'harvest': "Harvest timing depends on crop type. Look for visual cues: color change, firmness, size. I can provide specific guidance for your crop.",
            'storage': "Proper storage prevents 30-40% post-harvest losses. Keep produce cool, dry, and well-ventilated.",
            'organic': "Organic farming tips: Use compost, crop rotation, companion planting, beneficial insects, and organic fertilizers like vermicompost.",
            'default': "I can help with crop advice, weather updates, market prices, disease diagnosis, and farming best practices. What specific information do you need?"
        },
        'hi': {
            'weather': f"वर्तमान मौसम: {get_current_weather_summary()}। अधिकांश फसलों के लिए उपयुक्त स्थितियां!",
            'price': f"नवीनतम बाजार भाव: {get_market_summary()}। कीमतें आम तौर पर स्थिर हैं।",
            'sensor': f"वर्तमान मिट्टी की स्थिति: {get_sensor_summary()}। आपकी मिट्टी का स्वास्थ्य अच्छा है!",
            'fertilizer': "अच्छी वृद्धि के लिए, फूल वाली फसलों के लिए NPK उर्वरक (10:26:26) या पत्तेदार सब्जियों के लिए यूरिया का उपयोग करें। हमेशा पहले मिट्टी की जांच करें।",
            'disease': "सटीक रोग निदान के लिए प्रभावित पौधे की पत्तियों की स्पष्ट छवि अपलोड करें।",
            'irrigation': "सुबह जल्दी या शाम को पानी दें। 2-3 इंच की गहराई पर मिट्टी की नमी जांचें।",
            'pest': "पश्चिम बंगाल में आम कीट: एफिड्स, थ्रिप्स, बॉलवर्म। नीम तेल स्प्रे का उपयोग करें।",
            'harvest': "कटाई का समय फसल के प्रकार पर निर्भर करता है। रंग, कठोरता, आकार जैसे संकेतों को देखें।",
            'storage': "उचित भंडारण से 30-40% फसल के बाद के नुकसान को रोका जा सकता है।",
            'organic': "जैविक खेती: कंपोस्ट, फसल चक्र, साथी रोपण, और वर्मीकम्पोस्ट का उपयोग करें।",
            'default': "मैं फसल सलाह, मौसम अपडेट, बाजार भाव, रोग निदान में मदद कर सकता हूं। आपको कैसी जानकारी चाहिए?"
        },
        'bn': {
            'weather': f"বর্তমান আবহাওয়া: {get_current_weather_summary()}। বেশিরভাগ ফসলের জন্য উপযুক্ত অবস্থা!",
            'price': f"সর্বশেষ বাজার দর: {get_market_summary()}। দাম সাধারণত স্থিতিশীল।",
            'sensor': f"বর্তমান মাটির অবস্থা: {get_sensor_summary()}। আপনার মাটির স্বাস্থ্য ভালো!",
            'fertilizer': "ভালো বৃদ্ধির জন্য, ফুল ফসলের জন্য NPK সার (10:26:26) বা পাতা সবজির জন্য ইউরিয়া ব্যবহার করুন।",
            'disease': "সঠিক রোগ নির্ণয়ের জন্য আক্রান্ত গাছের পাতার স্পষ্ট ছবি আপলোড করুন।",
            'irrigation': "ভোরে বা সন্ধ্যায় পানি দিন। ২-৩ ইঞ্চি গভীরতায় মাটির আর্দ্রতা পরীক্ষা করুন।",
            'pest': "পশ্চিমবঙ্গের সাধারণ পোকা: এফিড, থ্রিপস, বলওয়ার্ম। নিম তেল স্প্রে ব্যবহার করুন।",
            'harvest': "ফসল কাটার সময় ফসলের ধরনের উপর নির্ভর করে। রং, কঠিনতা দেখুন।",
            'storage': "সঠিক সংরক্ষণ ৩০-৪০% ফসল পরবর্তী ক্ষতি প্রতিরোধ করে।",
            'organic': "জৈব চাষ: কম্পোস্ট, ফসল আবর্তন, সহচর রোপণ, কেঁচো কম্পোস্ট ব্যবহার করুন।",
            'default': "আমি ফসল পরামর্শ, আবহাওয়া আপডেট, বাজার দর, রোগ নির্ণয়ে সাহায্য করতে পারি। কী তথ্য দরকার?"
        }
    }
    
    response_key = 'default'
    keywords = {
        'weather': ['weather', 'मौसम', 'আবহাওয়া', 'rain', 'बारिश', 'বৃষ্টি'],
        'price': ['price', 'market', 'भाव', 'বাজার', 'cost', 'कीमत', 'দাম'],
        'sensor': ['soil', 'ph', 'moisture', 'मिट्टी', 'মাটি'],
        'fertilizer': ['fertilizer', 'उर्वरक', 'সার', 'npk', 'urea'],
        'disease': ['disease', 'sick', 'রোগ', 'रोग', 'problem'],
        'irrigation': ['water', 'irrigation', 'पानी', 'পানি'],
        'pest': ['pest', 'insect', 'कीट', 'পোকা'],
        'harvest': ['harvest', 'कटाई', 'ফসল কাটা'],
        'storage': ['storage', 'store', 'ভন্ডারন', 'সংরক্ষণ'],
        'organic': ['organic', 'जैविक', 'জৈব']
    }
    
    for key, words in keywords.items():
        if any(word in message for word in words):
            response_key = key
            break
    
    return responses.get(language, responses['en']).get(response_key, responses['en']['default'])

def get_current_weather_summary():
    try:
        weather = RealTimeWeatherService.get_comprehensive_weather()
        return f"{weather['temperature']}°C, {weather['description']}"
    except:
        return "25°C, Pleasant"

def get_market_summary():
    try:
        market = RealTimeMarketService.get_comprehensive_market_data()
        rice_price = market.get('rice', {}).get('price', 25)
        return f"Rice ₹{rice_price}/kg"
    except:
        return "Rice ₹25/kg"

def get_sensor_summary():
    return f"pH {sensor_data['soil_ph']:.1f}, Moisture {sensor_data['soil_moisture']}%"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize database
    if not initialize_database():
        print("WARNING: Database initialization failed!")
    
    # Load ML models
    if plant_detector.load_models():
        print("✓ Plant disease detection models loaded successfully")
    else:
        print("⚠ Models not found - using mock predictions for demo")
    
    # Start IoT simulation
    iot_simulator.start_simulation()
    
    print("🌱 Krishi Sahyog Agricultural Advisory System Starting...")
    print("📊 Real-time sensor simulation: ACTIVE")
    print("🤖 AI Plant Disease Detection: READY")
    print("🌤 Weather Integration: ACTIVE")
    print("💰 Market Price Tracking: ACTIVE")
    print("🗣 Multi-language Support: English, Hindi, Bengali")
    print("🔧 Debug routes available: /debug/session, /debug/create-user")
    print("📧 Test user: test@test.com / password: test123")
    
    # Run the application
    socketio.run(
        app, 
        debug=True, 
        host='127.0.0.1', 
        port=5000,
        allow_unsafe_werkzeug=True
    )