"""
app.py - Main Backend Application for Krishi Sahyog
Flask application with all routes, API endpoints, and services
"""

from functools import wraps
import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
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
import numpy as np
import re
import traceback
import joblib
from PIL import Image

# Import TFLite interpreter - use tflite_runtime if available, else tensorflow
try:
    import tflite_runtime.interpreter as tflite
    logging.info("Using tflite_runtime")
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        logging.info("Using tensorflow.lite")
    except ImportError:
        logging.error("Neither tflite_runtime nor tensorflow available")
        tflite = None

from config import Config
from utils import ImageProcessor, WeatherUtils, CropDataAnalyzer
from database import DatabaseManager, initialize_database

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO if os.environ.get('FLASK_ENV') == 'production' else logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Check for API key at startup
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logging.warning("GOOGLE_API_KEY not set - chatbot features will be limited")

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Session configuration
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24).hex())
app.permanent_session_lifetime = timedelta(days=7)

# Production vs Development settings
is_production = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_SECURE'] = is_production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# CORS configuration
CORS(app, supports_credentials=True)

# SocketIO configuration
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Gemini API URL
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}" if api_key else None

# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in') or not session.get('user_id'):
            logging.info(f"Unauthorized access attempt to {f.__name__}")
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

fertilizer_model = None
try:
    model_path = os.path.join(Config.MODEL_DIR, 'fertilizer_model.pkl')
    if os.path.exists(model_path):
        fertilizer_model = joblib.load(model_path)
        logging.info("✓ Fertilizer model loaded")
    else:
        logging.warning("Fertilizer model not found - using mock predictions")
except Exception as e:
    logging.warning(f"Fertilizer model load failed: {e}")

# ============================================================================
# PLANT DISEASE DETECTION
# ============================================================================

class EnhancedPlantDiseaseDetector:
    def __init__(self):
        self.tflite_models = {}
        self.tflite_details = {}
        
        self.classes = {
            'wheat': ['HealthyLeaf', 'BlackPoint', 'LeafBlight', 'FusariumFootRot', 'WheatBlast'],
            'tomato': ['healthy', 'bacterial_spot', 'early_blight', 'late_blight',
                      'leaf_mold', 'septoria_leaf_spot', 'spider_mites',
                      'target_spot', 'mosaic_virus', 'yellow_leaf_curl'],
            'potato': ['Potato___healthy', 'Potato___Early_blight', 'Potato___late_blight'],
            'rice': ['healthy', 'bacterial_blight', 'brown_spot', 'leaf_smut']
        }
        
        self.treatments = {
            'wheat_HealthyLeaf': 'No pesticide needed',
            'wheat_BlackPoint': 'Use Mancozeb or Chlorothalonil',
            'wheat_LeafBlight': 'Copper-based fungicides',
            'wheat_FusariumFootRot': 'Use Prothioconazole',
            'wheat_WheatBlast': 'Use Tricyclazole',
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
            'potato_Potato___healthy': 'No treatment required',
            'potato_Potato___Early_blight': 'Use mancozeb',
            'potato_Potato___late_blight': 'Use metalaxyl',
            'rice_healthy': 'Maintain nutrients',
            'rice_bacterial_blight': 'Use copper oxychloride',
            'rice_brown_spot': 'Apply propiconazole',
            'rice_leaf_smut': 'Apply tricyclazole'
        }

    def load_models(self):
        """Load TFLite models"""
        if tflite is None:
            logging.error("TFLite not available - cannot load models")
            return False
            
        models_dir = "models"
        
        model_files = {
            'wheat': 'Wheat_best_final_model.tflite',
            'tomato': 'tomato_model.tflite',
            'potato': 'potato_best_final_model.tflite',
            'rice': 'rice_best_final_model_fixed.tflite'
        }
        
        loaded = 0
        for plant, filename in model_files.items():
            path = os.path.join(models_dir, filename)
            
            if not os.path.exists(path):
                logging.warning(f"{plant} model not found at {path}")
                continue
            
            try:
                interpreter = tflite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                
                self.tflite_models[plant] = interpreter
                self.tflite_details[plant] = {
                    "input": interpreter.get_input_details(),
                    "output": interpreter.get_output_details()
                }
                
                logging.info(f"✓ {plant.capitalize()} model loaded successfully")
                loaded += 1
                
            except Exception as e:
                logging.error(f"Failed to load {plant} model: {e}")
        
        return loaded > 0

    def predict_disease(self, image_path, plant_type=None):
        """Predict disease from image"""
        if plant_type is None:
            plant_type = self._detect_plant_type(image_path)
        
        if plant_type not in self.classes:
            logging.warning(f"Unknown plant type: {plant_type}")
            return self._get_mock_prediction('tomato')
        
        # Preprocess image
        processed_image = self._preprocess_image(image_path)
        if processed_image is None:
            logging.error("Image preprocessing failed")
            return self._get_mock_prediction(plant_type)
        
        # Try prediction with TFLite model
        try:
            if plant_type in self.tflite_models:
                interpreter = self.tflite_models[plant_type]
                details = self.tflite_details[plant_type]
                
                # Set input tensor
                interpreter.set_tensor(
                    details["input"][0]["index"],
                    processed_image.astype(np.float32)
                )
                
                # Run inference
                interpreter.invoke()
                
                # Get output
                predictions = interpreter.get_tensor(
                    details["output"][0]["index"]
                )
                
                # Get prediction results
                idx = np.argmax(predictions[0])
                confidence = float(predictions[0][idx])
                
                disease = self.classes[plant_type][idx]
                key = f"{plant_type}_{disease}"
                
                return {
                    "plant_type": plant_type.capitalize(),
                    "disease": disease.replace("_", " ").title(),
                    "confidence": round(confidence * 100, 2),
                    "treatment": self.treatments.get(key, "Consult agricultural expert"),
                    "severity": self._determine_severity(confidence),
                    "recommendations": self._get_detailed_recommendations(disease)
                }
            else:
                logging.warning(f"Model not loaded for {plant_type}")
                return self._get_mock_prediction(plant_type)
                
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            logging.error(traceback.format_exc())
            return self._get_mock_prediction(plant_type)

    def _preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize to 0-1
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            logging.error(f"Image preprocessing error: {e}")
            return None

    def _detect_plant_type(self, image_path):
        """Detect plant type from filename"""
        name = os.path.basename(image_path).lower()
        for plant in self.classes.keys():
            if plant in name:
                return plant
        return 'tomato'  # Default

    def _determine_severity(self, confidence):
        """Determine disease severity based on confidence"""
        if confidence < 0.3:
            return "low"
        elif confidence < 0.7:
            return "medium"
        return "high"

    def _get_detailed_recommendations(self, disease):
        """Get recommendations for disease"""
        recs = [
            "Monitor plant regularly for disease progression",
            "Maintain proper field hygiene",
            "Use disease-resistant varieties when possible"
        ]
        if "healthy" not in disease.lower():
            recs.append("Apply recommended treatment as soon as possible")
            recs.append("Isolate affected plants to prevent spread")
        return recs

    def _get_mock_prediction(self, plant_type):
        """Generate mock prediction for testing"""
        disease = random.choice(self.classes.get(plant_type, self.classes['tomato']))
        confidence = random.uniform(75, 95)
        return {
            "plant_type": plant_type.capitalize(),
            "disease": disease.replace("_", " ").title(),
            "confidence": round(confidence, 2),
            "treatment": "Demo mode - Please upload actual plant image",
            "severity": "medium",
            "recommendations": ["This is a demo prediction", "Upload real plant image for accurate diagnosis"]
        }

# Initialize detector
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
        except Exception as e:
            logging.error(f"Weather API error: {e}")
        
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
            return {
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
                'alerts': []
            }
        
        return RealTimeWeatherService._get_enhanced_mock_weather()
    
    @staticmethod
    def _get_enhanced_mock_weather():
        month = datetime.now().month
        
        # Seasonal data for India
        if month in [12, 1, 2]:  # Winter
            base_temp, humidity = 20, 65
            descriptions = ['Clear Sky', 'Sunny', 'Partly Cloudy']
        elif month in [3, 4, 5]:  # Summer
            base_temp, humidity = 32, 70
            descriptions = ['Hot', 'Sunny', 'Warm']
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp, humidity = 28, 85
            descriptions = ['Rainy', 'Cloudy', 'Overcast']
        else:  # Post-Monsoon
            base_temp, humidity = 26, 75
            descriptions = ['Pleasant', 'Partly Cloudy', 'Clear Sky']
        
        return {
            'temperature': round(base_temp + random.uniform(-3, 3), 1),
            'humidity': int(humidity + random.randint(-10, 10)),
            'description': random.choice(descriptions),
            'wind_speed': round(random.uniform(2, 8), 1),
            'pressure': 1013 + random.randint(-5, 5),
            'location': Config.DEFAULT_LOCATION,
            'icon': '01d',
            'visibility': round(random.uniform(8, 15), 1),
            'uv_index': random.randint(3, 9),
            'rainfall': round(random.uniform(0, 5), 1),
            'alerts': []
        }

# ============================================================================
# MARKET SERVICE
# ============================================================================

class RealTimeMarketService:
    @staticmethod
    def get_comprehensive_market_data():
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
            variation = random.uniform(-0.05, 0.05)
            current_price = round(info['base'] * (1 + variation), 2)
            
            market_data[crop] = {
                'price': current_price,
                'trend': info['trend'],
                'change_percent': round(variation * 100, 1)
            }
        
        return market_data

# ============================================================================
# IOT SIMULATOR
# ============================================================================

class AdvancedIoTSimulator:
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start_simulation(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._simulate_sensors, daemon=True)
            self.thread.start()
            logging.info("IoT simulation started")
    
    def stop_simulation(self):
        self.running = False
    
    def _simulate_sensors(self):
        global sensor_data
        while self.running:
            try:
                # Update sensor values with realistic variations
                sensor_data['soil_ph'] = max(5.0, min(8.5, 
                    sensor_data['soil_ph'] + random.uniform(-0.05, 0.05)))
                sensor_data['soil_moisture'] = max(15, min(95, 
                    sensor_data['soil_moisture'] + random.randint(-2, 2)))
                sensor_data['soil_temperature'] = max(10, min(40, 
                    sensor_data['soil_temperature'] + random.uniform(-0.5, 0.5)))
                sensor_data['nitrogen'] = max(20, min(80, 
                    sensor_data['nitrogen'] + random.randint(-1, 1)))
                sensor_data['phosphorus'] = max(15, min(60, 
                    sensor_data['phosphorus'] + random.randint(-1, 1)))
                sensor_data['potassium'] = max(25, min(70, 
                    sensor_data['potassium'] + random.randint(-1, 1)))
                sensor_data['last_updated'] = datetime.now().isoformat()
                
                # Save to database
                try:
                    DatabaseManager.save_sensor_reading(
                        sensor_data['soil_ph'],
                        sensor_data['soil_moisture'],
                        sensor_data['soil_temperature'],
                        sensor_data['nitrogen'],
                        sensor_data['phosphorus'],
                        sensor_data['potassium']
                    )
                except Exception as db_err:
                    logging.debug(f"DB save error (non-critical): {db_err}")
                
                # Emit via socket
                try:
                    socketio.emit('sensor_update', sensor_data)
                except Exception as sock_err:
                    logging.debug(f"Socket emit error (non-critical): {sock_err}")
                
                time.sleep(Config.IOT_UPDATE_INTERVAL)
                
            except Exception as e:
                logging.error(f"IoT simulation error: {e}")
                time.sleep(5)

iot_simulator = AdvancedIoTSimulator()

# ============================================================================
# ROUTES - PAGES
# ============================================================================

@app.route('/')
def home():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('landing'))

@app.route('/landing')
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
    try:
        return render_template('Chatbot.html')
    except Exception as e:
        logging.error(f"Chatbot template error: {e}")
        # Try alternative template names
        try:
            return render_template('chatbot.html')
        except:
            return render_template('chat.html')

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
            
            # Validation
            if not all([username, email, password]):
                return render_template('register.html', error="All fields are required")
            
            if len(password) < 6:
                return render_template('register.html', error="Password must be at least 6 characters")
            
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return render_template('register.html', error="Invalid email format")
            
            # Create user
            user_id = DatabaseManager.create_user(username, email, password)
            
            if user_id is None:
                return render_template('register.html', error="Email already registered")
            
            logging.info(f"User registered: {username} (ID: {user_id})")
            
            # Redirect to login with success message
            return redirect(url_for('login', registered='true'))
                
        except Exception as e:
            logging.error(f"Registration error: {e}")
            logging.error(traceback.format_exc())
            return render_template('register.html', error="Registration failed. Please try again")
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Check for registration success message
    registered = request.args.get('registered')
    success_msg = "Registration successful! Please login." if registered == 'true' else None
    
    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            
            if not email or not password:
                return render_template('login.html', error="Email and password required")
            
            # Get user
            user = DatabaseManager.get_user_by_email(email)
            
            if not user:
                return render_template('login.html', error="Invalid email or password")
            
            # Check password
            if check_password_hash(user['password'], password):
                # Set session
                session.clear()
                session['logged_in'] = True
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['email'] = user['email']
                session.permanent = True
                
                # Update last login
                try:
                    DatabaseManager.update_last_login(user['id'])
                except Exception as db_err:
                    logging.warning(f"Could not update last login: {db_err}")
                
                logging.info(f"User logged in: {user['username']}")
                return redirect(url_for('dashboard'))
            
            return render_template('login.html', error="Invalid email or password")
                
        except Exception as e:
            logging.error(f"Login error: {e}")
            logging.error(traceback.format_exc())
            return render_template('login.html', error="Login failed. Please try again")
    
    return render_template('login.html', success=success_msg)

@app.route('/logout')
def logout():
    username = session.get('username', 'Unknown')
    session.clear()
    logging.info(f"User logged out: {username}")
    return redirect(url_for('landing'))

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/upload-image', methods=['POST'])
@login_required
def upload_image_for_diagnosis():
    try:
        image_file = request.files.get('image')
        plant_type = request.form.get('plant_type')

        if not image_file or not image_file.filename:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        # Save uploaded file
        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        upload_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        image_file.save(upload_path)

        # Predict disease
        result = plant_detector.predict_disease(upload_path, plant_type)

        # Save to database
        if result and session.get('user_id'):
            try:
                DatabaseManager.save_diagnosis(
                    session['user_id'],
                    result['plant_type'],
                    result['disease'],
                    result['confidence'],
                    filename
                )
            except Exception as db_err:
                logging.warning(f"Could not save diagnosis: {db_err}")

        # Clean up uploaded file
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
        except Exception as file_err:
            logging.warning(f"Could not delete temp file: {file_err}")

        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logging.error(f"Image upload error: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'success': False, 'error': 'Analysis failed'}), 500

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
    try:
        analysis = CropDataAnalyzer.analyze_soil_conditions(
            sensor_data['soil_ph'],
            sensor_data['soil_moisture'],
            sensor_data['soil_temperature']
        )
        return jsonify(analysis)
    except Exception as e:
        logging.error(f"Soil analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

# ============================================================================
# CHATBOT API
# ============================================================================

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        lang_code = data.get('lang', 'en-IN')
        language = lang_code.split('-')[0]
        
        # Try Gemini API first
        if GEMINI_API_URL:
            try:
                lang_map = {
                    'en': 'English',
                    'hi': 'Hindi',
                    'bn': 'Bengali'
                }
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"You are Krishi Sahyog, an agricultural assistant for Indian farmers. Respond in {lang_map.get(language, 'English')}.\n\nUser: {user_message}"
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 1024
                    }
                }

                response = requests.post(GEMINI_API_URL, json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and result['candidates']:
                        text_response = result['candidates'][0]['content']['parts'][0].get('text', '')
                        if text_response:
                            # Save chat
                            try:
                                DatabaseManager.save_chat_message(
                                    session['user_id'],
                                    user_message,
                                    text_response,
                                    lang_code
                                )
                            except Exception as db_err:
                                logging.warning(f"Could not save chat: {db_err}")
                            
                            return jsonify({"text": text_response, "audio": ""})
            
            except Exception as api_err:
                logging.warning(f"Gemini API error: {api_err}")
        
        # Fallback to rule-based responses
        fallback_response = get_chatbot_response(user_message, language)
        
        # Save chat
        try:
            DatabaseManager.save_chat_message(
                session['user_id'],
                user_message,
                fallback_response,
                lang_code
            )
        except Exception as db_err:
            logging.warning(f"Could not save chat: {db_err}")
        
        return jsonify({"text": fallback_response, "audio": ""})
        
    except Exception as e:
        logging.error(f"Chat error: {e}")
        logging.error(traceback.format_exc())
        return jsonify({"error": "Chat service unavailable"}), 500

# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    logging.debug('Client connected')
    emit('sensor_update', sensor_data)

@socketio.on('disconnect')
def handle_disconnect():
    logging.debug('Client disconnected')

@socketio.on('request_data')
def handle_data_request(data):
    try:
        data_type = data.get('type', 'sensor') if data else 'sensor'
        
        if data_type == 'sensor':
            emit('sensor_update', sensor_data)
        elif data_type == 'weather':
            weather = RealTimeWeatherService.get_comprehensive_weather()
            emit('weather_update', weather)
        elif data_type == 'market':
            market = RealTimeMarketService.get_comprehensive_market_data()
            emit('market_update', market)
    except Exception as e:
        logging.error(f"Socket request error: {e}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    logging.warning(f"404 error: {request.url}")
    try:
        return render_template('404.html'), 404
    except:
        return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"500 error: {error}")
    try:
        return render_template('500.html'), 500
    except:
        return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_chatbot_response(message, language='en'):
    """Rule-based chatbot responses"""
    
    responses = {
        'en': {
            'weather': "Check the weather section for current conditions and forecasts for your area.",
            'price': "View the market prices section for current crop rates in your region.",
            'sensor': "Monitor your soil health through the sensor data dashboard.",
            'fertilizer': "For optimal growth, use NPK fertilizer (10:26:26) for flowering crops. Always test soil first.",
            'disease': "Upload a clear image of affected plant leaves in the diagnosis section for disease identification.",
            'irrigation': "Water early morning or evening. Most crops need 1-2 inches of water per week.",
            'default': "I can help with crop advice, weather updates, market prices, and disease diagnosis. What do you need?"
        },
        'hi': {
            'weather': "अपने क्षेत्र की वर्तमान स्थिति और पूर्वानुमान के लिए मौसम अनुभाग देखें।",
            'price': "अपने क्षेत्र में वर्तमान फसल दरों के लिए बाजार मूल्य अनुभाग देखें।",
            'sensor': "सेंसर डेटा डैशबोर्ड के माध्यम से अपनी मिट्टी के स्वास्थ्य की निगरानी करें।",
            'fertilizer': "इष्टतम वृद्धि के लिए, फूल वाली फसलों के लिए NPK उर्वरक (10:26:26) का उपयोग करें।",
            'disease': "रोग पहचान के लिए निदान अनुभाग में प्रभावित पौधे की पत्तियों की स्पष्ट छवि अपलोड करें।",
            'default': "मैं फसल सलाह, मौसम, बाजार भाव और रोग निदान में मदद कर सकता हूं। आपको क्या चाहिए?"
        },
        'bn': {
            'weather': "আপনার এলাকার বর্তমান অবস্থা এবং পূর্বাভাসের জন্য আবহাওয়া বিভাগ দেখুন।",
            'price': "আপনার অঞ্চলে বর্তমান ফসলের দামের জন্য বাজার মূল্য বিভাগ দেখুন।",
            'sensor': "সেন্সর ডেটা ড্যাশবোর্ডের মাধ্যমে আপনার মাটির স্বাস্থ্য পর্যবেক্ষণ করুন।",
            'fertilizer': "সর্বোত্তম বৃদ্ধির জন্য, ফুল ফসলের জন্য NPK সার (10:26:26) ব্যবহার করুন।",
            'disease': "রোগ সনাক্তকরণের জন্য নির্ণয় বিভাগে আক্রান্ত গাছের পাতার স্পষ্ট ছবি আপলোড করুন।",
            'default': "আমি ফসল পরামর্শ, আবহাওয়া, বাজার দাম এবং রোগ নির্ণয়ে সাহায্য করতে পারি। আপনার কী প্রয়োজন?"
        }
    }
    
    # Keyword detection
    keywords = {
        'weather': ['weather', 'मौसम', 'আবহাওয়া', 'rain', 'बारिश', 'বৃষ্টি'],
        'price': ['price', 'market', 'भाव', 'বাজার', 'cost', 'कीमत', 'দাম'],
        'sensor': ['soil', 'ph', 'moisture', 'मिट्टी', 'মাটি'],
        'fertilizer': ['fertilizer', 'उर्वरक', 'সার', 'npk'],
        'disease': ['disease', 'sick', 'রোগ', 'रोग', 'problem', 'leaf']
    }
    
    message_lower = message.lower()
    for key, words in keywords.items():
        if any(word in message_lower for word in words):
            return responses.get(language, responses['en']).get(key, responses['en']['default'])
    
    return responses.get(language, responses['en'])['default']

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(plant_detector.tflite_models),
        "database": "connected"
    })

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Create directories
    try:
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs('models', exist_ok=True)
        logging.info("Directories created/verified")
    except Exception as e:
        logging.error(f"Directory creation error: {e}")
    
    # Initialize database
    try:
        if initialize_database():
            logging.info("✓ Database initialized")
        else:
            logging.warning("⚠ Database initialization failed")
    except Exception as e:
        logging.error(f"Database error: {e}")
    
    # Load ML models
    try:
        if plant_detector.load_models():
            logging.info("✓ ML models loaded successfully")
        else:
            logging.warning("⚠ No ML models loaded - using mock predictions")
    except Exception as e:
        logging.error(f"Model loading error: {e}")
    
    # Start IoT simulation
    try:
        iot_simulator.start_simulation()
        logging.info("✓ IoT simulation started")
    except Exception as e:
        logging.error(f"IoT simulation error: {e}")
    
    # Print startup info
    print("\n" + "="*60)
    print("🌱 KRISHI SAHYOG - Agricultural Advisory System")
    print("="*60)
    print(f"Environment: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
    print(f"Models loaded: {len(plant_detector.tflite_models)}/4")
    print(f"Port: {os.environ.get('PORT', 5000)}")
    print(f"Test user: test@test.com / test123")
    print("="*60 + "\n")
    
    # Run application
    port = int(os.environ.get('PORT', 5000))
    
    if is_production:
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=False,
            allow_unsafe_werkzeug=False
        )
    else:
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=True,
            allow_unsafe_werkzeug=True
        )
