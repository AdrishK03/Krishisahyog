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

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    tflite = None

from config import Config
from database import DatabaseManager, initialize_database
from utils import ImageProcessor, CropDataAnalyzer

load_dotenv()

log_level = logging.INFO if os.environ.get('FLASK_ENV') == 'production' else logging.DEBUG
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

if GOOGLE_API_KEY:
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
else:
    GEMINI_API_URL = None
    logger.warning("GOOGLE_API_KEY not set - chatbot will use fallback")

app = Flask(__name__)
app.config.from_object(Config)

app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24).hex())
app.permanent_session_lifetime = timedelta(days=7)

IS_PRODUCTION = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_SECURE'] = IS_PRODUCTION
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in') or not session.get('user_id'):
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
# PLANT DISEASE DETECTOR
# ============================================================================

class PlantDiseaseDetector:
    def __init__(self):
        self.models = {}
        self.model_details = {}
        self.available_models = []
        
        self.classes = {
                'wheat': [
                'wheat__healthy',
                'wheat__black_point',
                'wheat__leaf_blight',
                'wheat__fusarium_foot_rot',
                'wheat__wheat_blast'
            ],

            'tomato': [
                'tomato__healthy',
                'tomato__bacterial_spot',
                'tomato__early_blight',
                'tomato__late_blight',
                'tomato__leaf_mold',
                'tomato__septoria_leaf_spot',
                'tomato__spider_mites',
                'tomato__target_spot',
                'tomato__mosaic_virus',
                'tomato__yellow_leaf_curl'
            ],

            'rice': [
                'bacterial_leaf_blight',
                'brown_spot',
                'healthy',
                'leaf_blast',
                'leaf_scald',
                'narrow_brown_spot'
            ],

            'potato': [
                'Early Blight',
                'Fungi',
                'Healthy',
                'Late Blight',
                'Pest',
                'Virus'
            ]

        }

        # ===================== TREATMENTS =====================
        self.treatments = {

            # Wheat
            'wheat__healthy': 'No pesticide needed',
            'wheat__black_point': 'Use Mancozeb or Chlorothalonil',
            'wheat__leaf_blight': 'Copper-based fungicides',
            'wheat__fusarium_foot_rot': 'Use Prothioconazole',
            'wheat__wheat_blast': 'Use Tricyclazole',

            # Tomato
            'tomato__healthy': 'Continue preventive care',
            'tomato__bacterial_spot': 'Use copper-based bactericides',
            'tomato__early_blight': 'Apply chlorothalonil',
            'tomato__late_blight': 'Apply metalaxyl',
            'tomato__leaf_mold': 'Improve ventilation',
            'tomato__septoria_leaf_spot': 'Use copper fungicides',
            'tomato__spider_mites': 'Apply neem oil',
            'tomato__target_spot': 'Apply chlorothalonil',
            'tomato__mosaic_virus': 'Remove infected plants',
            'tomato__yellow_leaf_curl': 'Control whiteflies',

            # Rice
            'rice__bacterial_leaf_blight': 'Use copper oxychloride',
            'rice__brown_spot': 'Apply propiconazole',
            'rice__healthy': 'Maintain nutrients',
            'rice__leaf_blast': 'Apply tricyclazole',
            'rice__leaf_scald': 'Apply tricyclazole',
            'rice__narrow_brown_spot': 'Use mancozeb',

            # ---------- Potato ----------
            'potato__Early Blight': 'Use mancozeb',
            'potato__Fungi': 'Apply broad spectrum fungicide',
            'potato__Healthy': 'No treatment required',
            'potato__Late Blight': 'Use metalaxyl',
            'potato__Pest': 'Use recommended insecticide',
            'potato__Virus': 'Remove infected plants'
        }

    def load_models(self):
        """Check available models"""
        if not TFLITE_AVAILABLE:
            logger.warning("TFLite not available")
            return False
        
        models_dir = Config.MODEL_DIR
        model_files = {
            'wheat': 'Wheat_best_final_model.tflite',
            'tomato': 'tomato_model.tflite',
            'potato': 'potato_best_final_model.tflite',
            'rice': 'rice_best_final_model_fixed.tflite'
        }
        
        available = []
        for plant_type, filename in model_files.items():
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                available.append(plant_type)
                logger.info(f"Found model: {plant_type}")
        
        self.available_models = available
        return len(available) > 0

    def _load_model(self, plant_type):
        """Load model on demand"""
        if plant_type in self.models:
            return True
        
        if plant_type not in self.available_models:
            return False
        
        model_files = {
            'wheat': 'Wheat_best_final_model.tflite',
            'tomato': 'tomato_model.tflite',
            'potato': 'potato_best_final_model.tflite',
            'rice': 'rice_best_final_model_fixed.tflite'
        }
        
        try:
            path = os.path.join(Config.MODEL_DIR, model_files[plant_type])
            interpreter = tflite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            
            self.models[plant_type] = interpreter
            self.model_details[plant_type] = {
                'input': interpreter.get_input_details(),
                'output': interpreter.get_output_details()
            }
            
            logger.info(f"Loaded model: {plant_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {plant_type}: {e}")
            return False

    def predict(self, image_path, plant_type=None):
        """Predict disease"""
        if plant_type is None:
            plant_type = self._detect_plant_type(image_path)
        
        if plant_type not in self.classes:
            plant_type = 'tomato'
        
        processed_image = ImageProcessor.preprocess_for_ml(image_path)
        if processed_image is None:
            return self._mock_prediction(plant_type)
        
        if self._load_model(plant_type):
            try:
                interpreter = self.models[plant_type]
                details = self.model_details[plant_type]
                
                interpreter.set_tensor(details['input'][0]['index'], processed_image)
                interpreter.invoke()
                
                output = interpreter.get_tensor(details['output'][0]['index'])
                idx = np.argmax(output[0])
                confidence = float(output[0][idx])
                disease = self.classes[plant_type][idx]
                
                treatment_key = f"{plant_type}_{disease}"
                treatment = self.treatments.get(treatment_key, "Consult agricultural expert")
                
                if len(self.models) > 1:
                    del self.models[plant_type]
                    del self.model_details[plant_type]
                
                return {
                    'plant_type': plant_type.capitalize(),
                    'disease': disease.replace('_', ' ').title(),
                    'confidence': round(confidence * 100, 2),
                    'treatment': treatment,
                    'severity': 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.5 else 'low',
                    'recommendations': ['Monitor regularly', 'Apply treatment promptly']
                }
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return self._mock_prediction(plant_type)
        
        return self._mock_prediction(plant_type)

    def _detect_plant_type(self, image_path):
        """Detect plant type from filename"""
        filename = os.path.basename(image_path).lower()
        for plant in self.classes.keys():
            if plant in filename:
                return plant
        return 'tomato'

    def _mock_prediction(self, plant_type):
        """Mock prediction"""
        disease = random.choice(self.classes.get(plant_type, self.classes['tomato']))
        return {
            'plant_type': plant_type.capitalize(),
            'disease': disease.replace('_', ' ').title(),
            'confidence': round(random.uniform(75, 95), 2),
            'treatment': 'Demo - upload real image',
            'severity': 'medium',
            'recommendations': ['Demo prediction']
        }

disease_detector = PlantDiseaseDetector()

# ============================================================================
# WEATHER SERVICE - FREE API (NO KEY NEEDED)
# ============================================================================

class WeatherService:
    @staticmethod
    def get_weather(lat=None, lon=None):
        """Get weather using free Open-Meteo API"""
        if lat is None:
            lat = Config.DEFAULT_LAT
        if lon is None:
            lon = Config.DEFAULT_LON
        
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m',
                'timezone': 'auto'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get('current', {})
                
                weather_codes = {
                    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
                    45: 'Fog', 48: 'Fog', 51: 'Drizzle', 53: 'Drizzle', 55: 'Drizzle',
                    61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
                    80: 'Rain showers', 81: 'Rain showers', 82: 'Rain showers',
                    95: 'Thunderstorm', 96: 'Thunderstorm', 99: 'Thunderstorm'
                }
                
                code = current.get('weather_code', 0)
                description = weather_codes.get(code, 'Unknown')
                
                logger.info("Weather fetched from Open-Meteo API")
                return {
                    'temperature': round(current.get('temperature_2m', 0), 1),
                    'humidity': current.get('relative_humidity_2m', 0),
                    'description': description,
                    'wind_speed': round(current.get('wind_speed_10m', 0), 1),
                    'location': Config.DEFAULT_LOCATION
                }
        except Exception as e:
            logger.warning(f"Weather API error: {e}")
            return WeatherService._mock_weather()
    
    @staticmethod
    def _mock_weather():
        """Fallback mock weather"""
        month = datetime.now().month
        
        if month in [12, 1, 2]:
            temp, humid = 20, 65
        elif month in [3, 4, 5]:
            temp, humid = 32, 70
        elif month in [6, 7, 8, 9]:
            temp, humid = 28, 85
        else:
            temp, humid = 26, 75
        
        return {
            'temperature': round(temp + random.uniform(-3, 3), 1),
            'humidity': int(humid + random.randint(-10, 10)),
            'description': 'Partly Cloudy',
            'wind_speed': round(random.uniform(2, 8), 1),
            'location': Config.DEFAULT_LOCATION
        }

# ============================================================================
# MARKET SERVICE
# ============================================================================

class MarketService:
    @staticmethod
    def get_prices():
        """Get market prices"""
        crops = {
            'rice': 28, 'wheat': 32, 'potato': 18,
            'onion': 15, 'tomato': 25, 'corn': 22
        }
        
        market_data = {}
        for crop, base_price in crops.items():
            variation = random.uniform(0.95, 1.05)
            market_data[crop] = {
                'price': round(base_price * variation, 2),
                'trend': random.choice(['up', 'down', 'stable'])
            }
        
        return market_data

# ============================================================================
# IOT SIMULATOR
# ============================================================================

class IoTSimulator:
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start(self):
        """Start simulation"""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._simulate, daemon=True)
            self.thread.start()
            logger.info("IoT simulation started")
    
    def stop(self):
        """Stop simulation"""
        self.running = False
    
    def _simulate(self):
        """Simulate sensor data"""
        global sensor_data
        while self.running:
            try:
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
                
                try:
                    DatabaseManager.save_sensor_reading(
                        sensor_data['soil_ph'],
                        sensor_data['soil_moisture'],
                        sensor_data['soil_temperature'],
                        sensor_data['nitrogen'],
                        sensor_data['phosphorus'],
                        sensor_data['potassium']
                    )
                except:
                    pass
                
                try:
                    socketio.emit('sensor_update', sensor_data)
                except:
                    pass
                
                time.sleep(Config.IOT_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"IoT error: {e}")
                time.sleep(5)

iot_sim = IoTSimulator()

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    success_msg = "Registration successful! Please login." if request.args.get('registered') == 'true' else None
    
    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            
            if not email or not password:
                return render_template('login.html', error="Email and password required", success=success_msg)
            
            user = DatabaseManager.get_user_by_email(email)
            
            if user:
                user_id, username, user_email, user_password = user
                
                if check_password_hash(user_password, password):
                    session.clear()
                    session['logged_in'] = True
                    session['user_id'] = user_id
                    session['username'] = username
                    session['email'] = user_email
                    session.permanent = True
                    
                    try:
                        DatabaseManager.update_last_login(user_id)
                    except:
                        pass
                    
                    logger.info(f"User logged in: {username}")
                    return redirect(url_for('dashboard'))
                else:
                    return render_template('login.html', error="Invalid password", success=success_msg)
            else:
                return render_template('login.html', error="Email not registered", success=success_msg)
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            logger.error(traceback.format_exc())
            return render_template('login.html', error="Login failed", success=success_msg)
    
    return render_template('login.html', success=success_msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            
            if not all([username, email, password]):
                return render_template('register.html', error="All fields required")
            
            if len(password) < 6:
                return render_template('register.html', error="Password must be 6+ characters")
            
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return render_template('register.html', error="Invalid email")
            
            user_id = DatabaseManager.create_user(username, email, password)
            
            if user_id is None:
                return render_template('register.html', error="Email or username already registered")
            
            logger.info(f"User registered: {username}")
            return redirect(url_for('login', registered='true'))
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            logger.error(traceback.format_exc())
            return render_template('register.html', error="Registration failed")
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'User')
    session.clear()
    logger.info(f"{username} logged out")
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/diagnosis')
@login_required
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/soil')
@login_required
def soil():
    return render_template('soil.html')

@app.route('/chatbot')
@login_required
def chatbot():
    try:
        return render_template('Chatbot.html')
    except:
        try:
            return render_template('chatbot.html')
        except:
            return "Chatbot page not found", 404

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

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/upload-image', methods=['POST'])
@login_required
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        file = request.files['image']
        if not file.filename:
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        plant_type = request.form.get('plant_type')
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        result = disease_detector.predict(filepath, plant_type)
        
        try:
            DatabaseManager.save_diagnosis(
                session['user_id'],
                result['plant_type'],
                result['disease'],
                result['confidence'],
                filename
            )
        except:
            pass
        
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': 'Upload failed'}), 500

@app.route('/api/sensor-data')
@login_required
def get_sensor_data():
    return jsonify(sensor_data)

@app.route('/api/weather-data')
@login_required
def get_weather_data():
    return jsonify(WeatherService.get_weather())

@app.route('/api/market-data')
@login_required
def get_market_data():
    return jsonify(MarketService.get_prices())

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
        logger.error(f"Soil analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        if not data or not data.get('message'):
            return jsonify({'error': 'No message'}), 400
        
        user_message = data['message'].strip()
        lang = data.get('lang', 'en-IN')
        if '-' in lang:
            lang = lang.split('-')[0]
        
        if GEMINI_API_URL and GOOGLE_API_KEY:
            try:
                lang_map = {
                    'en': 'English',
                    'hi': 'Hindi',
                    'bn': 'Bengali'
                }
                
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": f"You are Krishi Sahyog, a helpful agricultural assistant for Indian farmers. Respond in {lang_map.get(lang, 'English')}. Be concise and helpful.\n\nFarmer's question: {user_message}"
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 500
                    }
                }
                
                response = requests.post(GEMINI_API_URL, json=payload, timeout=20)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'candidates' in result and len(result['candidates']) > 0:
                        candidate = result['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            text = candidate['content']['parts'][0].get('text', '')
                            
                            if text:
                                try:
                                    DatabaseManager.save_chat_message(
                                        session['user_id'],
                                        user_message,
                                        text,
                                        lang
                                    )
                                except:
                                    pass
                                
                                logger.info("Chat response from Gemini API")
                                return jsonify({'text': text, 'audio': ''})
                
                logger.warning(f"Gemini API response error: {response.status_code}")
                
            except requests.exceptions.Timeout:
                logger.warning("Gemini API timeout")
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
        else:
            logger.warning("Gemini API not configured")
        
        fallback = get_fallback_response(user_message, lang)
        
        try:
            DatabaseManager.save_chat_message(
                session['user_id'],
                user_message,
                fallback,
                lang
            )
        except:
            pass
        
        logger.info("Using fallback chatbot response")
        return jsonify({'text': fallback, 'audio': ''})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Chat unavailable'}), 500

def get_fallback_response(message, lang='en'):
    """Fallback chatbot responses"""
    msg_lower = message.lower()
    
    responses = {
        'en': {
            'weather': "Check the weather section for current conditions.",
            'price': "Visit market section for latest crop prices.",
            'soil': f"pH {sensor_data['soil_ph']:.1f}, Moisture {sensor_data['soil_moisture']}%",
            'disease': "Upload plant image in diagnosis section.",
            'default': "I can help with weather, prices, soil analysis, and disease diagnosis."
        },
        'hi': {
            'weather': "मौसम अनुभाग में वर्तमान स्थिति देखें।",
            'price': "बाजार अनुभाग में नवीनतम फसल मूल्य देखें।",
            'soil': f"pH {sensor_data['soil_ph']:.1f}, नमी {sensor_data['soil_moisture']}%",
            'disease': "निदान अनुभाग में छवि अपलोड करें।",
            'default': "मैं मौसम, कीमतें, मिट्टी विश्लेषण में मदद कर सकता हूं।"
        },
        'bn': {
            'weather': "আবহাওয়া বিভাগে দেখুন।",
            'price': "বাজার বিভাগে দাম দেখুন।",
            'soil': f"pH {sensor_data['soil_ph']:.1f}, আর্দ্রতা {sensor_data['soil_moisture']}%",
            'disease': "ডায়াগনসিস বিভাগে ছবি আপলোড করুন।",
            'default': "আমি আবহাওয়া, দাম, মাটি বিশ্লেষণে সাহায্য করি।"
        }
    }
    
    keywords = {
        'weather': ['weather', 'मौसम', 'আবহাওয়া'],
        'price': ['price', 'भाव', 'দাম'],
        'soil': ['soil', 'मिट्टी', 'মাটি'],
        'disease': ['disease', 'रोग', 'রোগ']
    }
    
    key = 'default'
    for k, words in keywords.items():
        if any(w in msg_lower for w in words):
            key = k
            break
    
    return responses.get(lang, responses['en']).get(key, responses[lang]['default'])

# ============================================================================
# WEBSOCKET
# ============================================================================

@socketio.on('connect')
def on_connect():
    emit('sensor_update', sensor_data)

@socketio.on('disconnect')
def on_disconnect():
    pass

@socketio.on('request_data')
def on_request(data):
    try:
        dtype = data.get('type', 'sensor') if data else 'sensor'
        if dtype == 'sensor':
            emit('sensor_update', sensor_data)
        elif dtype == 'weather':
            emit('weather_update', WeatherService.get_weather())
        elif dtype == 'market':
            emit('market_update', MarketService.get_prices())
    except:
        pass

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    try:
        return render_template('404.html'), 404
    except:
        return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {e}")
    try:
        return render_template('500.html'), 500
    except:
        return jsonify({'error': 'Server error'}), 500

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': len(disease_detector.available_models),
        'db': 'ok' if DatabaseManager.get_connection() else 'error',
        'gemini_configured': bool(GEMINI_API_URL)
    })

# ============================================================================
# TEST ENDPOINTS
# ============================================================================

@app.route('/api/test-weather')
@login_required
def test_weather():
    """Test weather API"""
    try:
        weather = WeatherService.get_weather()
        return jsonify({
            'success': True,
            'weather': weather,
            'message': 'Weather API working (Open-Meteo - Free)'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-chat')
@login_required
def test_chat():
    """Test chatbot"""
    if not GOOGLE_API_KEY:
        return jsonify({
            'success': False,
            'error': 'GOOGLE_API_KEY not set',
            'message': 'Add GOOGLE_API_KEY to Render environment variables'
        })
    
    if not GEMINI_API_URL:
        return jsonify({
            'success': False,
            'error': 'GEMINI_API_URL not configured',
            'message': 'Check GOOGLE_API_KEY setup'
        })
    
    try:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Say hello in one word."
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 50
            }
        }
        
        response = requests.post(GEMINI_API_URL, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and result['candidates']:
                text = result['candidates'][0]['content']['parts'][0].get('text', '')
                return jsonify({
                    'success': True,
                    'message': 'Gemini API working',
                    'response': text
                })
        
        return jsonify({
            'success': False,
            'error': f'API returned status {response.status_code}',
            'details': response.text[:200]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Directory creation: {e}")
    
    try:
        if initialize_database():
            logger.info("Database ready")
    except Exception as e:
        logger.error(f"Database init error: {e}")
    
    if disease_detector.load_models():
        logger.info(f"Found {len(disease_detector.available_models)} models")
    else:
        logger.warning("No models found - using mock predictions")
    
    iot_sim.start()
    
    logger.info("=" * 60)
    logger.info("Krishi Sahyog Agricultural System")
    logger.info("=" * 60)
    logger.info(f"Models: {len(disease_detector.available_models)}")
    logger.info(f"Weather: Open-Meteo API (Free - No Key Needed)")
    logger.info(f"Chatbot: {'Gemini AI' if GEMINI_API_URL else 'Fallback Mode'}")
    logger.info(f"Test User: test@test.com / test123")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, use_reloader=False)
