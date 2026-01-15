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
import numpy as np
import re
import traceback
import joblib
from PIL import Image

# Import TFLite interpreter - ONLY tflite_runtime (lightweight)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    logging.warning("tflite_runtime not available - using mock predictions only")
    TFLITE_AVAILABLE = False
    tflite = None

from config import Config
from database import DatabaseManager, initialize_database

# Load environment variables
load_dotenv()

# Set up logging
log_level = logging.INFO if os.environ.get('FLASK_ENV') == 'production' else logging.DEBUG
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
import os
WEATHER_BASE_URL = os.getenv("WEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
MARKET_BASE_URL = os.getenv("MARKET_BASE_URL", "https://api.marketdata.com/v1")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set - chatbot will use fallback responses")

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Session configuration
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24).hex())
app.permanent_session_lifetime = timedelta(days=7)

# Production vs Development settings
IS_PRODUCTION = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_SECURE'] = IS_PRODUCTION
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# CORS configuration
CORS(app, supports_credentials=True)

# SocketIO configuration
socketio = SocketIO(app, cors_allowed_origins="*")

# Gemini API URL
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}" if GOOGLE_API_KEY else None

# ============================================================================
# UTILITY CLASSES
# ============================================================================

class ImageProcessor:
    @staticmethod
    def preprocess_for_ml(image_path, target_size=(224, 224)):
        """Preprocess image for ML model"""
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None

class CropDataAnalyzer:
    @staticmethod
    def analyze_soil_conditions(ph, moisture, temperature):
        """Analyze soil conditions"""
        analysis = {
            'ph_status': 'optimal' if 6.0 <= ph <= 7.5 else 'needs_adjustment',
            'moisture_status': 'good' if 40 <= moisture <= 80 else 'needs_attention',
            'temperature_status': 'suitable' if 15 <= temperature <= 35 else 'extreme',
            'recommendations': []
        }
        
        if ph < 6.0:
            analysis['recommendations'].append("Soil is acidic - consider adding lime")
        elif ph > 7.5:
            analysis['recommendations'].append("Soil is alkaline - consider adding sulfur")
        
        if moisture < 40:
            analysis['recommendations'].append("Increase irrigation frequency")
        elif moisture > 80:
            analysis['recommendations'].append("Reduce watering - improve drainage")
        
        if temperature < 15:
            analysis['recommendations'].append("Temperature low - protect sensitive crops")
        elif temperature > 35:
            analysis['recommendations'].append("High temperature - provide shade and increase watering")
        
        return analysis

# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in') or not session.get('user_id'):
            logger.debug(f"Unauthorized access to {f.__name__}")
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
# PLANT DISEASE DETECTION
# ============================================================================

class PlantDiseaseDetector:
    def __init__(self):
        self.models = {}
        self.model_details = {}
        self.available_models = []
        
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
        """Load TFLite models lazily - only load one at a time to save memory"""
        if not TFLITE_AVAILABLE:
            logger.warning("TFLite not available - cannot load models")
            return False
        
        models_dir = Config.MODEL_DIR
        model_files = {
            'wheat': 'Wheat_best_final_model.tflite',
            'tomato': 'tomato_model.tflite',
            'potato': 'potato_best_final_model.tflite',
            'rice': 'rice_best_final_model_fixed.tflite'
        }
        
        # Only check if files exist, don't load all into memory
        available_models = []
        for plant_type, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                available_models.append(plant_type)
                logger.info(f"✓ Found {plant_type} model")
            else:
                logger.warning(f"✗ {plant_type} model not found: {model_path}")
        
        self.available_models = available_models
        return len(available_models) > 0

    def _load_model_on_demand(self, plant_type):
        """Load model only when needed to save memory"""
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
            model_path = os.path.join(Config.MODEL_DIR, model_files[plant_type])
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            self.models[plant_type] = interpreter
            self.model_details[plant_type] = {
                'input': interpreter.get_input_details(),
                'output': interpreter.get_output_details()
            }
            
            logger.info(f"✓ Loaded {plant_type} model on demand")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {plant_type} model: {e}")
            return False

    def predict(self, image_path, plant_type=None):
        """Predict disease from image"""
        if plant_type is None:
            plant_type = self._detect_plant_type(image_path)
        
        if plant_type not in self.classes:
            plant_type = 'tomato'  # Default
        
        # Preprocess image
        processed_image = ImageProcessor.preprocess_for_ml(image_path)
        if processed_image is None:
            return self._mock_prediction(plant_type)
        
        # Load model on demand
        if self._load_model_on_demand(plant_type):
            try:
                interpreter = self.models[plant_type]
                details = self.model_details[plant_type]
                
                # Set input
                interpreter.set_tensor(details['input'][0]['index'], processed_image)
                
                # Run inference
                interpreter.invoke()
                
                # Get output
                output = interpreter.get_tensor(details['output'][0]['index'])
                
                # Get prediction
                predicted_idx = np.argmax(output[0])
                confidence = float(output[0][predicted_idx])
                disease = self.classes[plant_type][predicted_idx]
                
                treatment_key = f"{plant_type}_{disease}"
                treatment = self.treatments.get(treatment_key, "Consult agricultural expert")
                
                # Unload model to free memory
                if len(self.models) > 1:
                    del self.models[plant_type]
                    del self.model_details[plant_type]
                
                return {
                    'plant_type': plant_type.capitalize(),
                    'disease': disease.replace('_', ' ').replace('___', ' ').title(),
                    'confidence': round(confidence * 100, 2),
                    'treatment': treatment,
                    'severity': self._get_severity(confidence),
                    'recommendations': self._get_recommendations(disease)
                }
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return self._mock_prediction(plant_type)
        
        # Fallback to mock
        return self._mock_prediction(plant_type)

    def _detect_plant_type(self, image_path):
        """Detect plant type from filename"""
        filename = os.path.basename(image_path).lower()
        for plant in self.classes.keys():
            if plant in filename:
                return plant
        return 'tomato'

    def _get_severity(self, confidence):
        """Get severity level"""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.5:
            return 'medium'
        return 'low'

    def _get_recommendations(self, disease):
        """Get recommendations"""
        recs = ['Monitor plant regularly', 'Maintain field hygiene']
        if 'healthy' not in disease.lower():
            recs.extend(['Apply treatment promptly', 'Isolate affected plants'])
        return recs

    def _mock_prediction(self, plant_type):
        """Mock prediction for testing"""
        disease = random.choice(self.classes.get(plant_type, self.classes['tomato']))
        confidence = random.uniform(75, 95)
        return {
            'plant_type': plant_type.capitalize(),
            'disease': disease.replace('_', ' ').title(),
            'confidence': round(confidence, 2),
            'treatment': 'Demo mode - upload real image for actual diagnosis',
            'severity': 'medium',
            'recommendations': ['This is a demo prediction']
        }

# Initialize detector
disease_detector = PlantDiseaseDetector()

# ============================================================================
# WEATHER SERVICE
# ============================================================================

class WeatherService:
    @staticmethod
    def get_weather(lat=None, lon=None):
        """Fetch weather data using the custom WEATHER_BASE_URL"""
        if lat is None:
            lat = Config.DEFAULT_LAT
        if lon is None:
            lon = Config.DEFAULT_LON
        
        # Ensure your WEATHER_API_KEY is also set in Render Environment
        api_key = os.getenv("WEATHER_API_KEY", "").strip()
        
        if api_key and api_key != 'your_openweather_api_key_here':
            try:
                # Constructing the URL dynamically based on environment config
                url = f"{WEATHER_URL}/weather"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': api_key,
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
                        'location': data.get('name', Config.DEFAULT_LOCATION)
                    }
                else:
                    logger.error(f"Weather API returned status: {response.status_code}")
            except Exception as e:
                logger.debug(f"Weather API request failed: {e}")
        
        # Fallback to simulated data if API fails or key is missing
        return WeatherService._mock_weather()

    @staticmethod
    def _mock_weather():
        """Simulated weather for testing/fallback"""
        month = datetime.now().month
        temp, humid = (20, 65) if month in [12, 1, 2] else (32, 70) if month in [3, 4, 5] else (28, 85) if month in [6, 7, 8, 9] else (26, 75)
        return {
            'temperature': round(temp + random.uniform(-3, 3), 1),
            'humidity': int(humid + random.randint(-10, 10)),
            'description': 'Partly Cloudy',
            'wind_speed': round(random.uniform(2, 8), 1),
            'location': Config.DEFAULT_LOCATION
        }

class MarketService:
    @staticmethod
    def get_prices():
        """Market price logic using MARKET_BASE_URL if needed"""
        # Note: If using a real API, call f"{MARKET_URL}/prices"
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
                # Update values
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
                
                # Save to DB
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
                
                # Emit via socket
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
            
            # ✅ FIXED: Unpack tuple correctly
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
            
            # Validation
            if not all([username, email, password]):
                return render_template('register.html', error="All fields required")
            
            if len(password) < 6:
                return render_template('register.html', error="Password must be 6+ characters")
            
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return render_template('register.html', error="Invalid email")
            
            # Create user
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
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Predict
        result = disease_detector.predict(filepath, plant_type)
        
        # Save to DB
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
        
        # Cleanup
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
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
        lang = data.get('lang', 'en-IN').split('-')[0]
        
        # Try Gemini API
        if GEMINI_API_URL:
            try:
                payload = {
                    "contents": [{"parts": [{"text": f"You are Krishi Sahyog agricultural assistant. Respond in {'Hindi' if lang == 'hi' else 'Bengali' if lang == 'bn' else 'English'}.\n\n{user_message}"}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024}
                }
                
                response = requests.post(GEMINI_API_URL, json=payload, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and result['candidates']:
                        text = result['candidates'][0]['content']['parts'][0].get('text', '')
                        if text:
                            try:
                                DatabaseManager.save_chat_message(session['user_id'], user_message, text, lang)
                            except:
                                pass
                            return jsonify({'text': text, 'audio': ''})
            except:
                pass
        
        # Fallback
        fallback = get_fallback_response(user_message, lang)
        try:
            DatabaseManager.save_chat_message(session['user_id'], user_message, fallback, lang)
        except:
            pass
        
        return jsonify({'text': fallback, 'audio': ''})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Chat unavailable'}), 500

def get_fallback_response(message, lang='en'):
    """Simple fallback responses"""
    msg_lower = message.lower()
    
    responses = {
        'en': {
            'weather': "Check the weather section for current conditions and forecast.",
            'price': "Visit the market section for latest crop prices.",
            'soil': f"Current soil: pH {sensor_data['soil_ph']:.1f}, Moisture {sensor_data['soil_moisture']}%",
            'disease': "Upload a plant image in the diagnosis section for disease detection.",
            'default': "I can help with weather, prices, soil analysis, and disease diagnosis. What would you like to know?"
        },
        'hi': {
            'weather': "मौसम अनुभाग में वर्तमान स्थिति देखें।",
            'price': "बाजार अनुभाग में नवीनतम फसल मूल्य देखें।",
            'soil': f"वर्तमान मिट्टी: pH {sensor_data['soil_ph']:.1f}, नमी {sensor_data['soil_moisture']}%",
            'disease': "रोग पहचान के लिए निदान अनुभाग में पौधे की छवि अपलोड करें।",
            'default': "मैं मौसम, कीमतें, मिट्टी विश्लेषण में मदद कर सकता हूं। आप क्या जानना चाहेंगे?"
        },
        'bn': {
            'weather': "আবহাওয়া বিভাগে বর্তমান অবস্থা দেখুন।",
            'price': "বাজার বিভাগে সর্বশেষ ফসলের দাম দেখুন।",
            'soil': f"বর্তমান মাটি: pH {sensor_data['soil_ph']:.1f}, আর্দ্রতা {sensor_data['soil_moisture']}%",
            'disease': "রোগ নির্ণয়ের জন্য ডায়াগনসিস বিভাগে ছবি আপলোড করুন।",
            'default': "আমি আবহাওয়া, দাম, মাটি বিশ্লেষণে সাহায্য করতে পারি। আপনি কী জানতে চান?"
        }
    }
    
    keywords = {'weather': ['weather', 'मौसम', 'আবহাওয়া'], 'price': ['price', 'भाव', 'দাম'], 
                'soil': ['soil', 'मिट्टी', 'মাটি'], 'disease': ['disease', 'रोग', 'রোগ']}
    
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
        'google_api_key_set': bool(GOOGLE_API_KEY and GOOGLE_API_KEY != ''),
        'weather_api_key_set': bool(WEATHER_API_KEY and WEATHER_API_KEY != ''),
        'gemini_url_configured': bool(GEMINI_API_URL)
    })

@app.route('/api/test-gemini')
@login_required
def test_gemini():
    """Test Gemini API connection"""
    if not GEMINI_API_URL:
        return jsonify({
            'success': False,
            'error': 'GOOGLE_API_KEY not set',
            'api_key_set': bool(GOOGLE_API_KEY)
        })
    
    try:
        payload = {
            "contents": [{
                "parts": [{
                    "text": "Say 'Hello from Krishi Sahyog' in one sentence."
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 100
            }
        }
        
        response = requests.post(GEMINI_API_URL, json=payload, timeout=10)
        
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
            'response': response.text[:200]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Create directories
    try:
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Directory creation: {e}")
    
    # Initialize database
    try:
        if initialize_database():
            logger.info("✓ Database ready")
    except Exception as e:
        logger.error(f"Database init error: {e}")
    
    # Load models
    if disease_detector.load_models():
        logger.info(f"✓ Found {len(disease_detector.available_models)} models (lazy loading)")
    else:
        logger.warning("⚠ No models found - using mock predictions")
    
    # Start IoT
    iot_sim.start()
    
    # Display info
    logger.info("=" * 50)
    logger.info("🌱 Krishi Sahyog Agricultural System")
    logger.info("=" * 50)
    logger.info(f"Models: {len(disease_detector.available_models)}")
    logger.info(f"Chatbot: {'Gemini' if GEMINI_API_URL else 'Fallback'}")
    logger.info(f"Test: test@test.com / test123")
    logger.info("=" * 50)
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    
    # Always run in production mode on Render
    logger.info(f"Starting on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, use_reloader=False)
