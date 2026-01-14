"""
app.py - Main Backend Application for Krishi Sahyog
Refactored for Production Stability (TFLite, Render Deployment, Error Handling)
"""

from functools import wraps
import os
import logging
from datetime import datetime, timedelta
import random
import time
from threading import Thread
import re
import traceback
import json

# Flask & Extensions
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix # Critical for Render/Cloud deployment
from dotenv import load_dotenv

# Scientific & AI
import numpy as np
import requests
import joblib

# TFLite Runtime (Lightweight)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback if user accidentally installed full tensorflow
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("CRITICAL: Neither tflite_runtime nor tensorflow found.")
        tflite = None

# Custom Modules
from config import Config
from utils import ImageProcessor, CropDataAnalyzer
from database import DatabaseManager, initialize_database

# Load environment variables
load_dotenv()

# ============================================================================
# APP CONFIGURATION & SETUP
# ============================================================================

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Fix for Render/Heroku (Handle HTTPS headers behind proxy)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Session Configuration
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'krishi_sahyog_fallback_secret_key')
app.permanent_session_lifetime = timedelta(days=7)

# Cookie Security (Production vs Dev)
is_production = os.environ.get('FLASK_ENV') == 'production'
if is_production:
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax'
    )
else:
    app.config.update(
        SESSION_COOKIE_SECURE=False,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax'
    )

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# API Keys
api_key = os.getenv("GOOGLE_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in') or not session.get('user_id'):
            logger.info(f"Unauthorized access attempt to {f.__name__}")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# GLOBAL STATE
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
# AI MODELS (Fertilizer & Disease)
# ============================================================================

# Load Fertilizer Model
fertilizer_model = None
try:
    model_path = os.path.join(app.root_path, Config.MODEL_DIR, 'fertilizer_model.pkl')
    if os.path.exists(model_path):
        fertilizer_model = joblib.load(model_path)
        print("✓ Fertilizer recommendation model loaded successfully.")
    else:
        print(f"⚠ Fertilizer model not found at {model_path}")
except Exception as e:
    print(f"⚠ Error loading fertilizer model: {e}")

class EnhancedPlantDiseaseDetector:
    def __init__(self):
        self.interpreters = {}
        self.io_details = {}
        self.classes = {
            'wheat': ['HealthyLeaf', 'BlackPoint', 'LeafBlight', 'FusariumFootRot', 'WheatBlast'],
            'tomato': ['healthy', 'bacterial_spot', 'early_blight', 'late_blight',
                      'leaf_mold', 'septoria_leaf_spot', 'spider_mites',
                      'target_spot', 'mosaic_virus', 'yellow_leaf_curl'],
            'potato': ['Potato___healthy', 'Potato___Early_blight', 'Potato___late_blight'],
            'rice': ['healthy', 'bacterial_blight', 'brown_spot', 'leaf_smut']
        }
        
    def load_models(self):
        """Load TFLite models using tflite_runtime or tensorflow.lite"""
        if tflite is None:
            logger.error("TFLite library not installed.")
            return False

        models_dir = os.path.join(app.root_path, "models")
        
        model_files = {
            'wheat': 'Wheat_best_final_model.tflite',
            'tomato': 'tomato_model.tflite',
            'potato': 'potato_best_final_model.tflite',
            'rice': 'rice_best_final_model_fixed.tflite'
        }
        
        loaded_count = 0
        for plant, filename in model_files.items():
            path = os.path.join(models_dir, filename)
            
            if not os.path.exists(path):
                logger.warning(f"Model file missing: {path}")
                continue
            
            try:
                # Use the imported tflite alias (works for both runtime and full tf)
                interpreter = tflite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                
                self.interpreters[plant] = interpreter
                self.io_details[plant] = {
                    "input": interpreter.get_input_details(),
                    "output": interpreter.get_output_details()
                }
                loaded_count += 1
                logger.info(f"✓ {plant.capitalize()} model loaded.")
            except Exception as e:
                logger.error(f"Failed to load {plant} model: {e}")
        
        return loaded_count > 0

    def predict_disease(self, image_path, plant_type=None):
        if plant_type is None:
            plant_type = self._detect_plant_type(image_path)
        
        # Validation
        if plant_type not in self.classes:
            return self._get_mock_prediction('tomato', "Invalid Plant Type")

        # Mock if model not loaded
        if plant_type not in self.interpreters:
            logger.warning(f"Model for {plant_type} not loaded. Using Mock.")
            return self._get_mock_prediction(plant_type)

        try:
            # Preprocess
            processed_image = ImageProcessor.preprocess_for_ml(image_path)
            if processed_image is None:
                raise ValueError("Image preprocessing failed")

            # Inference
            interpreter = self.interpreters[plant_type]
            details = self.io_details[plant_type]
            
            # Ensure type compatibility (TFLite usually expects float32)
            input_data = processed_image.astype(np.float32)
            
            interpreter.set_tensor(details["input"][0]["index"], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(details["output"][0]["index"])
            
            # Post-process
            idx = np.argmax(predictions[0])
            confidence = float(predictions[0][idx])
            disease_name = self.classes[plant_type][idx]
            
            return {
                "plant_type": plant_type.capitalize(),
                "disease": disease_name.replace("_", " ").replace("Potato___", "").title(),
                "confidence": round(confidence * 100, 2),
                "treatment": self._get_treatment(disease_name),
                "severity": "High" if confidence > 0.8 else "Medium"
            }

        except Exception as e:
            logger.error(f"Prediction Error for {plant_type}: {e}")
            return self._get_mock_prediction(plant_type)

    def _detect_plant_type(self, image_path):
        name = image_path.lower()
        for plant in self.classes:
            if plant in name: return plant
        return 'tomato' # Default

    def _get_treatment(self, disease_name):
        # Simplified treatment lookup
        treatments = {
            'healthy': 'No pesticide needed. Maintain current care.',
            'blight': 'Use Copper fungicides or Mancozeb.',
            'spot': 'Use Copper-based bactericides.',
            'rust': 'Apply Sulphur dust or appropriate fungicide.',
            'virus': 'Remove infected plants immediately. Control vectors.'
        }
        for key, val in treatments.items():
            if key in disease_name.lower(): return val
        return "Consult a local agriculture expert."

    def _get_mock_prediction(self, plant_type, reason="Demo Mode"):
        disease = random.choice(self.classes.get(plant_type, ['Unknown']))
        return {
            "plant_type": plant_type.capitalize(),
            "disease": disease.replace("_", " ").title(),
            "confidence": round(random.uniform(75, 98), 2),
            "treatment": "Mock Data - Model not available.",
            "severity": "Medium"
        }

plant_detector = EnhancedPlantDiseaseDetector()

# ============================================================================
# SERVICES (Weather, Market, IoT)
# ============================================================================

class RealTimeWeatherService:
    @staticmethod
    def get_comprehensive_weather():
        # Fallback to mock if API key is missing or default
        if not Config.WEATHER_API_KEY or Config.WEATHER_API_KEY == 'your_openweather_api_key_here':
            return RealTimeWeatherService._get_mock_weather()
            
        try:
            url = f"{Config.WEATHER_BASE_URL}/weather"
            params = {
                'lat': Config.DEFAULT_LAT,
                'lon': Config.DEFAULT_LON,
                'appid': Config.WEATHER_API_KEY,
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': round(data['main']['temp'], 1),
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description'].title(),
                    'location': data.get('name', 'Unknown'),
                    'icon': data['weather'][0]['icon']
                }
        except Exception as e:
            logger.error(f"Weather API Error: {e}")
        
        return RealTimeWeatherService._get_mock_weather()

    @staticmethod
    def _get_mock_weather():
        return {
            'temperature': 28.5,
            'humidity': 72,
            'description': 'Partly Cloudy',
            'location': 'Kolkata (Demo)',
            'icon': '02d'
        }

class IoT_Simulator:
    def __init__(self):
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._run, daemon=True) # Daemon ensures it dies when app stops
            self.thread.start()
            print("✓ IoT Sensor simulation started")

    def _run(self):
        global sensor_data
        while self.running:
            # Simulate slight fluctuations
            sensor_data['soil_moisture'] = max(10, min(90, sensor_data['soil_moisture'] + random.randint(-2, 2)))
            sensor_data['soil_temperature'] = max(15, min(40, sensor_data['soil_temperature'] + random.uniform(-0.5, 0.5)))
            sensor_data['last_updated'] = datetime.now().isoformat()
            
            # Emit via SocketIO
            try:
                socketio.emit('sensor_update', sensor_data)
            except:
                pass 
            
            time.sleep(5)

iot_sim = IoT_Simulator()

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('landing'))

@app.route('/landing')
def landing():
    return render_template('landing.html')

# Authentication Routes

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            
            # Validation
            if not username or not email or not password:
                return render_template('register.html', error="All fields are required.")
            
            if len(password) < 6:
                return render_template('register.html', error="Password too short (min 6 chars).")
            
            # Database Call
            user_id = DatabaseManager.create_user(username, email, password)
            
            if user_id:
                logger.info(f"User Registered: {email} (ID: {user_id})")
                # SUCCESS: Redirect to LOGIN with success message
                return render_template('login.html', success="Registration successful! Please login.")
            else:
                # FAILURE: User likely exists
                return render_template('register.html', error="Email or Username already exists.")
                
        except Exception as e:
            logger.error(f"Registration Exception: {e}")
            return render_template('register.html', error="System error during registration.")
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        try:
            user = DatabaseManager.get_user_by_email(email)
            if user and check_password_hash(user[3], password):
                session.clear()
                session['logged_in'] = True
                session['user_id'] = user[0]
                session['username'] = user[1]
                session.permanent = True
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error="Invalid email or password.")
        except Exception as e:
            logger.error(f"Login Error: {e}")
            return render_template('login.html', error="Login failed. Try again.")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

# Main Pages

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
    # CASE SENSITIVITY FIX: Ensure file is named exactly 'Chatbot.html' in templates/
    try:
        return render_template('Chatbot.html')
    except Exception:
        # Fallback if user renamed it to lowercase
        return render_template('chatbot.html')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/upload-image', methods=['POST'])
@login_required
def upload_image_for_diagnosis():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    plant_type = request.form.get('plant_type', 'tomato')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    try:
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Predict
        result = plant_detector.predict_disease(filepath, plant_type)
        
        # Save to DB (Fire and forget, don't crash if DB fails)
        try:
            DatabaseManager.save_diagnosis(
                session['user_id'], result['plant_type'], result['disease'], 
                result['confidence'], filepath
            )
        except Exception as e:
            logger.error(f"DB Save Error: {e}")

        # Clean up file
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        logger.error(f"Upload API Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': 'Analysis failed'}), 500

@app.route('/api/sensor-data')
@login_required
def get_sensor_data():
    return jsonify(sensor_data)

@app.route('/api/weather-data')
@login_required
def get_weather_data():
    return jsonify(RealTimeWeatherService.get_comprehensive_weather())

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """Gemini-powered Chatbot Endpoint"""
    data = request.json
    user_msg = data.get('message', '')
    
    if not user_msg:
        return jsonify({'text': "Please say something."})
    
    # 1. Check API Key
    if not api_key or "your_google" in api_key:
        return jsonify({'text': "Chatbot is running in Demo mode (API Key missing). I can only answer basic queries."})

    # 2. Call Gemini
    try:
        payload = {
            "contents": [{
                "parts": [{"text": f"You are Krishi Sahyog, an expert Indian agriculture assistant. Be concise. Answer this: {user_msg}"}]
            }]
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            ai_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            return jsonify({'text': ai_text})
        else:
            logger.error(f"Gemini API Error: {response.text}")
            return jsonify({'text': "I am having trouble connecting to the server. Please try again."})
            
    except Exception as e:
        logger.error(f"Chat Exception: {e}")
        return jsonify({'text': "Sorry, something went wrong with the AI service."})

# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'models'), exist_ok=True)
    
    # Init DB
    if not initialize_database():
        print("⚠ Database init failed. Some features may not work.")
    
    # Load Models
    plant_detector.load_models()
    
    # Start IoT
    iot_sim.start()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Krishi Sahyog Server Starting on Port {port}")
    
    # Use allow_unsafe_werkzeug only for local dev/SocketIO
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
