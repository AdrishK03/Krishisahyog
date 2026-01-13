"""
config.py - Configuration settings for Krishi Sahyog
Environment-based configuration for different deployment stages
"""

import os
from datetime import timedelta
from pathlib import Path


class Config:
    """Base configuration class"""
    
    # ============================================================================
    # CORE CONFIGURATION
    # ============================================================================
    
    # Secret key for session management and CSRF protection
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or 'krishi_sahyog_secret_key_2024_change_in_production'
    
    # Environment detection
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = FLASK_ENV == 'development'
    TESTING = False
    
    # ============================================================================
    # DATABASE CONFIGURATION
    # ============================================================================
    
    # PostgreSQL Database (Render provides DATABASE_URL)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Convert postgres:// to postgresql:// if needed (Render compatibility)
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    
    # ============================================================================
    # FILE UPLOAD CONFIGURATION
    # ============================================================================
    
    # Upload folder for disease diagnosis images
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    
    # Maximum file size (16MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    # Allowed file extensions for upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # ============================================================================
    # API KEYS AND EXTERNAL SERVICES
    # ============================================================================
    
    # Google Gemini API Key
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    
    # OpenWeather API Key
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
    
    # Market Data API Key (data.gov.in)
    MARKET_API_KEY = os.environ.get('MARKET_API_KEY')
    
    # ============================================================================
    # CROP AND DISEASE CONFIGURATION
    # ============================================================================
    
    # Supported crop types for disease detection
    # Must match the TFLite model files available
    CROP_TYPES = ['wheat', 'tomato', 'potato', 'rice']
    
    # Disease model files (TFLite format for production)
    DISEASE_MODELS = {
        'wheat': 'Wheat_best_final_model.tflite',
        'tomato': 'tomato_model.tflite',
        'potato': 'potato_best_final_model.tflite',
        'rice': 'rice_best_final_model_fixed.tflite'
    }
    
    # ============================================================================
    # IOT SENSOR CONFIGURATION
    # ============================================================================
    
    # IoT sensor update interval (seconds)
    IOT_UPDATE_INTERVAL = 30
    
    # Default sensor ranges for validation
    SENSOR_RANGES = {
        'soil_ph': (5.0, 8.5),
        'soil_moisture': (15, 95),
        'soil_temperature': (10, 40),
        'nitrogen': (20, 80),
        'phosphorus': (15, 60),
        'potassium': (25, 70)
    }
    
    # ============================================================================
    # LANGUAGE CONFIGURATION
    # ============================================================================
    
    # Supported languages for the application
    LANGUAGES = {
        'en': 'English',
        'hi': 'Hindi',
        'bn': 'Bengali'
    }
    
    # Default language
    DEFAULT_LANGUAGE = 'en'
    
    # Language codes for API calls
    LANGUAGE_CODES = {
        'en': 'en-IN',
        'hi': 'hi-IN',
        'bn': 'bn-IN'
    }
    
    # ============================================================================
    # SESSION CONFIGURATION
    # ============================================================================
    
    # Session lifetime (7 days)
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    
    # Session cookie configuration
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_REFRESH_EACH_REQUEST = True
    
    # ============================================================================
    # EXTERNAL API ENDPOINTS
    # ============================================================================
    
    # OpenWeather API base URL
    WEATHER_BASE_URL = 'https://api.openweathermap.org/data/2.5'
    
    # Government market data API base URL
    # Note: Requires authentication and specific dataset resource IDs
    MARKET_BASE_URL = 'https://api.data.gov.in/resource'
    
    # Google Gemini API endpoint
    GEMINI_API_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'
    
    # ============================================================================
    # DEFAULT LOCATION (WEST BENGAL, INDIA)
    # ============================================================================
    
    # Default latitude (Kolkata, West Bengal)
    DEFAULT_LAT = 22.1667
    
    # Default longitude (Kolkata, West Bengal)
    DEFAULT_LON = 88.1833
    
    # Default location name
    DEFAULT_LOCATION = 'Kolkata, West Bengal'
    
    # ============================================================================
    # MODEL PATHS
    # ============================================================================
    
    # Base directory for ML models
    MODEL_DIR = os.path.join(os.getcwd(), 'models')
    
    # TFLite model paths (production)
    WHEAT_MODEL_PATH = os.path.join(MODEL_DIR, 'Wheat_best_final_model.tflite')
    TOMATO_MODEL_PATH = os.path.join(MODEL_DIR, 'tomato_model.tflite')
    POTATO_MODEL_PATH = os.path.join(MODEL_DIR, 'potato_best_final_model.tflite')
    RICE_MODEL_PATH = os.path.join(MODEL_DIR, 'rice_best_final_model_fixed.tflite')
    
    # Fertilizer recommendation model
    FERTILIZER_MODEL_PATH = os.path.join(MODEL_DIR, 'fertilizer_model.pkl')
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    # Log level
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # ============================================================================
    # SECURITY CONFIGURATION
    # ============================================================================
    
    # Minimum password length
    MIN_PASSWORD_LENGTH = 6
    
    # Maximum login attempts (for future rate limiting)
    MAX_LOGIN_ATTEMPTS = 5
    
    # Login timeout duration in minutes
    LOGIN_TIMEOUT = 15
    
    # ============================================================================
    # PAGINATION AND LIMITS
    # ============================================================================
    
    # Items per page for paginated results
    ITEMS_PER_PAGE = 20
    
    # Maximum sensor readings to retrieve
    MAX_SENSOR_READINGS = 100
    
    # Maximum diagnosis history records to retrieve
    MAX_DIAGNOSIS_HISTORY = 50
    
    # ============================================================================
    # SOCKET.IO CONFIGURATION
    # ============================================================================
    
    # Socket.IO async mode
    SOCKETIO_ASYNC_MODE = 'threading'
    
    # CORS allowed origins for Socket.IO
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    
    # ============================================================================
    # PERFORMANCE OPTIMIZATION
    # ============================================================================
    
    # Enable caching (for future implementation)
    CACHE_ENABLED = os.environ.get('CACHE_ENABLED', 'true').lower() == 'true'
    
    # Cache timeout in seconds
    CACHE_TIMEOUT = 300
    
    # ============================================================================
    # PRODUCTION-SPECIFIC SETTINGS
    # ============================================================================
    
    if FLASK_ENV == 'production':
        # Production mode settings
        PROPAGATE_EXCEPTIONS = True
        PRESERVE_CONTEXT_ON_EXCEPTION = False
        
        # Stricter session settings
        SESSION_COOKIE_SECURE = True
        SESSION_COOKIE_HTTPONLY = True
        
        # Disable debug toolbar
        DEBUG = False
        DEBUG_TB_ENABLED = False
    
    else:
        # Development mode settings
        PROPAGATE_EXCEPTIONS = True
        PRESERVE_CONTEXT_ON_EXCEPTION = True
        DEBUG_TB_ENABLED = True
    
    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================
    
    @staticmethod
    def validate_configuration():
        """Validate critical configuration settings"""
        errors = []
        
        # Check DATABASE_URL
        if not Config.DATABASE_URL:
            errors.append("DATABASE_URL environment variable is not set")
        
        # Check required directories
        try:
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create upload folder: {e}")
        
        # Check model directory
        try:
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create model directory: {e}")
        
        return errors
    
    @staticmethod
    def get_language_code(lang):
        """Get language code for API calls"""
        return Config.LANGUAGE_CODES.get(lang, 'en-IN')
    
    @staticmethod
    def is_allowed_file(filename):
        """Check if file extension is allowed"""
        if not filename or '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in Config.ALLOWED_EXTENSIONS


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'postgresql://localhost/krishi_test'
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True


# Configuration factory function
def get_config(env=None):
    """Get configuration object based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'testing': TestingConfig,
        'production': ProductionConfig
    }
    
    return config_map.get(env, DevelopmentConfig)
