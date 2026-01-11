"""
database.py - PostgreSQL Database operations for Krishi Sahyog
Handles all database connections, table creation, and data operations
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime
from werkzeug.security import generate_password_hash
import os

# Database configuration from environment variable (Render provides this)
DATABASE_URL = os.environ.get('DATABASE_URL')

class DatabaseManager:
    """Centralized database management for PostgreSQL"""
    
    @staticmethod
    def get_connection():
        """Get PostgreSQL database connection with proper error handling"""
        try:
            # Render uses DATABASE_URL, but some services use postgres:// which needs to be postgresql://
            db_url = DATABASE_URL
            if db_url and db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql://', 1)
            
            conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
            return conn
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            return None
    
    @staticmethod
    def create_tables():
        """Create all necessary database tables"""
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                raise Exception("Failed to get database connection")
            
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) NOT NULL UNIQUE,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Sensor readings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id SERIAL PRIMARY KEY,
                    soil_ph REAL,
                    soil_moisture INTEGER,
                    soil_temperature REAL,
                    nitrogen INTEGER,
                    phosphorus INTEGER,
                    potassium INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Crop data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crop_data (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    crop_name VARCHAR(255),
                    plant_date DATE,
                    expected_harvest DATE,
                    area REAL,
                    status VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Disease diagnosis history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diagnosis_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    plant_type VARCHAR(100),
                    disease VARCHAR(255),
                    confidence REAL,
                    image_path TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Chat history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    message TEXT,
                    response TEXT,
                    language VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            cursor.close()
            conn.close()
            logging.info("Database tables created successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error creating tables: {e}")
            if conn:
                conn.close()
            return False
    
    @staticmethod
    def create_user(username, email, password):
        """Create a new user"""
        conn = None
        try:
            hashed_password = generate_password_hash(password)
            conn = DatabaseManager.get_connection()
            if not conn:
                return None
                
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password) 
                VALUES (%s, %s, %s)
                RETURNING id
            ''', (username, email, hashed_password))
            
            user_id = cursor.fetchone()['id']
            conn.commit()
            cursor.close()
            conn.close()
            return user_id
            
        except psycopg2.IntegrityError as e:
            logging.error(f"User already exists: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return None
        except Exception as e:
            logging.error(f"Error creating user: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return None
    
    @staticmethod
    def get_user_by_email(email):
        """Get user by email"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection")
                return None
            
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password 
                FROM users 
                WHERE LOWER(email) = LOWER(%s)
            ''', (email,))
            
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user:
                logging.info(f"User found: {email}")
                # Convert to tuple for compatibility with existing code
                return (user['id'], user['username'], user['email'], user['password'])
            else:
                logging.info(f"No user found with email: {email}")
                return None
            
        except Exception as e:
            logging.error(f"Error getting user by email: {e}")
            if conn:
                conn.close()
            return None
    
    @staticmethod
    def update_last_login(user_id):
        """Update user's last login timestamp"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return False
                
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET last_login = %s 
                WHERE id = %s
            ''', (datetime.now(), user_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error updating last login: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    @staticmethod
    def save_sensor_reading(soil_ph, soil_moisture, soil_temperature, 
                           nitrogen, phosphorus, potassium):
        """Save sensor reading to database"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return False
                
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sensor_readings 
                (soil_ph, soil_moisture, soil_temperature, nitrogen, phosphorus, potassium)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (soil_ph, soil_moisture, soil_temperature, 
                  nitrogen, phosphorus, potassium))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error saving sensor reading: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    @staticmethod
    def get_recent_sensor_readings(limit=10):
        """Get recent sensor readings"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return []
                
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM sensor_readings 
                ORDER BY timestamp DESC 
                LIMIT %s
            ''', (limit,))
            
            readings = cursor.fetchall()
            cursor.close()
            conn.close()
            return readings
            
        except Exception as e:
            logging.error(f"Error getting sensor readings: {e}")
            if conn:
                conn.close()
            return []
    
    @staticmethod
    def save_diagnosis(user_id, plant_type, disease, confidence, image_path):
        """Save disease diagnosis to history"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return False
                
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO diagnosis_history 
                (user_id, plant_type, disease, confidence, image_path)
                VALUES (%s, %s, %s, %s, %s)
            ''', (user_id, plant_type, disease, confidence, image_path))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error saving diagnosis: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    @staticmethod
    def get_user_diagnosis_history(user_id, limit=20):
        """Get user's diagnosis history"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return []
                
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM diagnosis_history 
                WHERE user_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
            ''', (user_id, limit))
            
            history = cursor.fetchall()
            cursor.close()
            conn.close()
            return history
            
        except Exception as e:
            logging.error(f"Error getting diagnosis history: {e}")
            if conn:
                conn.close()
            return []
    
    @staticmethod
    def save_chat_message(user_id, message, response, language):
        """Save chat interaction"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return False
                
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history 
                (user_id, message, response, language)
                VALUES (%s, %s, %s, %s)
            ''', (user_id, message, response, language))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error saving chat message: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    @staticmethod
    def create_test_user():
        """Create test user for development"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return False
                
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM users WHERE email = %s', ('test@test.com',))
            if not cursor.fetchone():
                hashed_password = generate_password_hash('test123')
                cursor.execute('''
                    INSERT INTO users (username, email, password) VALUES (%s, %s, %s)
                ''', ('testuser', 'test@test.com', hashed_password))
                conn.commit()
                logging.info("Test user created: test@test.com / test123")
            else:
                logging.info("Test user already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error creating test user: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False


# Initialize database on module import
def initialize_database():
    """Initialize database with tables and test data"""
    try:
        logging.info("Initializing PostgreSQL database...")
        
        if not DATABASE_URL:
            logging.error("DATABASE_URL environment variable not set!")
            return False
        
        if DatabaseManager.create_tables():
            logging.info("Database tables created successfully")
            DatabaseManager.create_test_user()
            return True
        else:
            logging.error("Failed to create database tables")
            return False
            
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        return False
