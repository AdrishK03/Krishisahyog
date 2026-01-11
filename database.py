"""
database.py - Database operations for Krishi Sahyog
Handles all database connections, table creation, and data operations
"""

import sqlite3
import logging
from datetime import datetime
from werkzeug.security import generate_password_hash

# Database configuration
DATABASE_NAME = 'krishi_sahyog.db'

class DatabaseManager:
    """Centralized database management"""
    
    @staticmethod
    def get_connection():
        """Get database connection with proper error handling"""
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
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
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Sensor readings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    crop_name TEXT,
                    plant_date DATE,
                    expected_harvest DATE,
                    area REAL,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Disease diagnosis history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diagnosis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    plant_type TEXT,
                    disease TEXT,
                    confidence REAL,
                    image_path TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Chat history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    message TEXT,
                    response TEXT,
                    language TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Database tables created successfully")
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Error creating tables: {e}")
            return False
    
    @staticmethod
    def create_user(username, email, password):
        """Create a new user"""
        try:
            # FIX: Removed explicit method to allow default secure hashing
            hashed_password = generate_password_hash(password)
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password) 
                VALUES (?, ?, ?)
            ''', (username, email, hashed_password))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return user_id
            
        except sqlite3.IntegrityError:
            return None
        except Exception as e:
            logging.error(f"Error creating user: {e}")
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
                WHERE LOWER(email) = ?
            ''', (email.lower(),))
            
            user = cursor.fetchone()
            
            if user:
                logging.info(f"User found: {email}")
            else:
                logging.info(f"No user found with email: {email}")
            
            return user
            
        except Exception as e:
            logging.error(f"Error getting user by email: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def update_last_login(user_id):
        """Update user's last login timestamp"""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET last_login = ? 
                WHERE id = ?
            ''', (datetime.now(), user_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error updating last login: {e}")
            return False
    
    @staticmethod
    def save_sensor_reading(soil_ph, soil_moisture, soil_temperature, 
                           nitrogen, phosphorus, potassium):
        """Save sensor reading to database"""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sensor_readings 
                (soil_ph, soil_moisture, soil_temperature, nitrogen, phosphorus, potassium)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (soil_ph, soil_moisture, soil_temperature, 
                  nitrogen, phosphorus, potassium))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error saving sensor reading: {e}")
            return False
    
    @staticmethod
    def get_recent_sensor_readings(limit=10):
        """Get recent sensor readings"""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM sensor_readings 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            readings = cursor.fetchall()
            conn.close()
            return readings
            
        except Exception as e:
            logging.error(f"Error getting sensor readings: {e}")
            return []
    
    @staticmethod
    def save_diagnosis(user_id, plant_type, disease, confidence, image_path):
        """Save disease diagnosis to history"""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO diagnosis_history 
                (user_id, plant_type, disease, confidence, image_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, plant_type, disease, confidence, image_path))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error saving diagnosis: {e}")
            return False
    
    @staticmethod
    def get_user_diagnosis_history(user_id, limit=20):
        """Get user's diagnosis history"""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM diagnosis_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            history = cursor.fetchall()
            conn.close()
            return history
            
        except Exception as e:
            logging.error(f"Error getting diagnosis history: {e}")
            return []
    
    @staticmethod
    def save_chat_message(user_id, message, response, language):
        """Save chat interaction"""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history 
                (user_id, message, response, language)
                VALUES (?, ?, ?, ?)
            ''', (user_id, message, response, language))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error saving chat message: {e}")
            return False
    
    @staticmethod
    def create_test_user():
        """Create test user for development"""
        try:
            conn = DatabaseManager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM users WHERE email = ?', ('test@test.com',))
            if not cursor.fetchone():
                # FIX: Removed explicit method for test user too
                hashed_password = generate_password_hash('test123')
                cursor.execute('''
                    INSERT INTO users (username, email, password) VALUES (?, ?, ?)
                ''', ('testuser', 'test@test.com', hashed_password))
                conn.commit()
                logging.info("Test user created: test@test.com / test123")
            else:
                logging.info("Test user already exists")
            
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"Error creating test user: {e}")
            return False


# Initialize database on module import
def initialize_database():
    """Initialize database with tables and test data"""
    try:
        logging.info("Initializing database...")
        
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