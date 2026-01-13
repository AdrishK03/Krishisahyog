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
            if db_url is None:
                logging.error("DATABASE_URL not set in environment variables")
                return None
            
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql://', 1)
            
            conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
            logging.debug("Database connection established successfully")
            return conn
        except psycopg2.OperationalError as e:
            logging.error(f"Database connection error (OperationalError): {e}")
            return None
        except psycopg2.Error as e:
            logging.error(f"Database connection error (psycopg2 Error): {e}")
            return None
        except Exception as e:
            logging.error(f"Database connection error (General): {e}")
            return None
    
    @staticmethod
    def create_tables():
        """Create all necessary database tables"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for table creation")
                return False
            
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
            logging.info("Users table created/verified")
            
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
            logging.info("Sensor readings table created/verified")
            
            # Crop data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crop_data (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    crop_name VARCHAR(255),
                    plant_date DATE,
                    expected_harvest DATE,
                    area REAL,
                    status VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            logging.info("Crop data table created/verified")
            
            # Disease diagnosis history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diagnosis_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    plant_type VARCHAR(100),
                    disease VARCHAR(255),
                    confidence REAL,
                    image_path TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            logging.info("Diagnosis history table created/verified")
            
            # Chat history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    message TEXT,
                    response TEXT,
                    language VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            logging.info("Chat history table created/verified")
            
            conn.commit()
            cursor.close()
            logging.info("All database tables created successfully")
            return True
            
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error creating tables: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logging.error(f"Unexpected error creating tables: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def create_user(username, email, password):
        """Create a new user"""
        conn = None
        try:
            if not username or not email or not password:
                logging.warning("Missing required fields for user creation")
                return None
            
            hashed_password = generate_password_hash(password)
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for user creation")
                return None
            
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password) 
                VALUES (%s, %s, %s)
                RETURNING id
            ''', (username, email, hashed_password))
            
            result = cursor.fetchone()
            if result is None:
                logging.error("No result returned after user insertion")
                conn.rollback()
                return None
            
            user_id = result['id']
            conn.commit()
            logging.info(f"User created successfully with ID: {user_id}")
            return user_id
            
        except psycopg2.IntegrityError as e:
            logging.warning(f"User already exists (IntegrityError): {e}")
            if conn:
                conn.rollback()
            return None
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error creating user: {e}")
            if conn:
                conn.rollback()
            return None
        except Exception as e:
            logging.error(f"Unexpected error creating user: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def get_user_by_email(email):
        """Get user by email"""
        conn = None
        try:
            if not email:
                logging.warning("Email not provided for user lookup")
                return None
            
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for user lookup")
                return None
            
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password 
                FROM users 
                WHERE LOWER(email) = LOWER(%s)
            ''', (email,))
            
            user = cursor.fetchone()
            
            if user:
                logging.info(f"User found for email: {email}")
                # Convert to tuple for compatibility with existing code
                return (user['id'], user['username'], user['email'], user['password'])
            else:
                logging.info(f"No user found with email: {email}")
                return None
            
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error getting user by email: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting user by email: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def update_last_login(user_id):
        """Update user's last login timestamp"""
        conn = None
        try:
            if not user_id:
                logging.warning("User ID not provided for last login update")
                return False
            
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for last login update")
                return False
            
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET last_login = %s 
                WHERE id = %s
            ''', (datetime.now(), user_id))
            
            conn.commit()
            logging.info(f"Last login updated for user: {user_id}")
            return True
            
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error updating last login: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logging.error(f"Unexpected error updating last login: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def save_sensor_reading(soil_ph, soil_moisture, soil_temperature, 
                           nitrogen, phosphorus, potassium):
        """Save sensor reading to database"""
        conn = None
        try:
            # Validate inputs
            if soil_ph is None or soil_moisture is None or soil_temperature is None:
                logging.warning("Missing required sensor values")
                return False
            
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for sensor reading")
                return False
            
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sensor_readings 
                (soil_ph, soil_moisture, soil_temperature, nitrogen, phosphorus, potassium)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (soil_ph, soil_moisture, soil_temperature, 
                  nitrogen, phosphorus, potassium))
            
            conn.commit()
            logging.debug("Sensor reading saved successfully")
            return True
            
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error saving sensor reading: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logging.error(f"Unexpected error saving sensor reading: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def get_recent_sensor_readings(limit=10):
        """Get recent sensor readings"""
        conn = None
        try:
            if limit <= 0:
                limit = 10
            
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for sensor readings")
                return []
            
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM sensor_readings 
                ORDER BY timestamp DESC 
                LIMIT %s
            ''', (limit,))
            
            readings = cursor.fetchall()
            logging.info(f"Retrieved {len(readings)} sensor readings")
            return readings if readings else []
            
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error getting sensor readings: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error getting sensor readings: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def save_diagnosis(user_id, plant_type, disease, confidence, image_path):
        """Save disease diagnosis to history"""
        conn = None
        try:
            if not user_id or not plant_type or not disease:
                logging.warning("Missing required fields for diagnosis save")
                return False
            
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for diagnosis save")
                return False
            
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO diagnosis_history 
                (user_id, plant_type, disease, confidence, image_path)
                VALUES (%s, %s, %s, %s, %s)
            ''', (user_id, plant_type, disease, confidence, image_path))
            
            conn.commit()
            logging.info(f"Diagnosis saved for user {user_id}: {disease}")
            return True
            
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error saving diagnosis: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logging.error(f"Unexpected error saving diagnosis: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def get_user_diagnosis_history(user_id, limit=20):
        """Get user's diagnosis history"""
        conn = None
        try:
            if not user_id:
                logging.warning("User ID not provided for diagnosis history")
                return []
            
            if limit <= 0:
                limit = 20
            
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for diagnosis history")
                return []
            
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM diagnosis_history 
                WHERE user_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
            ''', (user_id, limit))
            
            history = cursor.fetchall()
            logging.info(f"Retrieved {len(history)} diagnosis records for user {user_id}")
            return history if history else []
            
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error getting diagnosis history: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error getting diagnosis history: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def save_chat_message(user_id, message, response, language):
        """Save chat interaction"""
        conn = None
        try:
            if not user_id or not message or not response:
                logging.warning("Missing required fields for chat save")
                return False
            
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for chat save")
                return False
            
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history 
                (user_id, message, response, language)
                VALUES (%s, %s, %s, %s)
            ''', (user_id, message, response, language))
            
            conn.commit()
            logging.debug(f"Chat message saved for user {user_id}")
            return True
            
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error saving chat message: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logging.error(f"Unexpected error saving chat message: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def create_test_user():
        """Create test user for development"""
        conn = None
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                logging.error("Failed to get database connection for test user creation")
                return False
            
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM users WHERE email = %s', ('test@test.com',))
            existing_user = cursor.fetchone()
            
            if not existing_user:
                hashed_password = generate_password_hash('test123')
                cursor.execute('''
                    INSERT INTO users (username, email, password) VALUES (%s, %s, %s)
                ''', ('testuser', 'test@test.com', hashed_password))
                conn.commit()
                logging.info("Test user created: test@test.com / test123")
            else:
                logging.info("Test user already exists")
            
            return True
            
        except psycopg2.IntegrityError:
            logging.info("Test user already exists (IntegrityError)")
            if conn:
                conn.rollback()
            return True
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error creating test user: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logging.error(f"Unexpected error creating test user: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()


def initialize_database():
    """Initialize database with tables and test data"""
    try:
        logging.info("Initializing PostgreSQL database...")
        
        if not DATABASE_URL:
            logging.error("DATABASE_URL environment variable not set!")
            return False
        
        logging.info(f"Database URL format: {DATABASE_URL[:20]}...")
        
        if DatabaseManager.create_tables():
            logging.info("Database tables created successfully")
            DatabaseManager.create_test_user()
            logging.info("Database initialization completed successfully")
            return True
        else:
            logging.error("Failed to create database tables")
            return False
            
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        return False
