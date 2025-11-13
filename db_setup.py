
import sqlite3
import os
import psycopg2 # Uncomment if using PostgreSQL and ensure it's installed

DB_FILE = 'app_data.db'

def get_db_connection():
    db_url = os.getenv('DATABASE_URL')

    if db_url:
        # Assuming PostgreSQL for external database
        # import psycopg2 # Make sure psycopg2 is imported if this path is taken
        try:
            conn = psycopg2.connect(db_url)
            return conn
        except Exception as e:
            print(f"Error connecting to external database: {e}")
            return None
    else:
        # Default to SQLite
        try:
            conn = sqlite3.connect(DB_FILE)
            return conn
        except sqlite3.Error as e:
            print(f"Error connecting to local SQLite database: {e}")
            return None

def setup_database():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("Failed to establish database connection. Exiting setup.")
            return

        cursor = conn.cursor()

        # Create feedback_logs table
        cursor.execute("""
            
CREATE TABLE IF NOT EXISTS feedback_logs (
id INTEGER PRIMARY KEY AUTOINCREMENT,
timestamp TEXT NOT NULL,
feedback_text TEXT
)

        """)

        # Create usage_logs table
        cursor.execute("""
            
CREATE TABLE IF NOT EXISTS usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL
)

        """)

        conn.commit()
        print(f"Database '{DB_FILE}' and tables 'feedback_logs', 'usage_logs' initialized successfully.")

    except sqlite3.Error as e:
        print(f"Error setting up database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during database setup: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    setup_database()
