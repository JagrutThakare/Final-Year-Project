import io
import sqlite3
import streamlit as st

@st.cache_resource
def init_db():

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            address TEXT,
            phone TEXT,
            email TEXT UNIQUE,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()


    conn = sqlite3.connect('detection_results.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS number_plate_detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            confidence REAL,
            image BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS helmet_detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_class TEXT,
            confidence REAL,
            bounding_box TEXT,
            image BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS triple_ride_detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_class TEXT,
            confidence REAL,
            bounding_box TEXT,
            image BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def check_user_exists(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    return user is not None

def add_user(name, address, phone, email, username, password):
    if check_user_exists(username):
        return False  
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO users (name, address, phone, email, username, password) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, address, phone, email, username, password))
    conn.commit()
    conn.close()
    return True

def authenticate(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None

def image_to_binary(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")  
    return img_byte_arr.getvalue()

def save_number_plate_data(plate_number, confidence, image):
    conn = sqlite3.connect("detection_results.db")
    c = conn.cursor()
    c.execute("INSERT INTO number_plate_detection (plate_number, confidence, image) VALUES (?, ?, ?)", 
              (plate_number, confidence, image_to_binary(image)))
    conn.commit()
    conn.close()
    
def save_helmet_data(detection_class, confidence, bounding_box, image):
    conn = sqlite3.connect("detection_results.db")
    c = conn.cursor()
    c.execute("INSERT INTO helmet_detection (detection_class, confidence, bounding_box, image) VALUES (?, ?, ?, ?)", 
              (detection_class, confidence, bounding_box, image_to_binary(image)))
    conn.commit()
    conn.close()
    
def save_triple_ride_data(detection_class, confidence, bounding_box, image):
    conn = sqlite3.connect("detection_results.db")
    c = conn.cursor()
    c.execute("INSERT INTO triple_ride_detection (detection_class, confidence, bounding_box, image) VALUES (?, ?, ?, ?)", 
              (detection_class, confidence, bounding_box, image_to_binary(image)))
    conn.commit()
    conn.close()   
