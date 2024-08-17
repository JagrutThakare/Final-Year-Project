import base64
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
from app1 import run_number_plate_detection
from app2 import run_helmet_detection
from app3 import run_triple_ride_detection
import io

# -------------------- Database + Helpers --------------------

def fetch_database_data():
    conn = sqlite3.connect("detection_results.db")
    st.subheader("📋 Detection Records")

    st.markdown("### 🚘 Number Plate Detections")
    df_number_plate = pd.read_sql_query("SELECT * FROM number_plate_detection", conn)
    display_data_with_images(df_number_plate, "image")

    st.markdown("### ⛑️ Helmet Detections")
    df_helmet = pd.read_sql_query("SELECT * FROM helmet_detection", conn)
    display_data_with_images(df_helmet, "image")

    st.markdown("### 🏍️ Triple Ride Detections")
    df_triple_ride = pd.read_sql_query("SELECT * FROM triple_ride_detection", conn)
    display_data_with_images(df_triple_ride, "image")

    conn.close()

def display_data_with_images(df, image_column):
    if not df.empty:
        def convert_image(img_data):
            if isinstance(img_data, bytes):
                try:
                    return Image.open(io.BytesIO(img_data))  
                except:
                    return None
            elif isinstance(img_data, str):
                try:
                    img_array = np.fromstring(img_data, sep=",", dtype=np.uint8)
                    img = img_array.reshape((128, 128, 3))  
                    return Image.fromarray(img)
                except:
                    return None
            return None

        display_df = df.copy()
        display_df[image_column] = display_df[image_column].apply(lambda x: convert_image(x))
        
        def image_formatter(img):
            if img is not None:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f'<img src="data:image/png;base64,{img_str}" width="100">'
            return "No Image"

        html_df = display_df.copy()
        html_df[image_column] = html_df[image_column].apply(image_formatter)

        html_table = html_df.to_html(
            escape=False,
            index=False,
            classes='table table-bordered table-striped',
            justify='center'
        )

        st.markdown("""
            <style>
            .table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .table th, .table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            .table th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            .table-striped tbody tr:nth-of-type(odd) {
                background-color: #f9f9f9;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.write("No data available.")

def fetch_user_data():
    conn = sqlite3.connect("users.db")
    st.subheader("👤 Registered Users")
    df_users = pd.read_sql_query("SELECT id, name, address, phone, email, username FROM users", conn)
    st.dataframe(df_users if not df_users.empty else "No registered users found.")
    conn.close()

# -------------------- Pages --------------------

def admin_dashboard():
    st.title("📊 Admin Dashboard")
    fetch_database_data()
    fetch_user_data()

def user_dashboard():
    st.title("Hybrid Detection App")
    st.markdown("""
        This app allows you to detect **Number Plates**, **Helmets**, and **Triple Rides** in images.
        Select the detection type below.
    """)

    app_selection = st.selectbox(
        "Choose a detection app", 
        ["Number Plate Detection", "Helmet Detection", "Triple Ride Detection"]
    )

    if app_selection == "Number Plate Detection":
        run_number_plate_detection()
    elif app_selection == "Helmet Detection":
        run_helmet_detection()
    elif app_selection == "Triple Ride Detection":
        run_triple_ride_detection()

# -------------------- Main --------------------

if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False

# Top-right button
st.markdown(
    """
    <style>
    .top-right {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    </style>
    """, unsafe_allow_html=True,
)

# Sidebar toggle button
with st.sidebar:
    button_label = "Go to User Dashboard" if st.session_state.admin_mode else "Go to Admin Dashboard"
    if st.button(button_label, key="admin_btn"):
        st.session_state.admin_mode = not st.session_state.admin_mode
        st.rerun()

# Render correct view
if st.session_state.admin_mode:
    admin_dashboard()
else:
    user_dashboard()
