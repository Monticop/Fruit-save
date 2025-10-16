# app.py
import streamlit as st
import os
import zipfile
import tempfile
import time
import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
import base64

# Configure page
st.set_page_config(
    page_title="FruitSave - AI Agriculture",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍎"
)


# --------------------------
# Modern CSS Styling with Enhanced Colors & Animations
# --------------------------
def apply_enhanced_styles():
    st.markdown("""
    <style>
    /* Modern Variables */
    :root {
        --primary-green: #10b981;
        --dark-green: #059669;
        --light-green: #34d399;
        --gold: #f59e0b;
        --orange: #f97316;
        --background-green: #ecfdf5;
        --gradient-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --text-dark: #1f2937;
        --text-light: #6b7280;
        --white: #ffffff;
    }

    /* Main Container */
    .main .block-container {
        padding-top: 1rem;
        background: var(--white);
    }

    /* Sidebar Styling - IMPROVED VISIBILITY */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, var(--dark-green) 0%, var(--primary-green) 100%) !important;
    }

    .sidebar .sidebar-content {
        background: transparent !important;
    }

    .sidebar-title {
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }

    /* Sidebar Navigation Items - IMPROVED VISIBILITY */
    .stButton button {
        width: 100%;
        background: rgba(255,255,255,0.9) !important;
        color: var(--dark-green) !important;
        border: 2px solid rgba(255,255,255,0.8) !important;
        border-radius: 15px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        font-size: 0.95rem !important;
    }

    .stButton button:hover {
        background: rgba(255,255,255,1) !important;
        border-color: white !important;
        transform: translateX(5px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }

    /* User Info in Sidebar */
    .user-info {
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }

    /* Hero Section with Animation */
    .hero-section {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--dark-green) 100%);
        color: white;
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 20px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" opacity="0.1"><circle cx="20" cy="20" r="8" fill="white"/><circle cx="80" cy="40" r="12" fill="white"/><circle cx="40" cy="80" r="10" fill="white"/></svg>');
        animation: float 20s infinite linear;
    }

    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        100% { transform: translateY(-100px) rotate(360deg); }
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, white, #f0fdf4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
    }

    /* Animated Elements */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }

    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Enhanced Cards */
    .feature-card {
        background: var(--white);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
        height: 100%;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-green), var(--gold));
    }

    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.2);
        border-color: var(--primary-green);
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, var(--white), var(--background-green));
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 2px solid var(--light-green);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(16, 185, 129, 0.2);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary-green);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: var(--text-light);
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, var(--white), #f8fafc);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border-left: 5px solid;
        transition: all 0.3s ease;
    }

    .result-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .fresh { border-left-color: #10b981; }
    .medium { border-left-color: #f59e0b; }
    .rotten { border-left-color: #ef4444; }

    /* Upload Area Styling */
    .upload-area {
        border: 3px dashed var(--primary-green) !important;
        border-radius: 20px !important;
        background: var(--background-green) !important;
        padding: 3rem !important;
        text-align: center !important;
    }

    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-green), var(--gold));
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: var(--background-green);
        border-radius: 10px 10px 0px 0px;
        gap: 1rem;
        padding: 1rem 2rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-green) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --------------------------
# Animated Image Display Function with FALLBACK
# --------------------------
def display_animated_image(url, caption="", width=None):
    """Display image with floating animation and fallback"""
    try:
        st.markdown(f"""
        <div class="floating" style="text-align: center;">
            <img src="{url}" style="border-radius: 15px; max-width: {width or '100%'}; height: auto;" alt="{caption}" onerror="this.style.display='none'">
            {f'<p style="margin-top: 0.5rem; color: var(--text-light); font-weight: 600;">{caption}</p>' if caption else ''}
        </div>
        """, unsafe_allow_html=True)
    except:
        # Fallback if image fails to load
        st.markdown(f"""
        <div class="floating" style="text-align: center; padding: 2rem; background: var(--background-green); border-radius: 15px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌱</div>
            <p style="color: var(--text-light); font-weight: 600;">{caption or 'AI Agriculture'}</p>
        </div>
        """, unsafe_allow_html=True)


# --------------------------
# Sidebar Navigation with BETTER VISIBILITY
# --------------------------
def create_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🍎 FruitSave AI</div>', unsafe_allow_html=True)

        if st.session_state.logged_in:
            # User info at top with better visibility
            st.markdown(f'<div class="user-info">Welcome, {st.session_state.email.split("@")[0].title()}!</div>',
                        unsafe_allow_html=True)
            st.markdown("---")

            # Navigation buttons with clear labels
            if st.button("🏠 Overview Dashboard", use_container_width=True, help="View your main dashboard"):
                st.session_state.page = "overview"
                st.rerun()

            if st.button("📤 Upload & Analyze Files", use_container_width=True,
                         help="Upload images, videos, or zip files"):
                st.session_state.page = "upload"
                st.rerun()

            if st.button("🤖 View Predictions", use_container_width=True, help="See all analysis results"):
                st.session_state.page = "predictions"
                st.rerun()

            if st.button("📊 Analytics Dashboard", use_container_width=True, help="Detailed analytics and insights"):
                st.session_state.page = "dashboard"
                st.rerun()

            if st.button("👤 My Account Settings", use_container_width=True, help="Manage your account and profile"):
                st.session_state.page = "account"
                st.rerun()

            st.markdown("---")

            if st.button("🚪 Logout from Account", use_container_width=True, help="Sign out of your account"):
                st.session_state.logged_in = False
                st.session_state.page = "home"
                st.rerun()

        else:
            # Guest navigation
            if st.button("🏠 Home Page", use_container_width=True, help="Go to homepage"):
                st.session_state.page = "home"
                st.rerun()

            if st.button("🔐 Login to Account", use_container_width=True, help="Sign in to your account"):
                st.session_state.page = "login"
                st.rerun()

            if st.button("👤 Create New Account", use_container_width=True, help="Sign up for a new account"):
                st.session_state.page = "signup"
                st.rerun()


# --------------------------
# Enhanced Home Page with BETTER BUTTONS and FALLBACK IMAGES
# --------------------------
def show_home():
    apply_enhanced_styles()

    # Hero Section with Animation
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">AI-Powered Agriculture Revolution</h1>
            <p style="font-size: 1.3rem; opacity: 0.9; margin-bottom: 2rem;">
                Transforming food waste reduction with cutting-edge artificial intelligence. 
                Join the movement to modernize agriculture and save our planet.
            </p>
            <div class="pulse">
                <button style="background: rgba(255,255,255,0.9); border: 2px solid white; color: var(--dark-green); 
                              padding: 1rem 2rem; border-radius: 25px; font-size: 1.1rem; font-weight: 700;
                              cursor: pointer; transition: all 0.3s ease;">
                    🚀 Start Your AI Journey Now
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Animated agriculture image with FALLBACK
        display_animated_image(
            "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400&fit=crop",
            "Modern AI Agriculture",
            "100%"
        )

    # Impact Statistics with Clear Labels
    st.markdown("## 🌍 The Global Impact We're Addressing")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem;">🍎</div>
            <div class="metric-value">1.3B</div>
            <div class="metric-label">Tons of Food Wasted Yearly</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem;">💸</div>
            <div class="metric-value">$1T</div>
            <div class="metric-label">Economic Loss Annually</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem;">🌱</div>
            <div class="metric-value">40%</div>
            <div class="metric-label">Fruits & Vegetables Wasted</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem;">🔥</div>
            <div class="metric-value">8%</div>
            <div class="metric-label">Global Greenhouse Emissions</div>
        </div>
        """, unsafe_allow_html=True)

    # Features Grid with CLEAR CALL-TO-ACTION BUTTONS
    st.markdown("## 🚀 How We're Modernizing Agriculture")

    features = [
        {
            "icon": "🤖",
            "title": "AI Vision Technology",
            "desc": "Advanced computer vision algorithms that accurately detect freshness with 96% precision",
            "image": "https://images.unsplash.com/photo-1555255707-c07966088b7b?w=300&fit=crop",
            "button": "Try AI Analysis"
        },
        {
            "icon": "📱",
            "title": "Smart Mobile Integration",
            "desc": "Real-time analysis on any device, bringing AI to farmers' fingertips",
            "image": "https://images.unsplash.com/photo-1512941937669-90a1b58e7e9c?w=300&fit=crop",
            "button": "Use Mobile App"
        },
        {
            "icon": "🌐",
            "title": "Cloud-Powered Analytics",
            "desc": "Scalable infrastructure that learns and improves with every analysis",
            "image": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=300&fit=crop",
            "button": "View Analytics"
        }
    ]

    cols = st.columns(3)
    for idx, feature in enumerate(features):
        with cols[idx]:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{feature['icon']}</div>
                <h3 style="color: var(--text-dark); margin-bottom: 1rem; font-size: 1.3rem;">{feature['title']}</h3>
                <p style="color: var(--text-light); line-height: 1.6; margin-bottom: 1.5rem;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            display_animated_image(feature['image'], feature['title'], "100%")
            # ADDED CLEAR BUTTON
            if st.button(feature['button'], key=f"feat_btn_{idx}", use_container_width=True):
                st.session_state.page = "signup"
                st.rerun()

    # Second row of features
    features2 = [
        {
            "icon": "📊",
            "title": "Data-Driven Insights",
            "desc": "Comprehensive analytics to optimize supply chains and reduce waste",
            "image": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=300&fit=crop",
            "button": "See Insights"
        },
        {
            "icon": "🔄",
            "title": "Sustainable Impact",
            "desc": "Direct contribution to UN Sustainable Development Goals",
            "image": "https://www.freepik.com/free-ai-image/digital-environment-scene_377117627.htm#fromView=search&page=1&position=30&uuid=eec45aee-b44e-4290-afce-47657faa9a65&query=ai+in+agriculture",
            "button": "Learn More"
        },
        {
            "icon": "⚡",
            "title": "Instant Processing",
            "desc": "Lightning-fast analysis that delivers results in under 3 seconds",
            "image": "https://www.freepik.com/free-photo/medical-team-researcher-working-pharmacology-laboratory-examining-organic-food_16048570.htm#fromView=search&page=1&position=5&uuid=eec45aee-b44e-4290-afce-47657faa9a65&query=ai+in+agriculture",
            "button": "Start Analysis"
        }
    ]

    cols = st.columns(3)
    for idx, feature in enumerate(features2):
        with cols[idx]:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{feature['icon']}</div>
                <h3 style="color: var(--text-dark); margin-bottom: 1rem; font-size: 1.3rem;">{feature['title']}</h3>
                <p style="color: var(--text-light); line-height: 1.6; margin-bottom: 1.5rem;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            display_animated_image(feature['image'], feature['title'], "100%")
            # ADDED CLEAR BUTTON
            if st.button(feature['button'], key=f"feat2_btn_{idx}", use_container_width=True):
                st.session_state.page = "signup"
                st.rerun()

    # CTA Section with Clear Action Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h2 style="color: var(--text-dark); margin-bottom: 1rem; font-size: 2.2rem;">Ready to Join the AI Agriculture Revolution?</h2>
            <p style="color: var(--text-light); margin-bottom: 2rem; font-size: 1.1rem;">Be part of the solution to global food waste. Start making an impact today.</p>
        </div>
        """, unsafe_allow_html=True)

        # Clear call-to-action button
        col4, col5, col6 = st.columns([1, 2, 1])
        with col5:
            if st.button("🚀 Start Your Free AI Journey Today", use_container_width=True, type="primary"):
                st.session_state.page = "signup"
                st.rerun()

            st.markdown("""
            <div style="text-align: center; margin-top: 1rem;">
                <p style="color: var(--text-light); font-size: 0.9rem;">
                    ✅ No credit card required • 🆓 Free forever • ⚡ Instant setup
                </p>
            </div>
            """, unsafe_allow_html=True)


# --------------------------
# Overview Page with Clear Action Buttons
# --------------------------
def show_overview():
    apply_enhanced_styles()
    st.title("🏠 Overview Dashboard")

    # Welcome Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, var(--primary-green), var(--dark-green)); 
                    color: white; padding: 2rem; border-radius: 15px;">
            <h2 style="margin-bottom: 0.5rem;">Welcome back, {st.session_state.email.split('@')[0].title()}! 👋</h2>
            <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">
                Ready to continue your journey in reducing food waste with AI technology.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        display_animated_image(
            "https://www.freepik.com/premium-photo/smart-robotic-farmers-harvest-agriculture-futuristic-robot-automation-work-technology-increase-efficiency_20158569.htm#fromView=search&page=1&position=48&uuid=eec45aee-b44e-4290-afce-47657faa9a65&query=ai+in+agriculture",
            "Your AI Dashboard",
            "100%"
        )

    # Quick Stats with Clear Labels
    st.subheader("📈 Your Quick Stats Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyses", st.session_state.total_processed, "Your contributions to date")

    with col2:
        unique_types = len(st.session_state.counts)
        st.metric("Fruit Types Analyzed", unique_types, "Variety of produce analyzed")

    with col3:
        fresh_count = sum(count for label, count in st.session_state.counts.items() if "Fresh" in label)
        st.metric("Fresh Items Found", fresh_count, "Items in good condition")

    with col4:
        saved_kg = st.session_state.total_processed * 0.15
        st.metric("Waste Prevented", f"{saved_kg:.1f}kg", "Environmental impact made")

    # Recent Activity Section
    st.subheader("📋 Recent Analysis Activity")

    if st.session_state.predictions:
        recent = st.session_state.predictions[-5:]
        for filename, label, confidence in reversed(recent):
            status_class = "fresh" if "Fresh" in label else "rotten" if "Rotten" in label else "medium"
            st.markdown(f"""
            <div class="result-card {status_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; font-size: 1.1rem;">{label}</div>
                        <div style="color: var(--text-light); font-size: 0.9rem;">File: {filename}</div>
                    </div>
                    <div style="font-weight: 700; color: var(--text-dark);">Confidence: {confidence}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📭 No analyses yet. Start by uploading some images to see your activity here!")
        if st.button("📤 Upload Your First Images Now", use_container_width=True, type="primary"):
            st.session_state.page = "upload"
            st.rerun()

    # Quick Actions with Clear Labels
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📸 Analyze New Images", use_container_width=True, type="primary",
                     help="Upload and analyze new fruit images"):
            st.session_state.page = "upload"
            st.rerun()

    with col2:
        if st.button("📊 View Full Analytics", use_container_width=True, help="See detailed analytics and charts"):
            st.session_state.page = "dashboard"
            st.rerun()

    with col3:
        if st.button("🤖 Check All Predictions", use_container_width=True, help="View all analysis results"):
            st.session_state.page = "predictions"
            st.rerun()


# --------------------------
# Unified Upload & Analyze Page
# --------------------------
def show_upload():
    apply_enhanced_styles()
    st.title("📤 Upload & Analyze")

    # Upload Options in Tabs
    tab1, tab2 = st.tabs(["🎯 Upload Files", "📷 Live Camera"])

    with tab1:
        st.subheader("Upload Multiple File Types")

        # Unified file uploader
        uploaded_files = st.file_uploader(
            "Choose files to analyze",
            type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "zip"],
            accept_multiple_files=True,
            help="You can upload images, videos, or zip files all at once!"
        )

        if uploaded_files:
            # Categorize files
            images = [f for f in uploaded_files if f.type.startswith('image')]
            videos = [f for f in uploaded_files if f.type.startswith('video')]
            zips = [f for f in uploaded_files if f.name.endswith('.zip')]

            st.info(f"📁 Found: {len(images)} images, {len(videos)} videos, {len(zips)} zip files")

            if st.button("🚀 Process All Files", use_container_width=True, type="primary"):
                process_all_files(images, videos, zips)

    with tab2:
        st.subheader("Live Camera Capture")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Capture real-time images**")
            camera_img = st.camera_input("Take a picture of fruits", key="main_camera")

            if camera_img:
                st.session_state.camera_bytes = camera_img.getvalue()
                if st.button("🔍 Analyze This Photo", use_container_width=True, type="primary"):
                    with st.spinner("Analyzing with AI..."):
                        process_camera_image()
                        st.success("Analysis complete!")

        with col2:
            if st.session_state.get('camera_bytes'):
                st.write("**Preview**")
                img = Image.open(io.BytesIO(st.session_state.camera_bytes))
                st.image(img, use_container_width=True, caption="Ready for analysis")


# --------------------------
# Processing Functions
# --------------------------
def process_all_files(images, videos, zips):
    total_processed = 0

    # Process images
    if images:
        with st.spinner("🖼️ Processing images..."):
            for file in images:
                success = process_single_file(file)
                if success:
                    total_processed += 1

    # Process videos
    if videos and CV2_AVAILABLE:
        with st.spinner("🎬 Processing videos..."):
            for file in videos:
                processed = process_video_file(file)
                total_processed += processed

    # Process zip files
    if zips:
        with st.spinner("📦 Processing zip files..."):
            for file in zips:
                processed = process_zip_file(file)
                total_processed += processed

    if total_processed > 0:
        st.balloons()
        st.success(f"🎉 Successfully processed {total_processed} items!")

        # Show quick results
        if st.session_state.predictions:
            st.subheader("Quick Results Preview")
            recent = st.session_state.predictions[-3:]
            for filename, label, confidence in recent:
                status_class = "fresh" if "Fresh" in label else "rotten" if "Rotten" in label else "medium"
                st.markdown(f"""
                <div class="result-card {status_class}">
                    <strong>{label}</strong> - {filename}<br>
                    <small>Confidence: {confidence}</small>
                </div>
                """, unsafe_allow_html=True)


def process_single_file(file):
    try:
        # Save temporarily
        temp_path = os.path.join("uploads", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        # Process based on file type
        if file.type.startswith('image'):
            img = Image.open(temp_path).convert("RGB")
            label, conf = predict_image(img)

            # Store results
            st.session_state.predictions.append((file.name, label, f"{conf * 100:.1f}%"))
            st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
            st.session_state.total_processed += 1

            # Show result
            st.write(f"✅ {file.name}: {label} ({conf * 100:.1f}%)")

        # Clean up
        os.remove(temp_path)
        return True

    except Exception as e:
        st.error(f"❌ Error processing {file.name}: {str(e)}")
        return False


def process_video_file(file):
    if not CV2_AVAILABLE:
        st.warning(f"Video processing not available for {file.name}")
        return 0

    try:
        # Save video temporarily
        temp_path = os.path.join("uploads", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        # Process video
        processed = process_video_frames(temp_path)

        # Clean up
        os.remove(temp_path)
        return processed

    except Exception as e:
        st.error(f"❌ Error processing video {file.name}: {str(e)}")
        return 0


def process_zip_file(file):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find and process images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(Path(temp_dir).rglob(ext))

            processed = 0
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    label, conf = predict_image(img)

                    # Store results
                    st.session_state.predictions.append((img_path.name, label, f"{conf * 100:.1f}%"))
                    st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
                    st.session_state.total_processed += 1
                    processed += 1

                except Exception:
                    continue

            st.info(f"📦 Processed {processed} images from {file.name}")
            return processed

    except Exception as e:
        st.error(f"❌ Error processing zip {file.name}: {str(e)}")
        return 0


def process_camera_image():
    if st.session_state.camera_bytes:
        img = Image.open(io.BytesIO(st.session_state.camera_bytes))
        label, conf = predict_image(img)

        # Store results
        st.session_state.predictions.append(("Camera Capture", label, f"{conf * 100:.1f}%"))
        st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
        st.session_state.total_processed += 1

        # Display result
        status_class = "fresh" if "Fresh" in label else "rotten" if "Rotten" in label else "medium"
        st.markdown(f"""
        <div class="result-card {status_class}">
            <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;">
                {label}
            </div>
            <div style="color: var(--text-light);">
                Confidence: {conf * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)


def process_video_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    label, conf = predict_image(pil_img)

                    if conf > 0.6:
                        st.session_state.predictions.append((f"frame_{frame_count}", label, f"{conf * 100:.1f}%"))
                        st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
                        st.session_state.total_processed += 1
                        processed_count += 1

                except Exception:
                    continue

            frame_count += 1

        cap.release()
        return processed_count

    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        return 0


# --------------------------
# Predictions Page
# --------------------------
def show_predictions():
    apply_enhanced_styles()
    st.title("🤖 Analysis Predictions")

    if not st.session_state.predictions:
        st.info("No predictions yet. Start by uploading some files!")
        if st.button("📤 Upload Files", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
        return

    # Summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Predictions", len(st.session_state.predictions))

    with col2:
        fresh_count = sum(1 for _, label, _ in st.session_state.predictions if "Fresh" in label)
        st.metric("Fresh Items", fresh_count)

    with col3:
        rotten_count = sum(1 for _, label, _ in st.session_state.predictions if "Rotten" in label)
        st.metric("Items to Use", rotten_count)

    # Data Table
    st.subheader("📋 All Predictions")

    # Convert to DataFrame
    df = pd.DataFrame(
        st.session_state.predictions,
        columns=["File Name", "Predicted Class", "Confidence"]
    )

    # Add color coding
    def color_class(val):
        if 'Fresh' in val:
            return 'color: #10b981; font-weight: bold;'
        elif 'Rotten' in val:
            return 'color: #ef4444; font-weight: bold;'
        else:
            return 'color: #f59e0b; font-weight: bold;'

    styled_df = df.style.map(color_class, subset=['Predicted Class'])
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV",
            data=csv,
            file_name="fruit_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        if st.button("🔄 Clear Predictions", use_container_width=True, type="secondary"):
            st.session_state.predictions = []
            st.session_state.counts = {}
            st.session_state.total_processed = 0
            st.success("Predictions cleared!")
            st.rerun()


# --------------------------
# Dashboard Page
# --------------------------
def show_dashboard():
    apply_enhanced_styles()
    st.title("📊 Analytics Dashboard")

    if st.session_state.total_processed == 0:
        st.info("Start analyzing files to see your dashboard data!")
        if st.button("📤 Start Analyzing", use_container_width=True, type="primary"):
            st.session_state.page = "upload"
            st.rerun()
        return

    # Impact Summary
    st.subheader("🌱 Your Environmental Impact")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_processed}</div>
            <div class="metric-label">Items Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        saved_kg = st.session_state.total_processed * 0.15
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{saved_kg:.1f}kg</div>
            <div class="metric-label">Waste Prevented</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        co2_saved = saved_kg * 2.5
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{co2_saved:.1f}kg</div>
            <div class="metric-label">CO₂ Saved</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        water_saved = saved_kg * 1000
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{water_saved:.0f}L</div>
            <div class="metric-label">Water Saved</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Freshness Distribution")
        if st.session_state.counts:
            fig, ax = plt.subplots(figsize=(10, 6))
            classes = list(st.session_state.counts.keys())
            counts = list(st.session_state.counts.values())

            colors = ['#ef4444' if 'Rotten' in label else '#10b981' if 'Fresh' in label else '#f59e0b' for label in
                      classes]

            bars = ax.bar(classes, counts, color=colors)
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    with col2:
        st.subheader("Category Overview")
        if st.session_state.counts:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)


# --------------------------
# Account Page
# --------------------------
def show_account():
    apply_enhanced_styles()
    st.title("👤 My Account")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profile Information")
        st.write(f"**Email:** {st.session_state.email}")
        st.write(f"**Member since:** October 2024")
        st.write(f"**Account type:** Free Plan")
        st.write(f"**Analyses performed:** {st.session_state.total_processed}")

        st.subheader("Account Management")
        if st.button("Reset All Data", type="secondary", use_container_width=True):
            st.session_state.predictions = []
            st.session_state.counts = {}
            st.session_state.total_processed = 0
            st.success("All data reset successfully!")

    with col2:
        st.subheader("Your Impact Summary")
        if st.session_state.total_processed > 0:
            saved_kg = st.session_state.total_processed * 0.15
            co2_saved = saved_kg * 2.5
            water_saved = saved_kg * 1000

            st.metric("Food Waste Prevented", f"{saved_kg:.1f} kg")
            st.metric("CO₂ Emissions Saved", f"{co2_saved:.1f} kg")
            st.metric("Water Conservation", f"{water_saved:.0f} liters")
            st.metric("Money Saved", f"${saved_kg * 2:.1f}")
        else:
            st.info("Start analyzing to see your impact!")
            display_animated_image(
                "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=300&fit=crop",
                "Make an Impact"
            )


# --------------------------
# Login & Signup Pages
# --------------------------
def show_login():
    apply_enhanced_styles()
    st.title("🔐 Login to FruitSave")

    with st.form("login_form"):
        email = st.text_input("📧 Email Address")
        password = st.text_input("🔒 Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
            elif email in st.session_state.users and st.session_state.users[email] == password:
                st.session_state.logged_in = True
                st.session_state.email = email
                st.session_state.page = "overview"
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    st.markdown("---")
    st.markdown("Don't have an account?")
    if st.button("Create Account", use_container_width=True):
        st.session_state.page = "signup"
        st.rerun()


def show_signup():
    apply_enhanced_styles()
    st.title("👤 Create Your Account")

    with st.form("signup_form"):
        email = st.text_input("📧 Email Address")
        password = st.text_input("🔒 Password", type="password")
        confirm = st.text_input("✅ Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please fill in all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif email in st.session_state.users:
                st.error("Email already registered.")
            else:
                st.session_state.users[email] = password
                st.success("Account created successfully! Please login.")
                st.session_state.page = "login"
                st.rerun()

    st.markdown("---")
    st.markdown("Already have an account?")
    if st.button("Sign In", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()


# --------------------------
# Initialize Session State & Core Functions
# --------------------------
def initialize_session_state():
    defaults = {
        "users": {"admin@fruitsave.com": "password123", "user@example.com": "password123"},
        "page": "home",
        "logged_in": False,
        "email": "",
        "predictions": [],
        "counts": {},
        "total_processed": 0,
        "camera_bytes": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize OpenCV
if 'opencv_checked' not in st.session_state:
    try:
        import cv2

        st.session_state.cv2 = cv2
        st.session_state.CV2_AVAILABLE = True
        st.session_state.opencv_checked = True
    except ImportError as e:
        st.session_state.cv2 = None
        st.session_state.CV2_AVAILABLE = False
        st.session_state.opencv_checked = True

CV2_AVAILABLE = st.session_state.get('CV2_AVAILABLE', False)
cv2 = st.session_state.get('cv2', None)

# Constants
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MODEL_PATH = "fruit_classifier_finetuned.h5"

INDONESIAN_CLASS_ORDER = [
    "jeruk_busuk", "jeruk_segar", "jeruk_segar_sedang",
    "tomat_busuk", "tomat_segar", "tomat_segar_sedang",
    "wortel_busuk", "wortel_segar", "wortel_segar_sedang"
]

CLASS_NAME_MAPPING = {
    'wortel_busuk': 'Rotten Carrot', 'tomat_segar': 'Fresh Tomato',
    'wortel_segar_sedang': 'Medium Fresh Carrot', 'jeruk_segar_sedang': 'Medium Fresh Orange',
    'jeruk_segar': 'Fresh Orange', 'tomat_busuk': 'Rotten Tomato',
    'wortel_segar': 'Fresh Carrot', 'tomat_segar_sedang': 'Medium Fresh Tomato',
    'jeruk_busuk': 'Rotten Orange'
}


# Model Loading
@st.cache_resource
def load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model, True, ""
    except Exception as e:
        return None, False, str(e)


model, model_loaded, load_error = load_model(MODEL_PATH)


# Core ML Functions
def preprocess_image(pil_image: Image.Image, target_size=(150, 150)):
    img = pil_image.convert("RGB").resize(target_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr


def predict_image(pil_image: Image.Image, debug=False):
    if model is None:
        raise RuntimeError("Model not loaded.")
    arr = preprocess_image(pil_image)
    preds = model.predict(arr, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    ind_label = INDONESIAN_CLASS_ORDER[pred_idx]
    eng_label = CLASS_NAME_MAPPING.get(ind_label, ind_label)
    return eng_label, confidence


def main():
    initialize_session_state()
    create_sidebar()

    # Route pages
    if not st.session_state.logged_in:
        if st.session_state.page == "home":
            show_home()
        elif st.session_state.page == "login":
            show_login()
        elif st.session_state.page == "signup":
            show_signup()
        else:
            show_home()
    else:
        if st.session_state.page == "overview":
            show_overview()
        elif st.session_state.page == "upload":
            show_upload()
        elif st.session_state.page == "predictions":
            show_predictions()
        elif st.session_state.page == "dashboard":
            show_dashboard()
        elif st.session_state.page == "account":
            show_account()
        else:
            show_overview()


if __name__ == "__main__":
    main()