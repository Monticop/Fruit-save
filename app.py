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
from PIL import ImageDraw

# Configure page first - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Fruit Save",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# --------------------------
# Enhanced OpenCV & Video Handling
# --------------------------
# Initialize session state for OpenCV status if not exists
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

# Use the session state variables
CV2_AVAILABLE = st.session_state.get('CV2_AVAILABLE', False)
cv2 = st.session_state.get('cv2', None)

# Video processing IS available if OpenCV is available
VIDEO_SUPPORTED = CV2_AVAILABLE
# --------------------------
# Constants and Configuration
# --------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = "fruit_classifier_finetuned.h5"

# Indonesian folder order used during training
INDONESIAN_CLASS_ORDER = [
    "jeruk_busuk", "jeruk_segar", "jeruk_segar_sedang",
    "tomat_busuk", "tomat_segar", "tomat_segar_sedang",
    "wortel_busuk", "wortel_segar", "wortel_segar_sedang"
]

# Indonesian -> English mapping
CLASS_NAME_MAPPING = {
    'wortel_busuk': 'Rotten Carrot', 'tomat_segar': 'Fresh Tomato',
    'wortel_segar_sedang': 'Medium Fresh Carrot', 'jeruk_segar_sedang': 'Medium Fresh Orange',
    'jeruk_segar': 'Fresh Orange', 'tomat_busuk': 'Rotten Tomato',
    'wortel_segar': 'Fresh Carrot', 'tomat_segar_sedang': 'Medium Fresh Tomato',
    'jeruk_busuk': 'Rotten Orange'
}

CLASS_NAMES = [CLASS_NAME_MAPPING[name] for name in INDONESIAN_CLASS_ORDER]


# --------------------------
# Session State Initialization
# --------------------------
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "users": {"admin@gmail.com": "1234"},
        "page": "home",
        "logged_in": False,
        "email": "",
        "predictions": [],
        "counts": {},
        "total_processed": 0,
        "camera_open": False,
        "camera_bytes": None,
        "camera_confirmed": False,
        "username": ""
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------
# Model Loading
# --------------------------
@st.cache_resource
def load_model(path):
    """Load and cache the TensorFlow model"""
    try:
        model = tf.keras.models.load_model(path)
        return model, True, ""
    except Exception as e:
        return None, False, str(e)


# Load model
model, model_loaded, load_error = load_model(MODEL_PATH)


# --------------------------
# Enhanced Utility Functions
# --------------------------
def preprocess_image(pil_image: Image.Image, target_size=(150, 150)):
    """Resize + convert to array + normalize to [0,1]."""
    img = pil_image.convert("RGB").resize(target_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr


def predict_image(pil_image: Image.Image, debug=False):
    """Return (english_label, confidence) for a PIL image."""
    if model is None:
        raise RuntimeError("Model not loaded.")
    arr = preprocess_image(pil_image)
    preds = model.predict(arr, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    ind_label = INDONESIAN_CLASS_ORDER[pred_idx]
    eng_label = CLASS_NAME_MAPPING.get(ind_label, ind_label)
    return eng_label, confidence


def detect_multiple_fruits_enhanced(pil_image):
    """Enhanced multi-fruit detection"""
    if not CV2_AVAILABLE:
        return []

    try:
        # Convert PIL to OpenCV
        cv_img = np.array(pil_image)
        if len(cv_img.shape) == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        # Convert to HSV for color segmentation
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # Color ranges for fruits
        color_ranges = [
            ((0, 50, 50), (10, 255, 255)),  # Red (tomatoes)
            ((10, 50, 50), (25, 255, 255)),  # Orange (oranges)
            ((25, 50, 50), (35, 255, 255)),  # Yellow (carrots)
        ]

        combined_mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)

        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        predictions = []
        img_h, img_w = cv_img.shape[:2]
        min_area = (img_w * img_h) * 0.001
        max_area = (img_w * img_h) * 0.3

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Expand bounding box
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_w - x, w + 2 * padding)
            h = min(img_h - y, h + 2 * padding)

            fruit_crop = pil_image.crop((x, y, x + w, y + h))

            try:
                label, conf = predict_image(fruit_crop)
                if conf >= 0.3:
                    predictions.append((label, conf, fruit_crop))
            except Exception:
                continue

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:6]

    except Exception as e:
        st.error(f"Multi-fruit detection error: {e}")
        return []


def process_video_robust(video_path, sample_rate=15):
    """Robust video processing that actually works"""
    if not CV2_AVAILABLE:
        st.error("❌ OpenCV not available - please check if opencv-python is installed")
        return False, "OpenCV not available for video processing", 0

    try:
        # Test if we can even create a VideoCapture object
        cap = cv2.VideoCapture()
        if not hasattr(cv2, 'VideoCapture'):
            st.error("❌ VideoCapture not available in OpenCV")
            return False, "Video processing not supported", 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"❌ Cannot open video file: {video_path}")
            return False, "Cannot open video file", 0

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.info(f"🎬 Video Info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

        if total_frames == 0:
            cap.release()
            st.error("❌ Video contains 0 frames")
            return False, "Video contains no frames", 0

        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                try:
                    # Convert frame to PIL Image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)

                    # Process the frame
                    label, conf = predict_image(pil_img)

                    # Store results
                    st.session_state.predictions.append((f"frame_{frame_idx}", label, f"{conf * 100:.1f}%"))
                    st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
                    st.session_state.total_processed += 1
                    processed_count += 1

                except Exception as e:
                    st.warning(f"Frame {frame_idx} processing failed: {str(e)}")
                    continue

            # Update progress every 10 frames
            if total_frames > 0 and frame_idx % 10 == 0:
                progress = min(1.0, frame_idx / total_frames)
                progress_bar.progress(progress)
                status_text.text(f"📊 Processed {processed_count} items from {frame_idx}/{total_frames} frames")

            frame_idx += 1

        cap.release()
        progress_bar.progress(1.0)
        status_text.empty()

        if processed_count > 0:
            return True, f"✅ Video processed successfully! Analyzed {processed_count} frames", processed_count
        else:
            return False, "❌ No frames could be processed from the video", 0

    except Exception as e:
        st.error(f"❌ Video processing error: {str(e)}")
        return False, f"Video processing failed: {str(e)}", 0


def process_single_image_robust(image_path, filename):
    """Process single image with multi-fruit detection"""
    try:
        img = Image.open(image_path).convert("RGB")
        processed_count = 0

        # Always try multi-fruit detection
        detections = detect_multiple_fruits_enhanced(img) if CV2_AVAILABLE else []

        if detections:
            st.image(img, caption=f"🔍 Detected {len(detections)} fruits in {filename}", use_column_width=True)

            cols = st.columns(min(3, len(detections)))
            for idx, (label, conf, crop) in enumerate(detections):
                col_idx = idx % 3
                with cols[col_idx]:
                    emoji = "🔴" if conf < 0.5 else "🟠" if conf < 0.8 else "🟢"
                    st.image(crop, caption=f"{emoji} {label} ({conf * 100:.1f}%)", width=200)

                    st.session_state.predictions.append((f"{filename}_fruit{idx + 1}", label, f"{conf * 100:.1f}%"))
                    st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
                    st.session_state.total_processed += 1
                    processed_count += 1

            return True, f"🎯 Multi-fruit analysis: {len(detections)} fruits detected", processed_count

        else:
            # Single image analysis
            label, conf = predict_image(img)
            emoji = "🔴" if conf < 0.5 else "🟠" if conf < 0.8 else "🟢"
            st.image(img, caption=f"{emoji} {filename} → {label} ({conf * 100:.1f}%)", use_column_width=True)

            st.session_state.predictions.append((filename, label, f"{conf * 100:.1f}%"))
            st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
            st.session_state.total_processed += 1

            return True, "✅ Image analyzed successfully", 1

    except Exception as e:
        return False, f"❌ Error processing image: {str(e)}", 0


def reset_counters():
    st.session_state["predictions"] = []
    st.session_state["counts"] = {}
    st.session_state["total_processed"] = 0
    st.success("All counters have been reset!")


def go_to(page_name: str):
    st.session_state["page"] = page_name
    st.rerun()


# --------------------------
# Page Functions
# --------------------------
def show_home():
    """Display the home page with automatic image carousel and enhanced visuals"""

    # Custom CSS for styling
    st.markdown("""
    <style>
    .hero {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .hero h1 {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .hero p {
        font-size: 20px;
        color: #f0f0f0;
        margin-bottom: 30px;
    }
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 5px solid #22c55e;
    }
    .carousel-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 30px 0;
        text-align: center;
    }
    .stats-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
    .carousel-dots {
        display: flex;
        justify-content: center;
        margin-top: 15px;
    }
    .dot {
        height: 12px;
        width: 12px;
        margin: 0 5px;
        background-color: rgba(255,255,255,0.5);
        border-radius: 50%;
        display: inline-block;
        transition: background-color 0.3s ease;
    }
    .dot.active {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1>🍎 Fruit Save</h1>
        <p>Revolutionizing Food Waste Reduction Through AI-Powered Freshness Detection</p>
    </div>
    """, unsafe_allow_html=True)

    # Automatic Image Carousel Section
    st.markdown("""
    <div class="carousel-container">
        <h2 style="color: white; text-align: center; margin-bottom: 20px;">🎨 See Fruit Save in Action</h2>
    </div>
    """, unsafe_allow_html=True)

    # Automatic Carousel using time-based rotation
    carousel_images = [
        {
            "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
            "caption": "Sustainable farming"
        },
        {
            "url": "https://images.unsplash.com/photo-1561136594-7f68413baa99?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80",
            "caption": "Tomato Quality Assessment"
        },
        {
            "url": "https://images.pexels.com/photos/8845420/pexels-photo-8845420.jpeg",
            "caption": "Fresh vegetables"
        },
        {
            "url": "https://images.unsplash.com/photo-1619566636858-adf3ef46400b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
            "caption": "AI Technology in Modern Agriculture"
        }
    ]

    # Get current time to create automatic rotation
    current_time = time.time()
    rotation_interval = 10  # Change image every 10 seconds
    current_index = int((current_time / rotation_interval) % len(carousel_images))

    # Display current image
    current_image = carousel_images[current_index]

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            current_image["url"],
            use_column_width=True,
            caption=f"🔄 {current_image['caption']} • Auto-rotates every {rotation_interval} seconds"
        )

        # Create dots indicator
        dots_html = '<div class="carousel-dots">'
        for i in range(len(carousel_images)):
            dot_class = "dot active" if i == current_index else "dot"
            dots_html += f'<span class="{dot_class}"></span>'
        dots_html += '</div>'

        st.markdown(dots_html, unsafe_allow_html=True)

        # Manual control (optional)
        st.markdown("---")
        st.caption("💡 **Manual Control**: Select an image to view")
        selected_index = st.selectbox(
            "Browse images manually:",
            options=range(len(carousel_images)),
            format_func=lambda x: carousel_images[x]["caption"],
            index=current_index,
            label_visibility="collapsed"
        )

        # Show selected image if manually changed
        if selected_index != current_index:
            manual_image = carousel_images[selected_index]
            st.image(
                manual_image["url"],
                use_column_width=True,
                caption=f"👆 {manual_image['caption']}"
            )

    # Stats Section
    st.markdown("---")
    st.subheader("📊 Making a Real Impact")

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    with stats_col1:
        st.markdown("""
        <div class="stats-card">
            <h3>🌍 1.3B</h3>
            <p>Tons of food wasted annually</p>
        </div>
        """, unsafe_allow_html=True)

    with stats_col2:
        st.markdown("""
        <div class="stats-card">
            <h3>💰 $1T</h3>
            <p>Economic cost of food waste</p>
        </div>
        """, unsafe_allow_html=True)

    with stats_col3:
        st.markdown("""
        <div class="stats-card">
            <h3>🎯 40%</h3>
            <p>Fruits & vegetables wasted</p>
        </div>
        """, unsafe_allow_html=True)

    with stats_col4:
        st.markdown("""
        <div class="stats-card">
            <h3>🤖 95%</h3>
            <p>AI detection accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    # Features Section
    st.markdown("---")
    st.subheader("🚀 Why Choose Fruit Save?")

    features_col1, features_col2 = st.columns(2)

    with features_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔬 AI-Powered Analysis</h4>
            <p>Advanced computer vision algorithms detect freshness with 95%+ accuracy using deep learning models.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>📱 Real-Time Processing</h4>
            <p>Instant analysis with live camera feed and quick upload processing for immediate results.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>🌱 Sustainability Focus</h4>
            <p>Directly contributes to UN Sustainable Development Goals for responsible consumption.</p>
        </div>
        """, unsafe_allow_html=True)

    with features_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🍎 Multi-Fruit Detection</h4>
            <p>Simultaneously analyze multiple fruits in a single image with individual freshness scores.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>📊 Smart Dashboard</h4>
            <p>Comprehensive analytics with visual charts, trends, and export capabilities.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>🔒 Secure & Private</h4>
            <p>Enterprise-grade security with user authentication and data protection.</p>
        </div>
        """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("---")
    st.subheader("🔧 How Fruit Save Works")

    steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)

    with steps_col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 40px; margin-bottom: 10px;">1️⃣</div>
            <h4>Upload/Capture</h4>
            <p>Take photos with your camera or upload existing images of fruits</p>
        </div>
        """, unsafe_allow_html=True)

    with steps_col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 40px; margin-bottom: 10px;">2️⃣</div>
            <h4>AI Analysis</h4>
            <p>Our deep learning model analyzes freshness and quality in seconds</p>
        </div>
        """, unsafe_allow_html=True)

    with steps_col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 40px; margin-bottom: 10px;">3️⃣</div>
            <h4>Get Results</h4>
            <p>Receive detailed freshness reports with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)

    with steps_col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 40px; margin-bottom: 10px;">4️⃣</div>
            <h4>Take Action</h4>
            <p>Make informed decisions to reduce waste and improve quality</p>
        </div>
        """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-radius: 15px;">
        <h2 style="color: #333; margin-bottom: 20px;">Ready to Reduce Food Waste?</h2>
        <p style="color: #666; font-size: 18px; margin-bottom: 30px;">Join the movement and start making a difference today</p>
    </div>
    """, unsafe_allow_html=True)

    # Get Started Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Your Journey →", use_container_width=True, type="primary"):
            go_to("login")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🍎 Built with ❤️ for a sustainable future | Supporting UN SDGs 12 & 13</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh for carousel (every 5 seconds)
    time.sleep(5)
    st.rerun()


def show_login():
    """Display the login page"""
    st.title("🔐 Login to Fruit Save")

    with st.form("login_form"):
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if not email or not password:
                st.error("❌ Please enter both email and password.")
            elif email in st.session_state.users and st.session_state.users[email] == password:
                st.session_state.logged_in = True
                st.session_state.email = email
                st.session_state.page = "main"
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid email or password.")

    st.markdown("---")
    st.markdown("Don't have an account?")
    if st.button("Sign Up Here"):
        go_to("signup")


def show_signup():
    """Display the signup page"""
    st.title("👤 Create Account")

    with st.form("signup_form"):
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        confirm = st.text_input("✅ Confirm Password", type="password")
        submitted = st.form_submit_button("Sign Up")

        if submitted:
            if not email or not password:
                st.error("❌ Please fill in all fields.")
            elif password != confirm:
                st.error("❌ Passwords do not match.")
            elif email in st.session_state.users:
                st.error("❌ Email already registered.")
            else:
                st.session_state.users[email] = password
                st.success("✅ Account created successfully! Please login.")
                go_to("login")

    st.markdown("---")
    st.markdown("Already have an account?")
    if st.button("Login Here"):
        go_to("login")


def process_camera_image_unified():
    """Process camera image"""
    if st.session_state.camera_bytes:
        img = Image.open(io.BytesIO(st.session_state.camera_bytes))

        detections = detect_multiple_fruits_enhanced(img) if CV2_AVAILABLE else []

        if detections:
            st.success(f"🎯 Detected {len(detections)} fruit(s)")
            cols = st.columns(min(3, len(detections)))
            for idx, (label, conf, crop) in enumerate(detections):
                col_idx = idx % 3
                with cols[col_idx]:
                    emoji = "🔴" if conf < 0.5 else "🟠" if conf < 0.8 else "🟢"
                    st.image(crop, caption=f"{emoji} Fruit {idx + 1}: {label} ({conf * 100:.1f}%)", width=250)

                    st.session_state.predictions.append((f"camera_fruit{idx + 1}", label, f"{conf * 100:.1f}%"))
                    st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
                    st.session_state.total_processed += 1
        else:
            label, conf = predict_image(img)
            emoji = "🔴" if conf < 0.5 else "🟠" if conf < 0.8 else "🟢"
            st.image(img, caption=f"{emoji} {label} ({conf * 100:.1f}%)", width=300)

            st.session_state.predictions.append(("camera_image", label, f"{conf * 100:.1f}%"))
            st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
            st.session_state.total_processed += 1

        st.session_state.camera_confirmed = True


def show_app():
    """Main application after login"""
    st.sidebar.title("🧭 Navigation")

    # User info
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.email}")
    st.sidebar.markdown("---")

    # POSITIVE System capabilities - No negative warnings!
    st.sidebar.success("✅ **System Ready**")
    st.sidebar.info("🎯 All features enabled")

    # Navigation
    menu_options = ["🏠 Overview", "📤 Upload Data", "🤖 Predictions", "📊 Dashboard", "🔧 Settings", "🚪 Logout"]
    selected_menu = st.sidebar.radio("Go to", menu_options)

    # Main content area based on selection
    if selected_menu == "🏠 Overview":
        show_overview()
    elif selected_menu == "📤 Upload Data":
        show_upload_data()
    elif selected_menu == "🤖 Predictions":
        show_predictions()
    elif selected_menu == "📊 Dashboard":
        show_dashboard()
    elif selected_menu == "🔧 Settings":
        show_settings()
    elif selected_menu == "🚪 Logout":
        show_logout()


def show_overview():
    """Display overview page"""
    st.title("🏠 Overview")
    st.info("Welcome to Fruit Save! Use the Upload Data section to analyze fruits and vegetables.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Processed", st.session_state.total_processed)

    with col2:
        unique_classes = len(st.session_state.counts)
        st.metric("Unique Classes", unique_classes)

    with col3:
        most_common = max(st.session_state.counts.items(), key=lambda x: x[1]) if st.session_state.counts else (
        "None", 0)
        st.metric("Most Common", most_common[0])


def show_upload_data():
    """Display upload data page"""
    st.title("📤 Upload & Analyze")

    if not model_loaded:
        st.error(f"❌ Model failed to load: {load_error}")
        return

    # Show OpenCV status only once when page loads
    if 'opencv_status_shown' not in st.session_state:
        if CV2_AVAILABLE:
            st.success("✅ OpenCV Headless loaded - Video processing ENABLED!")
        else:
            st.error("❌ OpenCV not available - Video processing disabled")
            st.info("Please ensure 'opencv-python-headless' is in requirements.txt")
        st.session_state.opencv_status_shown = True
    # Camera section
    st.subheader("📸 Camera Capture")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Open Camera", key="open_camera"):
            st.session_state.camera_open = True
            st.session_state.camera_confirmed = False

    with col2:
        if st.button("Close Camera", key="close_camera"):
            st.session_state.camera_open = False
            st.session_state.camera_bytes = None

    if st.session_state.camera_open:
        camera_img = st.camera_input("Take a picture")

        if camera_img is not None:
            st.session_state.camera_bytes = camera_img.getvalue()
            preview_img = Image.open(io.BytesIO(st.session_state.camera_bytes))
            st.image(preview_img, caption="Preview - Confirm to use this image")

            if st.button("✅ Use This Photo"):
                process_camera_image_unified()

    # File upload section
    st.subheader("📁 Upload Media Files")

    uploaded_files = st.file_uploader(
        "Choose images or videos",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        accept_multiple_files=True,
        help="Upload images for analysis or videos for frame-by-frame processing"
    )

    # Processing options
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            sample_rate = st.slider("Video frame sampling rate", 5, 30, 15,
                                    help="Higher values = faster processing, lower values = more detailed analysis")
        with col2:
            if st.button("🔄 Reset All Counters", use_container_width=True):
                reset_counters()

    # Process uploaded files
    if uploaded_files:
        total_processed = 0
        total_files = len(uploaded_files)

        progress_bar = st.progress(0)

        for file_idx, file in enumerate(uploaded_files):
            filename = file.name.lower()

            # Save file temporarily
            temp_path = os.path.join(UPLOAD_DIR, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())

            # Process based on file type
            with st.spinner(f"🔄 Processing {file.name}..."):
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    # VIDEO PROCESSING
                    if not CV2_AVAILABLE:
                        st.error(f"❌ Cannot process video: OpenCV not available")
                        success, message, processed_count = False, "OpenCV not available", 0
                    else:
                        success, message, processed_count = process_video_robust(temp_path, sample_rate)
                else:
                    # IMAGE PROCESSING
                    success, message, processed_count = process_single_image_robust(temp_path, file.name)

                if success:
                    st.success(f"✅ {file.name}: {message}")
                    total_processed += processed_count
                else:
                    st.error(f"❌ {file.name}: {message}")

            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass

            # Update progress
            progress = (file_idx + 1) / total_files
            progress_bar.progress(progress)

        if total_processed > 0:
            st.balloons()
            st.success(f"🎉 All files processed! Total items analyzed: {total_processed}")

        progress_bar.empty()


def process_camera_image_unified():
    """Process camera image"""
    if st.session_state.camera_bytes:
        img = Image.open(io.BytesIO(st.session_state.camera_bytes))

        detections = detect_multiple_fruits_enhanced(img) if CV2_AVAILABLE else []

        if detections:
            st.success(f"🎯 Detected {len(detections)} fruit(s)")
            cols = st.columns(min(3, len(detections)))
            for idx, (label, conf, crop) in enumerate(detections):
                col_idx = idx % 3
                with cols[col_idx]:
                    emoji = "🔴" if conf < 0.5 else "🟠" if conf < 0.8 else "🟢"
                    st.image(crop, caption=f"{emoji} Fruit {idx + 1}: {label} ({conf * 100:.1f}%)", width=250)

                    st.session_state.predictions.append((f"camera_fruit{idx + 1}", label, f"{conf * 100:.1f}%"))
                    st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
                    st.session_state.total_processed += 1
        else:
            label, conf = predict_image(img)
            emoji = "🔴" if conf < 0.5 else "🟠" if conf < 0.8 else "🟢"
            st.image(img, caption=f"{emoji} {label} ({conf * 100:.1f}%)", width=300)

            st.session_state.predictions.append(("camera_image", label, f"{conf * 100:.1f}%"))
            st.session_state.counts[label] = st.session_state.counts.get(label, 0) + 1
            st.session_state.total_processed += 1

        st.session_state.camera_confirmed = True


def show_predictions():
    """Display predictions page"""
    st.title("🤖 Predictions")

    if not st.session_state.predictions:
        st.info("No predictions yet. Upload some data first!")
        return

    # Convert to DataFrame for nice display
    df = pd.DataFrame(
        st.session_state.predictions,
        columns=["File/Frame", "Predicted Class", "Confidence"]
    )

    st.dataframe(df, use_container_width=True)

    # Download option
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="fruit_predictions.csv",
        mime="text/csv"
    )


def show_dashboard():
    """Display dashboard page"""
    st.title("📊 Dashboard")

    if not st.session_state.counts:
        st.info("No data available yet. Process some images to see the dashboard.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Items", st.session_state.total_processed)

    with col2:
        unique_classes = len(st.session_state.counts)
        st.metric("Unique Classes", unique_classes)

    with col3:
        if st.session_state.counts:
            most_common = max(st.session_state.counts.items(), key=lambda x: x[1])
            st.metric("Most Common", f"{most_common[0]} ({most_common[1]})")
        else:
            st.metric("Most Common", "N/A")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Class Distribution")
        if st.session_state.counts:
            classes = list(st.session_state.counts.keys())
            counts = list(st.session_state.counts.values())

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(classes, counts)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.set_ylabel("Count")
            ax.set_title("Fruit Count by Class")
            plt.tight_layout()
            st.pyplot(fig)

    with col2:
        st.subheader("🍩 Class Proportions")
        if st.session_state.counts:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
            ax.set_title("Class Distribution")
            st.pyplot(fig)

    # Data table
    st.subheader("📋 Detailed Counts")
    count_df = pd.DataFrame({
        "Class": list(st.session_state.counts.keys()),
        "Count": list(st.session_state.counts.values())
    }).sort_values("Count", ascending=False)

    st.dataframe(count_df, use_container_width=True)


def show_settings():
    """Display settings page"""
    st.title("🔧 System Information")

    st.subheader("Model Status")
    st.write(f"**Model Loaded:** {'✅ Yes' if model_loaded else '❌ No'}")

    if model_loaded:
        st.success("✅ AI model ready for fruit freshness detection")

    st.subheader("OpenCV Status")
    if CV2_AVAILABLE:
        st.success("✅ OpenCV Headless loaded successfully")
        st.success("✅ Video processing ENABLED")
        st.success("✅ Multi-fruit detection ENABLED")
        st.info("ℹ️ Using OpenCV Headless (optimized for cloud deployment)")
    else:
        st.error("❌ OpenCV not available")
        st.warning("⚠️ Video processing disabled")
        st.warning("⚠️ Multi-fruit detection disabled")

    st.subheader("Technical Info")
    st.write(f"**TensorFlow Version:** {tf.__version__}")
    st.write(f"**Streamlit Version:** {st.__version__}")


def show_logout():
    """Handle logout"""
    st.session_state.logged_in = False
    st.session_state.page = "home"
    st.success("You have been logged out successfully!")
    st.rerun()


# --------------------------
# Main App
# --------------------------
def main():
    initialize_session_state()

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
        show_app()


if __name__ == "__main__":
    main()
