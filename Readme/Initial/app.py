import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import cv2
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from keras.applications.convnext import LayerScale
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(layout="wide")

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username == "Admin" and password == "123":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def home_page():
    st.title("AI Powered Crop Disease Detection")

    st.write("""
    This AI-powered system helps farmers, agronomists, and researchers detect diseases in crop images using deep learning.  
    It supports multiple crops and diseases with high accuracy.

    - Upload an image of a plant leaf  
    - Choose a deep learning model  
    - Get instant predictions about plant health

    ### Deep Learning Models Used
    - *ResNet101*
    - *ConvNeXtBase*
    - *DenseNet121*

    ### Supported Crop Classes & Diseases
    Below is a list of all supported crop diseases and healthy conditions, along with sample images and brief descriptions.
    """)

    disease_info = [
        {
            "title": "Corn - Healthy",
            "img": "examples/corn_healthy.jpg",
            "desc": "A healthy corn plant has upright, green leaves with no visible signs of disease. The leaf surface is smooth and free of spots or distortions, indicating strong photosynthetic activity and optimal growth conditions. Healthy corn leads to higher yields and resilience against pests and environmental stressors."
        },
        {
            "title": "Corn - Common Rust",
            "img": "examples/corn_common_rust.jpg",
            "desc": "Common rust is caused by the fungus Puccinia sorghi. It forms reddish-brown pustules on both leaf surfaces, often aligned in rows. Severe infection can reduce photosynthesis, weaken the plant, and reduce crop yield significantly. It spreads through airborne spores and is more prevalent in humid, warm conditions."
        },
        {
            "title": "Corn - Gray Leaf Spot",
            "img": "examples/corn_gray_leaf_spot.jpg",
            "desc": "Gray Leaf Spot is caused by the fungus Cercospora zeae-maydis. It produces narrow, rectangular, grayish lesions on the leaves. This disease disrupts photosynthesis and leads to premature leaf death, reducing grain fill and overall yield. It often thrives in no-till fields with high humidity and leaf moisture."
        },
        {
            "title": "Potato - Healthy",
            "img": "examples/potato_healthy.jpg",
            "desc": "Healthy potato plants exhibit vigorous green foliage with uniform leaf structure and no signs of wilting, yellowing, or lesions. Such plants grow rapidly, develop strong tubers, and are more resistant to pest or disease invasion under proper care and soil conditions."
        },
        {
            "title": "Potato - Bacterial Wilt",
            "img": "examples/potato_bacterial_wilt.jpg",
            "desc": "Bacterial Wilt is a serious disease caused by Ralstonia solanacearum. It leads to rapid wilting of leaves during the hottest part of the day, even when soil moisture is sufficient. Internally, vascular tissues appear brown. The disease spreads via contaminated soil and water, affecting tuber quality and storage life."
        },
        {
            "title": "Potato - Leafroll Virus",
            "img": "examples/potato_leafroll_virus.jpg",
            "desc": "Potato Leafroll Virus (PLRV) is a persistent virus transmitted by aphids. Symptoms include upward curling of leaves, chlorosis, and stunted growth. The virus impairs phloem transport, leading to poor tuber development and reduced yield. It's particularly problematic when seed potatoes are infected."
        },
        {
            "title": "Sugarcane - Healthy",
            "img": "examples/sugarcane_healthy.jpg",
            "desc": "A healthy sugarcane plant is tall and sturdy with bright green leaves and thick, juicy stalks. The leaves show no yellowing or necrosis, and the stalks are disease-free. Proper irrigation, fertilization, and pest control help maintain healthy crops and maximize sucrose content."
        },
        {
            "title": "Sugarcane - Red Rot",
            "img": "examples/sugarcane_red_rot.jpg",
            "desc": "Red Rot, caused by Colletotrichum falcatum, is one of the most destructive sugarcane diseases. It causes red discoloration with white patches inside stalks, leading to poor juice quality and plant death. Infected plants may show drying of the top leaves and stalk breakage. It spreads through infected setts and residues."
        },
        {
            "title": "Sugarcane - Yellow Leaf",
            "img": "examples/sugarcane_yellow_leaf.jpg",
            "desc": "Yellow Leaf Disease is caused by a virus spread primarily through aphids. Symptoms include progressive yellowing along the midrib, leaf tip drying, and sometimes dwarfing of plants. It affects photosynthesis and cane weight, leading to a noticeable drop in sugar recovery."
        },
        {
            "title": "Tomato - Healthy",
            "img": "examples/tomato_healthy.jpg",
            "desc": "Healthy tomato plants have bright green, well-shaped leaves and sturdy stems. There's no curling, discoloration, or spotting. Healthy plants flower and fruit efficiently, leading to high-quality tomatoes with minimal pest or disease intervention when grown in ideal conditions."
        },
        {
            "title": "Tomato - Leaf Curl",
            "img": "examples/tomato_leaf_curl.jpg",
            "desc": "Tomato Leaf Curl is caused by a virus, typically spread by whiteflies. It leads to severe upward curling of leaves, stunted growth, reduced flowering, and poor fruit development. The virus can devastate entire fields if not controlled early with vector management and resistant varieties."
        },
        {
            "title": "Tomato - Septoria Leaf Spot",
            "img": "examples/tomato_septoria_leaf_spot.jpg",
            "desc": "Septoria Leaf Spot is a fungal disease that begins with small, water-soaked spots which enlarge into circular lesions with dark brown borders and pale centers. It typically affects lower leaves first and gradually spreads upward. It reduces leaf area, impacting fruit production and quality."
        }
    ]

    for idx, item in enumerate(disease_info):
        padding_left, content, padding_right = st.columns([1, 6, 1])
        with content:
            if idx % 2 == 0:
                img_col, text_col = st.columns([1.5, 2])
            else:
                text_col, img_col = st.columns([2, 1.5])

            with img_col:
                st.image(item["img"], caption=item["title"], width=300)

            with text_col:
                st.subheader(item["title"])
                st.write(item["desc"])

        st.markdown("<hr style='margin:30px 0'>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    resnet_model = load_model("models/ResNet_101.keras")
    convnext_model = tf.keras.models.load_model(
        "models/ConvNeXtBase.keras",
        custom_objects={"LayerScale": LayerScale}
    )
    densenet_model = load_model("models/DenseNet_121.keras")

    return {
        "ResNet101": resnet_model,
        "ConvNeXtBase": convnext_model,
        "DenseNet121": densenet_model
    }

models_dict = load_models()

@st.cache_data(show_spinner=False)
def get_disease_advice_openrouter(disease_name: str) -> str:
    prompt = f"""
    I just diagnosed a plant with the following disease: {disease_name}.
    
    Please explain:
    1. What this disease is and how it affects the plant
    2. How to treat it (include chemical and organic options if available)
    3. How to prevent it from occurring again
    Keep it clear and useful for a farmer.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourapp.com",
        "X-Title": "AI Crop Doctor"
    }

    data = {
        "model": "tngtech/deepseek-r1t2-chimera:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers,
                                 data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Failed to fetch advice: {str(e)}"
    
def predict_image(image_file, model_name):
    model = models_dict[model_name]

    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_label = label_encoder.inverse_transform([np.argmax(pred)])
    return pred_label[0]

def prediction_page():
    st.title("Predict Crop Disease from Image")

    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    model_choice = st.radio("Select Model", ("ResNet101", "ConvNeXtBase", "DenseNet121"))

    if st.button("Predict"):
        if image_file is not None:
            with st.spinner("Predicting..."):
                result = predict_image(image_file, model_choice)
                st.image(image_file, caption="Uploaded Image")

                if "healthy" in result.lower():
                    st.success(f"Prediction: *{result}*")
                else:
                    st.error(f"Prediction: *{result}*")

                    with st.spinner("Fetching expert advice..."):
                        advice = get_disease_advice_openrouter(result)
                        st.subheader("Expert Advice & Remedies")
                        st.write(advice)
        else:
            st.warning("Please upload an image.")

def camera_capture_page():
    st.title("Capture Image via Camera")

    st.write("Use your device camera to capture a leaf image for disease prediction.")

    captured_image = st.camera_input("Take a photo")

    model_choice = st.radio("Select Model", ("ResNet101", "ConvNeXtBase", "DenseNet121"))

    if captured_image is not None:
        if st.button("Predict from Camera Image"):
            with st.spinner("Analyzing captured image..."):
                result = predict_image(captured_image, model_choice)
                st.image(captured_image, caption="Captured Image")

                if "healthy" in result.lower():
                    st.success(f"Prediction: *{result}*")
                else:
                    st.error(f"Prediction: *{result}*")

                    with st.spinner("Fetching expert advice..."):
                        advice = get_disease_advice_openrouter(result)
                        st.subheader("Expert Advice & Remedies")
                        st.write(advice)

def evaluation_metrics_page():
    st.title("Evaluation Metrics")

    # Accuracy plots
    st.subheader("Accuracy Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Training Accuracy")
        st.image("metrics/training_accuracy.png", use_column_width=True)
    with col2:
        st.markdown("#### Validation Accuracy")
        st.image("metrics/validation_accuracy.png", use_column_width=True)

    # Loss plots
    st.subheader("Loss Comparison")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Training Loss")
        st.image("metrics/training_loss.png", use_column_width=True)
    with col4:
        st.markdown("#### Validation Loss")
        st.image("metrics/validation_loss.png", use_column_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown("#### ResNet101")
        st.image("metrics/cm_resnet101.png", use_column_width=True)
    with col6:
        st.markdown("#### ConvNeXtBase")
        st.image("metrics/cm_convnextbase.png", use_column_width=True)
    with col7:
        st.markdown("#### DenseNet121")
        st.image("metrics/cm_densenet121.png", use_column_width=True)

def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "Camera Capture", "Evaluation Metrics"])

    # Logout button at the bottom
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    if page == "Home":
        home_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Camera Capture":
        camera_capture_page()
    elif page == "Evaluation Metrics":
        evaluation_metrics_page()

if not st.session_state.logged_in:
    login()
else:
    main_app()