import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Define the classes
classes = ['1st Molar', '1st Premolar', '2nd Molar', '2nd Premolar', 'Canine', 'Central Incisor', 'Lateral Incisor']

# Fixed colors for each class
class_colors = {
    '1st Molar': 'red',
    '1st Premolar': 'blue',
    '2nd Molar': 'green',
    '2nd Premolar': 'orange',
    'Canine': 'purple',
    'Central Incisor': 'cyan',
    'Lateral Incisor': 'magenta'
}

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

# Perform detection and return results above threshold
def detect_and_plot(image, model):
    results = model.predict(image)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    detected_classes = {}
    
    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = int(detection.cls[0].cpu().numpy())

        if conf >= 0.5:
            class_name = classes[cls]
            color = class_colors.get(class_name, 'white')

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, f"{class_name} {conf:.2f}",
                     color='black', fontsize=10, backgroundcolor=color)
            
            if class_name not in detected_classes:
                detected_classes[class_name] = 0
            detected_classes[class_name] += 1

    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf, detected_classes

# Function to get Gemini response for each detected class
def get_gemini_response(teeth_counts):
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = "Provide a short overview for each tooth type listed below. For each, include:\n" \
             "- A brief description of the tooth and its function.\n\n"
    for tooth, count in teeth_counts.items():
        prompt += f"- {tooth} detected\n"

    prompt += "\nFormat the response in markdown with headers and bullet points."

    response = model.generate_content(prompt)
    return response.text.strip()

# Function to handle general queries (optional, for future)
def get_gemini_response_for_query(user_query):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are a knowledgeable dental health assistant. Help users with accurate information about:
- Tooth anatomy and function
- Dental health and care
- Common dental conditions

Reject questions unrelated to dental topics with: "I'm sorry, I can only answer dental-related questions."

**User's question:** {user_query}
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit app setup
st.set_page_config(page_title="Teeth Detection and Classification & Counting System in Clinical Oral Photographs", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF0800;'>Teeth Detection and Classification & Counting System in Clinical Oral Photographs</h1>", unsafe_allow_html=True)

# Initialize session state for chat visibility and history
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Image upload and processing section
st.subheader("Upload Dental Image:")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.convert("RGB")
    
    #image = image.resize((x, x))

    st.subheader("Uploaded Image:")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    image_np = np.array(image)
    
    model_path = "dental_yolo_model.pt"  # Update this path to your teeth model
    model = load_model(model_path)
    
    if model is not None:
        result_plot, detected_teeth = detect_and_plot(image_np, model)
        
        st.subheader("Detection Results:")
        st.image(result_plot, caption='Detection Results', use_container_width=True)
        
        if detected_teeth:
            st.subheader("Detected Tooth Types:")
            for tooth, count in detected_teeth.items():
                st.markdown(f"- **{tooth}**: {count} detected")

            st.markdown("---")
            st.subheader("Teeth Information:")

            teeth_info = get_gemini_response(detected_teeth)
            st.markdown(teeth_info)

        else:
            st.subheader("No confident detection made. Please try with a clearer image.")

# Chat assistant section
if st.session_state.chat_visible:
    st.title("Dental Health Chat Assistant")
    st.write("Ask me anything about teeth or dental health!")

    for user_input, bot_response in st.session_state.chat_history:
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")
    
    user_input = st.text_input("Enter your dental query here:", key="chat_input")

    if user_input:
        bot_response = get_gemini_response_for_query(user_input)
        st.session_state.chat_history.append((user_input, bot_response))
        
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.chat_visible = False

else:
    if st.button("Start Chat Assistant"):
        st.session_state.chat_visible = True

st.empty()
