import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
from gtts import gTTS, gTTSError
from deep_translator import GoogleTranslator
import os
import base64
import tempfile
import gdown
import requests
import json
import time

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="🌿 AI Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for Amazing UI
# ----------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #2E8B57, #32CD32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        border: 2px dashed #32CD32;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f0f8f0, #e8f5e8);
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #32CD32;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #32CD32, #228B22);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .voice-controls {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #32CD32;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .disease-tag {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 3px 6px rgba(255,68,68,0.3);
        border: 2px solid #fff;
    }
    
    .healthy-tag {
        background: linear-gradient(135deg, #32CD32, #228B22);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 3px 6px rgba(50,205,50,0.3);
        border: 2px solid #fff;
    }
    
    .feature-box {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Class Labels
# ----------------------------
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Enhanced Loading Functions
# ----------------------------
def show_loading_spinner(message="Processing..."):
    """Display a loading spinner with message"""
    loading_placeholder = st.empty()
    loading_placeholder.markdown(f"""
    <div class="loading-spinner">
        <div>
            <div class="spinner"></div>
            <p style="margin-top: 1rem; color: #666;">{message}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return loading_placeholder

@st.cache_resource
def load_model():
    """Load the Swin Transformer model with caching"""
    model_path = "swin_model.pth"
    if not os.path.exists(model_path):
        st.info("🔄 Downloading AI model for the first time...")
        url = "https://drive.google.com/uc?id=1_I9YMROXu1mXQhiBiGxOrcAbG8-rJ1IL"
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")
    
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ----------------------------
# Image Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict(model, image):
    """Predict disease from image"""
    img = image.convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        return class_names[pred_idx], probs[0, pred_idx].item()

# ----------------------------
# Enhanced Gemini API Integration
# ----------------------------
GEMINI_API_KEY = "AIzaSyBR2iai4tQVwTYBKmhOSCUUlhK1ulkHyyQ"

def get_gemini_summary(disease_name):
    """Get concise disease summary from Gemini API"""
    if "healthy" in disease_name.lower():
        plant_type = disease_name.split('___')[0].replace('_', ' ').title()
        return f"✅ Great news! Your {plant_type} leaf appears healthy. No disease detected. Continue with regular care and monitoring."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # Enhanced prompt for concise summary
    prompt = f"""Provide a brief, practical summary about {disease_name} plant disease in exactly 3 sentences:
1. What it is and main symptoms
2. Primary causes
3. Key treatment/prevention methods
Keep it under 100 words and farmer-friendly."""
    
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            res_json = response.json()
            summary = res_json['candidates'][0]['content']['parts'][0]['text']
            return summary
        else:
            return f"⚠️ Disease detected: {disease_name.replace('_', ' ').title()}. Please consult agricultural experts for proper diagnosis and treatment."
    except Exception as e:
        return f"⚠️ Disease detected: {disease_name.replace('_', ' ').title()}. Please consult agricultural experts for proper diagnosis and treatment."

# ----------------------------
# Enhanced Translation and Voice
# ----------------------------
@st.cache_data
def translate_text(text, target_lang):
    """Translate text using deep_translator GoogleTranslator with caching"""
    if not text or target_lang == 'en':
        return text
    
    try:
        # Use GoogleTranslator from deep_translator
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}")
        return text

def generate_voice(text, lang_code):
    """Generate voice with error handling"""
    try:
        if not text:
            return None
        
        # Clean text for TTS
        clean_text = text.replace('*', '').replace('#', '').replace('✅', '').replace('⚠️', '')
        
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except gTTSError as e:
        st.error(f"Voice generation failed: {str(e)}")
        return None

def get_audio_player(audio_path):
    """Create audio player HTML"""
    if not audio_path or not os.path.exists(audio_path):
        return ""
    
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <div class="voice-controls">
        <h4>🔊 Voice Summary</h4>
        <audio controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
    </div>
    """
    return audio_html

# ----------------------------
# Main UI
# ----------------------------
def main():
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Header
    st.markdown('<h1 class="main-header">🌿 AI Plant Disease Diagnosis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a leaf image to get instant disease diagnosis with multilingual voice support</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🌍 Language Settings")
        
        language_map = {
            "English": "en",
            "Telugu": "te", 
            "Hindi": "hi",
            "Tamil": "ta"
        }
        
        selected_lang = st.selectbox(
            "Select Language", 
            list(language_map.keys()),
            help="Choose your preferred language for results and voice"
        )
        
        st.markdown("---")
        
        # Enhanced features with deep_translator
        st.header("✨ Features")
        st.markdown("""
        - 🤖 **AI-Powered** Swin Transformer
        - 🌐 **Multilingual** Support (deep_translator)
        - 🔊 **Voice** Summaries
        - 📱 **Mobile** Friendly
        - ⚡ **Real-time** Analysis
        - ☁️ **Cloud** Optimized
        """)
        
        st.markdown("---")
        
        # Supported plants
        st.header("🌱 Supported Plants")
        st.markdown("""
        - 🌶️ **Bell Pepper**
        - 🥔 **Potato**
        - 🍅 **Tomato**
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📸 Upload Plant Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the plant leaf"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📱 Uploaded Image", use_container_width=True)
    
    with col2:
        if uploaded_file:
            st.markdown("### 🧪 Analysis Options")
            
            analyze_button = st.button(
                "🔍 Analyze Disease", 
                type="primary",
                use_container_width=True
            )
            
            if analyze_button:
                # Reset voice state for new analysis
                st.session_state.voice_generated = False
                st.session_state.analysis_complete = False
                
                # Load model with progress
                with st.spinner("🤖 Loading AI model..."):
                    model = load_model()
                
                # Prediction with progress
                loading_placeholder = show_loading_spinner("🔬 Analyzing image...")
                time.sleep(1)  # Simulate processing time
                
                label, confidence = predict(model, image)
                loading_placeholder.empty()
                
                # Mark analysis as complete
                st.session_state.analysis_complete = True
                
                # Results
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Display prediction with better formatting
                crop_name = label.split('___')[0].replace('_', ' ').title()
                disease_name = label.split('___')[1].replace('_', ' ').title()
                
                st.markdown("### 🔍 Diagnosis Result")
                
                if "healthy" in disease_name.lower():
                    st.success(f"✅ **Healthy {crop_name}** - No disease detected!")
                    st.markdown(f'<div style="text-align: center; margin: 1rem 0;"><span class="healthy-tag">✅ Healthy {crop_name}</span></div>', unsafe_allow_html=True)
                else:
                    st.error(f"⚠️ **{disease_name}** detected in {crop_name}")
                    st.markdown(f'<div style="text-align: center; margin: 1rem 0;"><span class="disease-tag">⚠️ {disease_name}</span></div>', unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown(f"**Confidence: {confidence*100:.1f}%**")
                confidence_html = f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence*100}%;"></div>
                </div>
                """
                st.markdown(confidence_html, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Get summary
                with st.spinner("🧠 Generating summary..."):
                    gemini_summary = get_gemini_summary(label)
                
                # Translate if needed
                if selected_lang != "English":
                    with st.spinner(f"🌍 Translating to {selected_lang}..."):
                        translated = translate_text(gemini_summary, language_map[selected_lang])
                else:
                    translated = gemini_summary
                
                # Display summary
                st.markdown("### 📋 Summary")
                st.info(translated)
                
                # Voice controls with better state management
                st.markdown("### 🔊 Voice Summary")
                
                # Initialize session state for voice controls
                if 'auto_play_voice' not in st.session_state:
                    st.session_state.auto_play_voice = True
                if 'voice_generated' not in st.session_state:
                    st.session_state.voice_generated = False
                
                col_voice1, col_voice2 = st.columns([1, 1])
                
                with col_voice1:
                    generate_voice_btn = st.button("🎵 Generate Voice", use_container_width=True)
                
                with col_voice2:
                    # Use session state for auto-play to prevent summary closure
                    auto_play = st.checkbox(
                        "🔄 Auto-play voice", 
                        value=st.session_state.auto_play_voice,
                        key="auto_play_checkbox"
                    )
                    st.session_state.auto_play_voice = auto_play
                
                # Generate voice based on button click or auto-play
                should_generate_voice = generate_voice_btn or (auto_play and not st.session_state.voice_generated)
                
                if should_generate_voice:
                    with st.spinner("🎙️ Generating voice..."):
                        voice_path = generate_voice(translated, language_map[selected_lang])
                    
                    if voice_path:
                        audio_html = get_audio_player(voice_path)
                        st.markdown(audio_html, unsafe_allow_html=True)
                        st.session_state.voice_generated = True
                        
                        # Cleanup
                        if os.path.exists(voice_path):
                            os.remove(voice_path)
                    else:
                        st.warning("🔇 Voice generation failed. Please try again.")
                elif st.session_state.voice_generated and not auto_play:
                    st.info("🔇 Auto-play disabled. Click 'Generate Voice' to play again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Built with ❤️ using Swin Transformer + Gemini AI + Google TTS + deep_translator</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()