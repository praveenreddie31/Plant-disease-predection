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
import speech_recognition as sr
import pyaudio
import wave
import numpy as np
from io import BytesIO

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
    
    .speech-section {
        background: #e8f4fd;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #3498db;
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
    
    .conversation-bubble {
        background: #f1f3f4;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
    
    .user-bubble {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .ai-bubble {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .recording-indicator {
        background: #ff4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
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

# Initialize speech recognizer
r = sr.Recognizer()

def get_gemini_summary(disease_name):
    """Get concise disease summary from Gemini 2.0 Flash API"""
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Enhanced prompt for more structured and concise response
        prompt = f"""Provide a concise summary (2-3 sentences) about {disease_name} in plants, including:
        1. Main characteristics
        2. Common symptoms
        3. Basic prevention/treatment
        
        Keep the response under 100 words and in simple language."""
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 200,
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Information about {disease_name} is not available at the moment."
    except Exception as e:
        st.error(f"Error getting Gemini response: {str(e)}")
        return f"Could not retrieve information about {disease_name}. Please try again later."

def get_gemini_response(question, context):
    """Get response from Gemini 2.0 Flash API for user questions"""
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        prompt = f"""You are a helpful plant disease assistant. 
        Context: {context}
        
        Question: {question}
        
        Please provide a clear, concise, and helpful answer based on the context above. 
        Focus on practical advice for farmers and gardeners.
        If the question is not related to plant diseases, politely redirect to plant health topics.
        Keep the response under 150 words."""
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': GEMINI_API_KEY
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 300,
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "I'm sorry, I couldn't generate a response. Please try again."
    except Exception as e:
        st.error(f"Error getting Gemini response: {str(e)}")
        return "I'm having trouble connecting to the AI service. Please try again later."

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
        clean_text = clean_text.replace('🌿', '').replace('🔍', '').replace('💡', '')
        
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except gTTSError as e:
        st.error(f"Voice generation failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error in voice generation: {str(e)}")
        return None

def get_audio_player(audio_path):
    """Create audio player HTML"""
    if not audio_path or not os.path.exists(audio_path):
        return ""
    
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <div class="voice-controls">
            <audio controls style="width: 100%;" autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        </div>
        """
        return audio_html
    except Exception as e:
        st.error(f"Error creating audio player: {str(e)}")
        return ""

# ----------------------------
# Speech Recognition Functions
# ----------------------------
def record_audio(duration=5):
    """Record audio from microphone and return the audio data"""
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=44100,
                       input=True,
                       frames_per_buffer=1024)
        
        frames = []
        
        # Record for specified duration
        for _ in range(0, int(44100 / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to a BytesIO object
        audio_data = BytesIO()
        wf = wave.open(audio_data, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        audio_data.seek(0)
        return audio_data
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

def transcribe_audio(audio_data, language="en"):
    """Transcribe audio to text"""
    try:
        if not audio_data:
            return ""
            
        with sr.AudioFile(audio_data) as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        
        # Map language codes to recognizer's language codes
        lang_map = {
            "en": "en-US",
            "hi": "hi-IN",
            "te": "te-IN",
            "ta": "ta-IN"
        }
        
        text = r.recognize_google(audio, language=lang_map.get(language, "en-US"))
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        st.error(f"Error with speech recognition service: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Error in speech recognition: {str(e)}")
        return ""

# ----------------------------
# Speech-to-Speech Handler
# ----------------------------
def handle_speech_to_speech(current_disease, selected_language, lang_code):
    """Handle the complete speech-to-speech interaction"""
    st.markdown('<div class="speech-section">', unsafe_allow_html=True)
    st.markdown("### 🎤 Speech-to-Speech Q&A")
    st.info("Ask questions about the diagnosed disease using your voice and get spoken responses!")
    
    # Instructions
    st.markdown("""
    **How to use:**
    1. Click the 'Start Voice Question' button
    2. Speak your question clearly (you have 5 seconds)
    3. Wait for the AI to process and respond
    4. Listen to the spoken answer
    """)
    
    # Initialize speech session state
    if 'speech_active' not in st.session_state:
        st.session_state.speech_active = False
    if 'processing_speech' not in st.session_state:
        st.session_state.processing_speech = False
    
    # Voice recording button
    voice_question_btn = st.button("🎤 Start Voice Question", use_container_width=True, type="primary", key="voice_question_btn")
    
    if voice_question_btn and not st.session_state.processing_speech:
        st.session_state.processing_speech = True
        st.session_state.speech_active = True
        
        # Recording phase
        recording_placeholder = st.empty()
        recording_placeholder.markdown(
            '<div class="recording-indicator">🔴 Recording... Speak now!</div>', 
            unsafe_allow_html=True
        )
        
        # Record audio
        audio_data = record_audio(duration=5)
        recording_placeholder.empty()
        
        if audio_data:
            # Transcription phase
            with st.spinner("🔍 Understanding your question..."):
                question = transcribe_audio(audio_data, lang_code)
            
            if question:
                # Display transcribed question
                st.markdown(f'<div class="conversation-bubble user-bubble"><strong>🎤 You asked:</strong> {question}</div>', unsafe_allow_html=True)
                
                # Get AI response
                context = f"""
                Plant: {current_disease['crop_name']}
                Disease: {current_disease['disease_name']}
                Status: {current_disease['status']}
                Summary: {current_disease['summary']}
                Confidence: {current_disease['confidence']}%
                """
                
                with st.spinner("🤖 Generating response..."):
                    response = get_gemini_response(question, context)
                
                # Translate response if needed
                if selected_language != "English":
                    with st.spinner(f"🌍 Translating to {selected_language}..."):
                        translated_response = translate_text(response, lang_code)
                else:
                    translated_response = response
                
                # Display AI response
                st.markdown(f'<div class="conversation-bubble ai-bubble"><strong>🤖 AI Response:</strong> {translated_response}</div>', unsafe_allow_html=True)
                
                # Generate and play voice response
                with st.spinner("🔊 Converting to speech..."):
                    voice_path = generate_voice(translated_response, lang_code)
                
                if voice_path:
                    st.success("🔊 Playing AI response...")
                    audio_html = get_audio_player(voice_path)
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
                    # Cleanup
                    try:
                        os.remove(voice_path)
                    except:
                        pass
                else:
                    st.error("❌ Could not generate voice response")
                
                # Add to conversation history
                if 'conversation_history' not in st.session_state:
                    st.session_state.conversation_history = []
                
                st.session_state.conversation_history.append({
                    'question': question,
                    'response': translated_response,
                    'timestamp': time.time()
                })
                
            else:
                st.error("❌ Could not understand your question. Please try again with clearer speech.")
        else:
            st.error("❌ Failed to record audio. Please check your microphone.")
        
        # Reset processing state
        st.session_state.processing_speech = False
    
    # Show conversation history
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        st.markdown("### 📝 Conversation History")
        for i, conv in enumerate(st.session_state.conversation_history[-3:]):  # Show last 3 conversations
            st.markdown(f"""
            <div class="conversation-bubble">
                <strong>Q{i+1}:</strong> {conv['question']}<br>
                <strong>A{i+1}:</strong> {conv['response']}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Main Application
# ----------------------------
def main():
    # Initialize session states
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'voice_generated' not in st.session_state:
        st.session_state.voice_generated = False
    if 'auto_play_voice' not in st.session_state:
        st.session_state.auto_play_voice = False
    if 'summary_generated' not in st.session_state:
        st.session_state.summary_generated = False
    if 'translated_summary' not in st.session_state:
        st.session_state.translated_summary = ""
    
    # Header
    st.markdown('<h1 class="main-header">🌿 AI Plant Disease Diagnosis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a leaf image to get instant disease diagnosis with speech-to-speech support</p>', unsafe_allow_html=True)
    
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
            help="Choose your preferred language for results and voice",
            key="language_selector"
        )
        
        # Check if language changed and reset voice state
        if 'previous_language' not in st.session_state:
            st.session_state.previous_language = selected_lang
        
        if selected_lang != st.session_state.previous_language:
            st.session_state.voice_generated = False
            st.session_state.previous_language = selected_lang
            # Re-translate summary if analysis is complete
            if st.session_state.analysis_complete and st.session_state.current_disease:
                if selected_lang != "English":
                    st.session_state.translated_summary = translate_text(
                        st.session_state.current_disease['summary'], 
                        language_map[selected_lang]
                    )
                else:
                    st.session_state.translated_summary = st.session_state.current_disease['summary']
        
        st.markdown("---")
        
        # Enhanced features
        st.header("✨ Features")
        st.markdown("""
        - 🤖 **AI-Powered** Swin Transformer
        - 🌐 **Multilingual** Support
        - 🎤 **Speech-to-Speech** Interaction
        - 🔊 **Voice** Summaries
        - 📱 **Mobile** Friendly
        - ⚡ **Real-time** Analysis
        """)
        
        st.markdown("---")
        
        # Supported plants
        st.header("🌱 Supported Plants")
        st.markdown("""
        - 🌶️ **Bell Pepper**
        - 🥔 **Potato**
        - 🍅 **Tomato**
        """)
        
        st.markdown("---")
        
        # Clear conversation history
        if st.button("🗑️ Clear Chat History", key="clear_history_btn"):
            st.session_state.conversation_history = []
            st.rerun()
    
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
                # Reset states for new analysis
                st.session_state.voice_generated = False
                st.session_state.analysis_complete = False
                st.session_state.conversation_history = []
                st.session_state.summary_generated = False
                st.session_state.translated_summary = ""
                
                # Load model
                with st.spinner("🤖 Loading AI model..."):
                    model = load_model()
                
                # Prediction
                loading_placeholder = show_loading_spinner("🔬 Analyzing image...")
                time.sleep(1)
                
                label, confidence = predict(model, image)
                loading_placeholder.empty()
                
                # Process results
                crop_name = label.split('___')[0].replace('_', ' ').title()
                disease_name = label.split('___')[1].replace('_', ' ').title()
                is_healthy = "healthy" in disease_name.lower()
                
                # Store current disease info
                st.session_state.current_disease = {
                    'crop_name': crop_name,
                    'disease_name': disease_name,
                    'status': 'Healthy' if is_healthy else 'Diseased',
                    'confidence': round(confidence * 100, 1),
                    'full_label': label,
                    'summary': ''
                }
                
                # Get and display summary
                with st.spinner("🧠 Generating summary..."):
                    gemini_summary = get_gemini_summary(label)
                    st.session_state.current_disease['summary'] = gemini_summary
                
                # Translate if needed
                if selected_lang != "English":
                    with st.spinner(f"🌍 Translating to {selected_lang}..."):
                        translated_summary = translate_text(gemini_summary, language_map[selected_lang])
                else:
                    translated_summary = gemini_summary
                
                # Store translated summary
                st.session_state.translated_summary = translated_summary
                st.session_state.summary_generated = True
                
                # Mark analysis as complete
                st.session_state.analysis_complete = True
    
    # Display results if analysis is complete (outside the button click)
    if st.session_state.analysis_complete and st.session_state.current_disease:
        with col2:
            # Display results
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Diagnosis Result")
            
            crop_name = st.session_state.current_disease['crop_name']
            disease_name = st.session_state.current_disease['disease_name']
            confidence = st.session_state.current_disease['confidence']
            is_healthy = st.session_state.current_disease['status'] == 'Healthy'
            
            if is_healthy:
                st.success(f"✅ **Healthy {crop_name}** - No disease detected!")
                st.markdown(f'<div style="text-align: center; margin: 1rem 0;"><span class="healthy-tag">✅ Healthy {crop_name}</span></div>', unsafe_allow_html=True)
            else:
                st.error(f"⚠️ **{disease_name}** detected in {crop_name}")
                st.markdown(f'<div style="text-align: center; margin: 1rem 0;"><span class="disease-tag">⚠️ {disease_name}</span></div>', unsafe_allow_html=True)
            
            # Confidence display
            st.markdown(f"**Confidence: {confidence}%**")
            confidence_html = f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%;"></div>
            </div>
            """
            st.markdown(confidence_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display summary
            if st.session_state.summary_generated:
                st.markdown("### 📋 Summary")
                st.info(st.session_state.translated_summary)
                
                # Voice-based Q&A Section
                st.markdown("---")
                st.markdown("### 🎤 Ask Questions About This Disease")
                st.info("Click the button below to ask a question about the disease using your voice.")
                
                # Initialize question state
                if 'question_asked' not in st.session_state:
                    st.session_state.question_asked = False
                
                # Create a form for the question
                with st.form("question_form"):
                    if st.form_submit_button("🎤 Ask a Question (Hold to Speak)", use_container_width=True, type="secondary"):
                        st.session_state.question_asked = True
                        
                if st.session_state.question_asked:
                    with st.spinner("🎙️ Listening... Speak now!"):
                        try:
                            # Record audio
                            audio_data = record_audio()
                            
                            # Transcribe audio
                            question = transcribe_audio(audio_data, language_map[selected_lang])
                            
                            if question:
                                st.success(f"🎤 You asked: {question}")
                                
                                # Get response from Gemini
                                context = f"""
                                Crop: {crop_name}
                                Disease: {disease_name}
                                Summary: {st.session_state.translated_summary}
                                """
                                
                                with st.spinner("🤔 Thinking..."):
                                    response = get_gemini_response(question, context, language_map[selected_lang])
                                    
                                    # Display response
                                    st.markdown("### 💡 Response")
                                    st.info(response)
                                    
                                    # Convert response to speech
                                    with st.spinner("🔊 Converting response to speech..."):
                                        response_voice_path = generate_voice(response, language_map[selected_lang])
                                        if response_voice_path:
                                            response_audio = get_audio_player(response_voice_path)
                                            st.markdown(response_audio, unsafe_allow_html=True)
                                            
                                            # Cleanup
                                            if os.path.exists(response_voice_path):
                                                os.remove(response_voice_path)
                            else:
                                st.warning("Could not understand your question. Please try again.")
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            st.error("Please try again or refresh the page.")
                        
                        # Reset the question state
                        st.session_state.question_asked = False
    
    # Speech-to-Speech Section (Full Width)
    if st.session_state.analysis_complete and st.session_state.current_disease:
        st.markdown("---")
        handle_speech_to_speech(
            st.session_state.current_disease, 
            selected_lang, 
            language_map[selected_lang]
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Built with ❤️ using Swin Transformer + Gemini AI + Speech Recognition + Google TTS</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()