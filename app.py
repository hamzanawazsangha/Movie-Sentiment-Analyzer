import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from huggingface_hub import hf_hub_download
import os
import time

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

# Set page config
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #333333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .result-box {
            padding: 2rem;
            border-radius: 10px;
            margin-top: 2rem;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .positive {
            background-color: #d4edda;
            color: #155724;
            border-left: 5px solid #28a745;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
            border-left: 5px solid #dc3545;
        }
        .title {
            color: #4a4a4a;
        }
        .header {
            background-color: #6c757d;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Safe NLTK data download
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Load resources from Hugging Face Hub
@st.cache_resource
def load_resources():
    try:
        # Download tokenizer
        # tokenizer_path = hf_hub_download(
        #     repo_id="HamzaNawaz17/MovieSentimentAnalyzer",
        #     filename="tokenizer.pickle"
        # )
        tokenizer_path ="tokenizer.pickle"
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Download model
        model_path = hf_hub_download(
            repo_id="HamzaNawaz17/MovieSentimentAnalyzer",
            filename="final_lstm_model.keras"
        )
        model = tf.keras.models.load_model(model_path)
        
        # Get max_len (default to 200 if metadata not available)
        max_len = 200
        try:
            metadata_path = hf_hub_download(
                repo_id="HamzaNawaz17/MovieSentimentAnalyzer",
                filename="metadata.pickle"
            )
            with open(metadata_path, 'rb') as handle:
                metadata = pickle.load(handle)
                max_len = metadata.get('max_len', 200)
        except:
            pass
            
        return tokenizer, model, max_len
        
    except Exception as e:
        st.error(f"Error loading model files from Hugging Face: {str(e)}")
        st.stop()

# Main app
def main():
    st.markdown("<div class='header'><h1 style='text-align: center;'>🎬 Movie Review Sentiment Analyzer</h1></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <h3 class='title'>Analyze Movie Reviews</h3>
        <p>This app predicts whether a movie review is positive or negative using a deep learning model.</p>
        <p>Enter your movie review text below and click 'Analyze' to see the sentiment prediction.</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://cdn.pixabay.com/photo/2015/11/22/19/04/camera-1056511_640.jpg", width=200)
    
    # Initialize NLTK data
    with st.spinner("Loading NLP resources..."):
        download_nltk_data()
    
    # Text input
    review_text = st.text_area("Enter your movie review here:", height=150, key="review_input")
    
    if st.button("Analyze Sentiment", key="analyze_btn"):
        if not review_text.strip():
            st.warning("Please enter a movie review to analyze.")
        else:
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Load resources when needed
                    tokenizer, model, max_len = load_resources()
                    
                    # Clean and preprocess the text
                    cleaned_text = clean_text(review_text)
                    
                    # Tokenize and pad
                    sequence = tokenizer.texts_to_sequences([cleaned_text])
                    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
                    
                    # Make prediction
                    prediction = model.predict(padded_sequence)
                    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
                    confidence = prediction[0][0] if sentiment == "positive" else 1 - prediction[0][0]
                    
                    # Display result
                    st.markdown(f"""
                    <div class='result-box {sentiment}'>
                        <h3>Sentiment Analysis Result</h3>
                        <p><strong>Prediction:</strong> {sentiment.capitalize()}</p>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Processed Text:</strong> {cleaned_text[:200]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show some emoji based on sentiment
                    if sentiment == "positive":
                        st.balloons()
                        st.success("😊 Great! This seems like a positive review!")
                    else:
                        st.warning("😟 Hmm, this review seems negative.")
                        
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

    # About section
    with st.expander("ℹ️ About this model"):
        st.markdown("""
        **Model Details:**
        - Architecture: LSTM Neural Network
        - Training Data: IMDB Movie Reviews (50,000 samples)
        - Accuracy: ~86.7% on test data
        - Max Sequence Length: 200 tokens
        - Hosted on: [Hugging Face Hub](https://huggingface.co/HamzaNawaz17/MovieSentimentAnalyzer)
        
        **Text Processing:**
        - HTML tags removed
        - Non-alphabetic characters removed
        - Converted to lowercase
        - Stopwords removed
        - Words lemmatized
        
        **How it works:**
        1. Your input text is cleaned and preprocessed
        2. The text is converted to numerical sequences
        3. The trained LSTM model predicts sentiment
        4. Results are displayed with confidence score
        """)

if __name__ == "__main__":
    main()
