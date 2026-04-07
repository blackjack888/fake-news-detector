import streamlit as st
import pickle
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import train_model
from transformers import pipeline # <-- NEW IMPORT

# Load Model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except:
    st.error("Model not found. Run 'train_model.py' first.")
    st.stop()

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("🕵️ Fake News Detector")

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

tab1, tab2 = st.tabs(["Paste Text", "Check URL"])

with tab1:
    raw_text = st.text_area("Enter News Article:", height=200)
    if st.button("Check Text"):
        st.session_state.input_text = raw_text

with tab2:
    url = st.text_input("Enter News URL:")
    if st.button("Check URL"):
        if url:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'https://www.google.com/'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = soup.find('h1').get_text().strip() if soup.find('h1') else ""
                paragraphs = soup.find_all('p')
                text_content = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50]
                scraped_text = title + " " + ' '.join(text_content)
                scraped_text = re.sub(r'\s+', ' ', scraped_text).strip()

                if not scraped_text:
                    st.warning("Could not extract text. The website might be using JavaScript.")
                else:
                    st.info(f"Extracted {len(scraped_text)} characters from URL.")
                    st.session_state.input_text = scraped_text
                    with st.expander("Show Scraped Text"):
                        st.write(scraped_text)
            except Exception as e:
                st.error(f"Error fetching URL: {e}")

# --- TIER 1: FAST PAC PREDICTION ---
if st.session_state.input_text:
    vec_text = vectorizer.transform([st.session_state.input_text])
    prediction = model.predict(vec_text)
    
    st.markdown("---")
    st.subheader("⚡ Tier 1: Real-Time Linguistic Analysis (PAC)")
    
    if prediction[0] == 1:
        st.error("🚨 **RESULT: FAKE NEWS Detected**")
        
        with st.expander("Incorrect Prediction? Teach the Model"):
            st.write("If this is actually Real News, click below to retrain the local model.")
            if st.button("Mark as Real & Retrain"):
                new_row = pd.DataFrame({'title': [" "], 'text': [st.session_state.input_text], 'subject': ['manual'], 'date': ['2025']})
                if os.path.exists('True.csv'):
                    new_row.to_csv('True.csv', mode='a', header=False, index=False)
                    st.info("Added to dataset. Retraining model...")
                    train_model.train()
                    st.success("Model Retrained! Refresh to verify.")
                    st.rerun()
                else:
                    st.error("Could not find 'True.csv' to update.")
    else:
        st.success("✅ **RESULT: REAL News**")

    # Explainable AI Section
    with st.expander("🧠 Why did the AI make this decision? (Word Impact)"):
        feature_names = vectorizer.get_feature_names_out()
        input_vector = vec_text.toarray()[0] 
        word_impact = []
        for i in range(len(input_vector)):
            if input_vector[i] > 0: 
                word = feature_names[i]
                weight = model.coef_[0][i] 
                impact_score = input_vector[i] * weight
                word_impact.append((word, impact_score))
                
        word_impact.sort(key=lambda x: x[1])
        if len(word_impact) > 0:
            df_impact = pd.DataFrame(word_impact, columns=['Word', 'Impact Score'])
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 'Fake' Indicators:**")
                st.dataframe(df_impact.head(5).style.background_gradient(cmap='Reds'))
            with col2:
                st.write("**Top 'Real' Indicators:**")
                st.dataframe(df_impact.tail(5).sort_values(by='Impact Score', ascending=False).style.background_gradient(cmap='Greens'))

    # --- TIER 2: DEEP LEARNING (BERT) ---
    st.markdown("---")
    st.subheader("🔬 Tier 2: Deep Context Verification (BERT)")
    st.write("Our PAC model is optimized for sub-second filtering. For highly ambiguous articles, run a deep-learning transformer model to analyze the full context.")
    
    if st.button("Run Deep Analysis (BERT)"):
        with st.spinner("Loading BERT Transformer Model... (This takes a few seconds)"):
            try:
                # We use a tiny BERT fine-tuned for fake news to prevent Streamlit from crashing
                bert_analyzer = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
                
                # Transformers have a limit, so we analyze the first 500 characters
                short_text = st.session_state.input_text[:500] 
                bert_result = bert_analyzer(short_text)[0]
                
                label = bert_result['label']
                confidence = round(bert_result['score'] * 100, 2)
                
                if label == "FAKE" or label == "LABEL_1": # Some models use LABEL_1 for fake
                    st.error(f"**BERT Conclusion:** FAKE (Confidence: {confidence}%)")
                else:
                    st.success(f"**BERT Conclusion:** REAL (Confidence: {confidence}%)")
                    
            except Exception as e:
                st.error(f"Error loading BERT: {e}")