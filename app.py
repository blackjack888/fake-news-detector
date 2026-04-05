import streamlit as st
import pickle
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import train_model

# Load Model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except:
    st.error("Model not found. Run 'train_model.py' first.")
    st.stop()

st.set_page_config(page_title="Fake News Detector")
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
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Referer': 'https://www.google.com/'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get Title (h1) and Text (p) to match training data format
                title = soup.find('h1').get_text().strip() if soup.find('h1') else ""
                paragraphs = soup.find_all('p')
                # Filter out short paragraphs (ads, nav links) to improve accuracy
                text_content = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50]
                scraped_text = title + " " + ' '.join(text_content)
                scraped_text = re.sub(r'\s+', ' ', scraped_text).strip()

                if not scraped_text:
                    st.warning("Could not extract text. The website might be using JavaScript or blocking content.")
                else:
                    st.info(f"Extracted {len(scraped_text)} characters from URL.")
                    st.session_state.input_text = scraped_text
                    with st.expander("Show Scraped Text"):
                        st.write(scraped_text)
            except Exception as e:
                st.error(f"Error fetching URL: {e}")

# --- MERGED PREDICTION & EXPLAINABILITY SECTION ---
if st.session_state.input_text:
    # 1. Transform input and Predict
    vec_text = vectorizer.transform([st.session_state.input_text])
    prediction = model.predict(vec_text)
    
    st.markdown("---")
    
    # 2. Display Result
    if prediction[0] == 1:
        st.error("🚨 FAKE NEWS Detected")
        
        # Feedback Loop: Allow user to correct the model
        with st.expander("Incorrect Prediction? Teach the Model"):
            st.write("If this is actually Real News, click below to add it to the training data and retrain.")
            if st.button("Mark as Real & Retrain"):
                new_row = pd.DataFrame({'title': [" "], 'text': [st.session_state.input_text], 'subject': ['manual'], 'date': ['2025']})
                if os.path.exists('True.csv'):
                    new_row.to_csv('True.csv', mode='a', header=False, index=False)
                    st.info("Added to dataset. Retraining model... (this may take a minute)")
                    train_model.train()
                    st.success("Model Retrained! Refresh to verify.")
                    st.rerun()
                else:
                    st.error("Could not find 'True.csv' to update.")
    else:
        st.success("✅ REAL News")

    # 3. Explainable AI Section (Now using vec_text and input_text)
    st.markdown("### 🧠 How did the AI make this decision?")
    st.write("Here are the top words in this article that influenced the model:")
    
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
        
        # Display side-by-side for a cleaner UI
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 'Fake' Indicators (Negative Score):**")
            st.dataframe(df_impact.head(5).style.background_gradient(cmap='Reds'))
        with col2:
            st.write("**Top 'Real' Indicators (Positive Score):**")
            st.dataframe(df_impact.tail(5).sort_values(by='Impact Score', ascending=False).style.background_gradient(cmap='Greens'))