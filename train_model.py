import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

def load_data():
    
    if os.path.exists('Fake.csv') and os.path.exists('True.csv'):
        print("Found 'Fake.csv' and 'True.csv'. Merging them...")
        df_fake = pd.read_csv('Fake.csv')
        df_true = pd.read_csv('True.csv')
        
        # Add label column (1 = Fake, 0 = Real)
        df_fake['label'] = 1
        df_true['label'] = 0
        
        
        df = pd.concat([df_fake, df_true], axis=0)
        
        
        df['final_text'] = df['title'] + " " + df['text']
        return df['final_text'], df['label']

    
    elif os.path.exists('train.csv'):
        print("Found 'train.csv'. Loading...")
        try:
            df = pd.read_csv('train.csv')
            # Handle potential errors if file is bad
        except:
            print("Error: Could not read 'train.csv'. Make sure it is a valid text CSV, not a zip file.")
            return None, None
            
        # Check if it has 'text' and 'label' columns
        if 'text' in df.columns and 'label' in df.columns:
            # Drop empty rows
            df = df.dropna(subset=['text'])
            return df['text'], df['label']
        else:
            print("Error: 'train.csv' does not have 'text' and 'label' columns.")
            return None, None

    else:
        print("❌ No data found! Please put 'Fake.csv' & 'True.csv' OR 'train.csv' in this folder.")
        return None, None

def train():
    x, y = load_data()
    
    if x is None:
        return

    # Split Data
    print(f"Training on {len(x)} articles...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    # Vectorize
    # max_df=0.7 means ignore words that appear in >70% of news (too common)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2))
    tfidf_train = vectorizer.fit_transform(x_train.astype('U')) # .astype('U') ensures text is string
    tfidf_test = vectorizer.transform(x_test.astype('U'))

    # Train Model
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # Evaluate
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'✅ Model Accuracy: {round(score*100, 2)}%')

    # Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(pac, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Model saved as 'model.pkl'.")

if __name__ == "__main__":
    train()