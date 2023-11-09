import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
from sklearn.ensemble import RandomForestClassifier

# Sample data - Replace this with your own dataset
data = pd.read_csv("D:\Forage\Internship evaluation\internship task\stock_sentiment.csv")

# Preprocessing: Splitting data, text vectorization, and label extraction
X = data['Text']  # Cleaned headlines from the previous example
y = data['Sentiment']     # Labels: 1 for positive, 0 for negative

vectorizer = CountVectorizer(max_features=1000)  # You can adjust the number of features
X_vec = vectorizer.fit_transform(X)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_vec, y)

# Function to preprocess input text
def remove_punctuations(Text):
    Text = Text.lower()
    Text = re.sub(f"[{re.escape(string.punctuation)}]", "", Text)
    return Text

# Streamlit web app
st.title('Stock News Prediction App')

# Input box for user to enter news headline
news_headline = st.text_area('Enter News Headline:', '')

# Preprocess the input text
cleaned_headline = remove_punctuations(news_headline)

# Perform prediction when a headline is provided
if st.button('Predict'):
    if cleaned_headline:
        news_vec = vectorizer.transform([cleaned_headline])
        predicted_movement = rf_model.predict(news_vec)
        if predicted_movement[0] == 1:
            prediction_text = 'Positive'
        else:
            prediction_text = 'Negative'
        st.write(f'Predicted Stock Movement: {prediction_text}')
    else:
        st.write('Please enter a news headline for prediction.')

primaryColor = '#FF8C02' # Bright Orange