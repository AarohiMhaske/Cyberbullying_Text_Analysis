import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Load the trained Logistic Regression model
model_filename = 'logistic_regression_model.h5'
model = joblib.load(model_filename)

cv_filename = 'count_vectorizer.pkl'
cv = joblib.load(cv_filename)

# Create a Streamlit web app
st.title("Cyberbullying Detection App")
st.write("This app uses an NLP model to detect cyberbullying in text.")

# Input box for user to enter text
user_text = st.text_area("Enter text for cyberbullying detection")

# Preprocess function similar to the one in your Colab notebook
def clean_text(text):
    text = re.sub('<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

    # Remove URLs, mentions, and hashtags from the text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    # Join the words back into a string
    text = ' '.join(words)
    return text
    
    # Implement your text cleaning and preprocessing logic here
    # You can use the same clean_text function from your Colab notebook

# "Predict" button to classify user input
if st.button("Predict"):
    if user_text:
        # Preprocess the user's input text (clean and vectorize)
        cleaned_text = clean_text(user_text)
        user_text = cv.transform([cleaned_text])

        # Predict the class of the user's input text
        prediction = model.predict(user_text)

        # Map the class label to a meaningful name
    
        predicted_class = prediction[0]

        # Display the prediction result
        st.write("Prediction:", predicted_class)

# Optionally, you can include additional visualizations and insights here

st.sidebar.markdown("About")
st.sidebar.write("This app is used for cyberbullying detection using NLP.")
st.sidebar.write("Developed by Aarohi & Anisha")

# To run the Streamlit app, execute the script in your terminal with the command:
# streamlit run your_script_name.py
