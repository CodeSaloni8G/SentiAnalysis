from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

# Download the stopwords dataset
nltk.download('stopwords')

# Initialize the Flask app
app = Flask(__name__)

# Define the route for the form
@app.route('/')
def my_form():
    return render_template('form.html', title="Optimistic")

# Define the route for processing the form submission
@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = set(stopwords.words('english'))
    
    # Get the input text and convert to lowercase
    text1 = request.form['text1'].lower()
    
    # Remove digits
    text_no_digits = ''.join(c for c in text1 if not c.isdigit())
    
    # Remove punctuation
    text_no_punct = ''.join(c for c in text_no_digits if c not in punctuation)
    
    # Remove stopwords
    processed_doc = ' '.join([word for word in text_no_punct.split() if word not in stop_words])

    # VADER Sentiment Analysis
    sa = SentimentIntensityAnalyzer()
    vader_scores = sa.polarity_scores(text=processed_doc)
    vader_compound = round(((1 + vader_scores['compound']) / 2) * 100, 2)
    
    # TextBlob Sentiment Analysis
    blob = TextBlob(processed_doc)
    blob_sentiment = blob.sentiment.polarity
    blob_compound = round(((1 + blob_sentiment) / 2) * 100, 2)
    
    # Combine VADER and TextBlob scores for a more balanced result
    combined_compound = round((vader_compound + blob_compound) / 2, 2)

    # Render the template with the sentiment scores
    return render_template('form.html', final=combined_compound, text1=text1, 
                           text2=vader_scores['pos'], text5=vader_scores['neg'], 
                           text4=combined_compound, text3=vader_scores['neu'], title="Optimistic")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
