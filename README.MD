Sentiment Analysis Web App
Hey there! Welcome to the Sentiment Analysis Web App. This little tool helps you analyze the sentiment of any text you input. Let's get you set up and running in no time!

Getting Started
Here's a step-by-step guide to getting everything up and running on Windows:

Step 1: Clone the Repo
First things first, clone the repository to your local machine.

terminal/ cmd/ vs code terminal

git clone https://github.com/yourusername/sentiment-analysis-webapp.git
cd sentiment-analysis-webapp
Step 2: Set Up a Virtual Environment
It's always a good idea to work in a virtual environment. Here's how you can set one up:

terminal/ cmd/ vs code terminal

# If you don't have virtualenv installed, install it first
pip install virtualenv

# Create a virtual environment
venv venv

# Change directory to venv
cd venv

# Change directory to Scripts
cd Scripts

# Activate the virtual environment
.\activate.ps1

# Go back to the project root directory
cd..
cd..

Step 3: Install Dependencies
With your virtual environment activated, install the necessary dependencies:

terminal/ cmd/ vs code terminal

pip install -r requirements.txt


Step 4: Download NLTK Stopwords
You'll need to download the NLTK stopwords data. Just run the following commands in a Python shell:

python

import nltk
nltk.download('stopwords')
Step 5: Run the App
You're all set! Now you can run the Flask app:

terminal/ cmd/ vs code terminal

python app.py
Step 6: Access the App
Open your favorite web browser and go to http://127.0.0.1:5002. You'll see the sentiment analysis app ready for you to use.

Using the App
Enter Your Text: Type (or paste) the text you want to analyze into the text area.
Submit: Hit the "Submit" button.
See the Results: The app will display how positive, neutral, and negative your text is, along with a compound score.

Example
Here’s a quick example. Enter the text:
Hey, how are you folks? Seen you after a long time. How's your day, bro? I hate you a lot, I never want to see you again, go to hell dude.

You'll get results showing the sentiment analysis of your input text.#   S e n t i A n a l y s i s  
 