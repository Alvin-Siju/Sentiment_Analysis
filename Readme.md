Sentiment Analysis

This project is designed to perform sentiment analysis on text data using a Naive Bayes classifier. The model is trained on the ACL-IMDB dataset, which contains movie reviews labeled as either positive or negative. The trained model can then predict the sentiment of new text data, classifying it as either positive or negative.

Project Structure

Files in the Repository:

acllmdb_v1.tar.gz: 
                 A compressed file containing the ACL-IMDB dataset. The dataset consists of movie reviews labeled with their corresponding sentiment (positive/negative).

app.ipynb: 
                 A Jupyter notebook used for data exploration, preprocessing, model training, and evaluation. It contains step-by-step processes to build and evaluate the sentiment analysis model.

app.py: 
                The main Python application script that loads the trained Naive Bayes model and vectorizer, and allows users to input new text for sentiment prediction. This script can be used to run the sentiment analysis on any text data.

naive_bayes_model.pkl:  
                A serialized (pickled) file containing the trained Naive Bayes model. This model is used to classify new text data as either positive or negative sentiment.

Procfile: 
                A file used for deployment purposes (if you choose to deploy the app to a platform like Heroku). It specifies the commands to run the application on the server.

requirements.txt: 
                A list of Python dependencies required to run the project. It includes libraries such as scikit-learn, pandas, numpy, and others necessary for model training and running the application.

vectorizer.pkl: 
                A serialized file containing the trained TF-IDF vectorizer. This vectorizer is used to convert text data into numerical form that the machine learning model can process.

Installation

To run this project locally, follow these steps:

1. Clone the Repository
Start by cloning the repository to your local machine:

bash

git clone https://github.com/Alvin-Siju/Sentiment_Analysis.git
cd Sentiment_Analysis
2. Set Up the Virtual Environment
It's recommended to set up a virtual environment to manage dependencies:

bash

# Create a virtual environment (replace "env" with your preferred name)
python -m venv env

# Activate the virtual environment
# For Windows:
env\Scripts\activate
# For macOS/Linux:
source env/bin/activate
3. Install Dependencies
Next, install the required Python packages from the requirements.txt file:

bash

pip install -r requirements.txt
This will install all the necessary libraries to run the project, including scikit-learn, pandas, numpy, and others.

Usage
1. Data Preprocessing and Model Training
If you'd like to understand the process of preparing the data and training the model, you can refer to the app.ipynb Jupyter notebook. In the notebook, you will find:

Data Exploration: Overview of the dataset, including a look at the number of positive and negative reviews.
Preprocessing: Steps for cleaning the text data, including tokenization and vectorization.
Model Training: Training the Naive Bayes model using the preprocessed text data.
Model Evaluation: Testing the model's performance using accuracy, precision, recall, and F1-score.
2. Using the Trained Model for Sentiment Prediction
To use the trained model for sentiment prediction, you can run the app.py script. This script loads the trained model and vectorizer and allows you to input text for sentiment prediction.

Example usage:

python

import pickle

# Load the trained model and vectorizer
model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Example text
text = "I love this movie! It was amazing."

# Transform the text using the vectorizer and predict sentiment
text_vectorized = vectorizer.transform([text])
sentiment = model.predict(text_vectorized)

# Print the predicted sentiment
if sentiment == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
This code loads the trained model and vectorizer, then uses them to predict the sentiment of the input text. You can replace the text in the variable text with any sentence or movie review to see the result.

3. Exploring the Dataset
If you want to explore the dataset itself, you can extract the acllmdb_v1.tar.gz file. Inside, you'll find the data organized into positive and negative reviews. This dataset can be used to retrain the model or for further analysis.

To extract the dataset:

bash

tar -xvzf acllmdb_v1.tar.gz
4. Running the Application
The app.py script can be modified to create a simple command-line or graphical interface to allow users to input text for sentiment prediction.

Model Details
The project uses a Naive Bayes classifier to perform sentiment analysis. The model is trained using the TF-IDF (Term Frequency-Inverse Document Frequency) method for text vectorization. TF-IDF helps convert raw text data into a form that machine learning algorithms can understand by assigning a weight to each term in the document based on its importance in the overall dataset.