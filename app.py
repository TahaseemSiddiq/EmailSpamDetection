from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
tfidf_vectorizer = joblib.load('tfdif_vectorizer.pkl')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the email content from the form
    email_content = request.form['email_content']

    # Transform the input email into a vector using the TF-IDF vectorizer
    tfidf_vectorized = tfidf_vectorizer.transform([email_content])

    # Make prediction
    prediction = model.predict(tfidf_vectorized)

    # Return the result
    if prediction[0] == 1:
        return render_template('index.html', prediction_text="This is a Spam email!")
    else:
        return render_template('index.html', prediction_text="This is a Non-Spam email.")

if __name__ == "__main__":
    app.run(debug=True)
