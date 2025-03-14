import os
import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, render_template, request, jsonify
import io

# Load the trained model
model = tf.keras.models.load_model('ser_model.h5')

# Define the emotions for classification
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

# Initialize the Flask app
app = Flask(__name__)

def preprocess_audio(audio_data):
    try:
        # Load the audio file with librosa and convert to 16kHz
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=16000, duration=3.0)  # 3 seconds of audio

        # Ensure that the audio is loaded correctly
        if audio is None or len(audio) == 0:
            raise ValueError("Audio data is empty or not loaded properly")

        # Extract MFCC features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Check if MFCC extraction was successful
        if mfcc.shape[1] == 0:
            raise ValueError("MFCC extraction returned an empty array")

        # Average the MFCC features over the time axis
        mfcc = np.mean(mfcc.T, axis=0)

        # Pad the sequence to match the model's expected input shape (e.g., 128 timesteps)
        max_sequence_length = 128
        if mfcc.shape[0] < max_sequence_length:
            # Pad the sequence if it's too short
            padding = np.zeros((max_sequence_length - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack([mfcc, padding])
        elif mfcc.shape[0] > max_sequence_length:
            # Trim the sequence if it's too long
            mfcc = mfcc[:max_sequence_length]

        return mfcc

    except Exception as e:
        print(f"Error in audio preprocessing: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded audio file from the request
        audio_file = request.files['audio']

        # Read the audio file into a byte stream
        audio_data = audio_file.read()

        # Preprocess the audio data and extract features
        mfcc = preprocess_audio(audio_data)

        # Reshape the MFCC array to the shape the model expects
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(mfcc)

        # Get the predicted emotion with the highest probability
        predicted_emotion = emotions[np.argmax(prediction)]

        return jsonify({'emotion': predicted_emotion})

    except Exception as e:
        print(f"Error processing the audio file: {str(e)}")
        return jsonify({'error': 'Error processing the audio file. Please try again.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
