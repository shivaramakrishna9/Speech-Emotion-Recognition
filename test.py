import librosa
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('ser_model.h5')

# Function to predict emotion from the audio file
def predict_emotion(model, audio_path):
    # Load audio file directly using librosa
    data, sample_rate = librosa.load(audio_path, sr=None, duration=2.5, offset=0.6)
    
    # Extract features (MFCC)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0)
    mfcc = mfcc.reshape(1, -1)  # Reshape to match the model input

    # Predict the emotion
    emotion_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    prediction = model.predict(mfcc)
    predicted_emotion = emotion_map[np.argmax(prediction)]

    return predicted_emotion

# Test the model with a sample audio file
audio_file_path = './AudioWAV/1001_IEO_FEA_HI.wav'  # Change to your file path
predicted_emotion = predict_emotion(model, audio_file_path)

# Output the result
print(f'Predicted Emotion: {predicted_emotion}')
