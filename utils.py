import librosa
import numpy as np
import torch

def preprocess_audio(audio_path):
    """
    Extract features (e.g., MFCCs) from the audio file.
    """
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0)  # Mean of MFCCs
    return features

def predict_emotion(features, model, device):
    """
    Use the trained model to predict emotion from audio features.
    """
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(features_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    emotion_labels = ['Unstressed', 'Stressed']  # Adjust to match your classes
    return emotion_labels[predicted_class]
