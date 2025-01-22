from flask import Flask, request, render_template, jsonify
import torch
from model import SpeechEmotionRecognitionModel
from utils import preprocess_audio, predict_emotion

app = Flask(__name__)

# Load the trained PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechEmotionRecognitionModel()  # Replace with your model class
state_dict = torch.load('model.pth', map_location=device)
model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Get the uploaded file
    audio_file = request.files['audio']
    audio_path = f"./uploads/{audio_file.filename}"
    audio_file.save(audio_path)

    # Preprocess the audio and make predictions
    features = preprocess_audio(audio_path)
    predicted_emotion = predict_emotion(features, model, device)

    return jsonify({"emotion": predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)

