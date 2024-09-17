import subprocess
import ffmpeg
from flask import Flask, request, redirect, url_for, render_template
import os
import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = Flask(__name__)

# Define the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTION_FOLDER = 'transcriptions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# Load Whisper model and processor
model_name = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
print(f"Model loaded: {model}")

# Configure the upload folder in the Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    # Debugging step 1: Check if POST request is received
    print("Received POST request")

    if 'audio_file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)

    audio_file = request.files['audio_file']

    # Debugging step 2: Check if a file was selected
    if audio_file.filename == '':
        print("No file selected")
        return redirect(request.url)

    if audio_file:
        # Save the file to the uploads folder
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(audio_path)

        # Debugging step 3: Confirm file has been saved
        print(f"File {audio_file.filename} uploaded successfully at {audio_path}")

        # Transcribe the audio
        transcribed_text = preprocess_and_transcribe(audio_path)

        # Save the transcription to a text file
        transcription_filename = f"{audio_file.filename.rsplit('.', 1)[0]}.txt"
        transcription_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)
        with open(transcription_path, 'w') as f:
            f.write(transcribed_text)

        print(f"Transcription saved to {transcription_path}")
        # Display the transcription on the webpage
        return render_template('transcription_result.html', transcription=transcribed_text, filename=transcription_filename)

    return redirect(request.url)

@app.route('/upload-mic', methods=['POST'])
def upload_mic_file():
    # Debugging step: Check if POST request is received
    print("Received POST request from microphone")

    if 'mic_audio_file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)

    mic_audio_file = request.files['mic_audio_file']

    if mic_audio_file.filename == '':
        print("No file selected")
        return redirect(request.url)

    if mic_audio_file:
        # Save the microphone recording to the uploads folder
        mic_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], mic_audio_file.filename)
        mic_audio_file.save(mic_audio_path)

        # Transcribe the audio
        transcribed_text = preprocess_and_transcribe(mic_audio_path)

        # Save the transcription to a text file
        transcription_filename = f"{mic_audio_file.filename.rsplit('.', 1)[0]}.txt"
        transcription_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)
        with open(transcription_path, 'w') as f:
            f.write(transcribed_text)

        print(f"Transcription saved to {transcription_path}")

        return render_template('transcription_result.html', transcription=transcribed_text, filename=transcription_filename)

    return redirect(request.url)


# Function to preprocess audio using FFmpeg (resample, convert to mono)
def preprocess_audio(input_path, output_path):
    print(f"Preprocessing audio from {input_path} to {output_path}")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not os.path.isdir(os.path.dirname(output_path)):
        raise FileNotFoundError(f"Output directory not found: {os.path.dirname(output_path)}")
    ffmpeg.input(input_path).output(output_path, ar=16000, ac=1).run(overwrite_output=True,
        cmd='C:/Users/DELL/Downloads/ffmpeg-7.0.2-essentials_build/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe')
    return output_path

# Function to transcribe audio using Whisper
def preprocess_and_transcribe(input_path):
    # Preprocess audio (resample to 16kHz mono)
    preprocessed_path = preprocess_audio(input_path, input_path.replace(".wav", "_preprocessed.wav"))

    # Load audio file
    audio_tensor, sample_rate = torchaudio.load(preprocessed_path)

    # Debugging: Print audio tensor details
    print(f"Audio tensor shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")

    # Resample if needed and make sure it's mono
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = resampler(audio_tensor)

    # Squeeze the audio tensor to 1D array
    audio_input = audio_tensor.squeeze().numpy()

    # Preprocess audio for Whisper model
    input_features = processor(audio=audio_input, return_tensors="pt", sampling_rate=16000)

    # Debugging: Print input features details
    print(f"Input features: {input_features}")

    # Generate transcription
    with torch.no_grad():
        transcription_ids = model.generate(input_features.input_features)
        print(f"Generated IDs: {transcription_ids}")

    # Decode transcription
    transcription_text = processor.batch_decode(transcription_ids, skip_special_tokens=True)[0]

    return transcription_text


if __name__ == "__main__":
    app.run(debug=True)
