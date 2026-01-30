Robust Multilingual Speech Intent Classification 

This project implements a multilingual speech-to-text and spoken query understanding pipeline using OpenAI Whisper and NLP techniques. The system records user speech, transcribes it into text, classifies intent, and evaluates robustness under noisy conditions by measuring Word Error Rate (WER) and intent agreement.

The project also performs experimental reliability analysis by comparing clean and noisy transcriptions to study degradation patterns in real-world speech scenarios.

🚀 Features

Real-time audio recording

Multilingual speech recognition with automatic language detection

Text preprocessing using NLTK

Intent classification using TF-IDF + Logistic Regression

Noise robustness evaluation

Word Error Rate (WER) based accuracy analysis

Reliability metric computation

CSV-based experimental result logging

🛠 Tech Stack

Python

OpenAI Whisper

PyTorch

SoundDevice

NLTK

Scikit-learn

Pandas

NumPy

Jiwer

📂 Project Structure

speech-intent-whisper/

main.py
README.md
requirements.txt
.gitignore
speech_results.csv

⚙️ Setup

Install dependencies:

pip install -r requirements.txt

Ensure FFmpeg is installed and added to system PATH.

Download NLTK resources (first run only):

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

▶️ Run

python main.py

Follow terminal instructions and speak when prompted.

📊 Output

The program generates speech_results.csv containing:

sample

style

detected language

ground truth text

clean transcription

noisy transcription

predicted intents

WER (clean + noisy)

reliability metrics

🧠 Method Overview

Record speech samples under different speaking styles

Transcribe audio using Whisper

Add artificial noise to simulate real-world conditions

Compute Word Error Rate against ground truth

Train intent classifier using TF-IDF + Logistic Regression

Compare clean vs noisy intent predictions

Calculate system reliability

Save full experiment results to CSV

👤 Author

Raj
BTech CSE Student

