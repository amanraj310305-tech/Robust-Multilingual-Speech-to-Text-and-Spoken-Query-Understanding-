import whisper
import sounddevice as sd
import numpy as np
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from jiwer import wer


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


SAMPLE_RATE = 16000
DURATION = 6
NUM_SAMPLES = 3
NOISE_LEVEL = 0.005

INTENT_DATA = {
    "education": [
        "exam schedule",
        "college admission process",
        "syllabus information"
    ],
    "agriculture": [
        "crop disease treatment",
        "fertilizer advice",
        "irrigation methods"
    ],
    "general": [
        "tell me something",
        "what is happening",
        "give information"
    ]
}


print("Loading Whisper model...")
model = whisper.load_model("base", device="cpu")
print("Whisper loaded.")

stop = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop]
    return " ".join(tokens)

def add_noise(audio, level=0.005):
    noise = np.random.randn(len(audio)).astype(np.float32)
    return audio + level * noise


results = []

styles = ["normal", "slow", "fast"]

print("\nYou will record", NUM_SAMPLES, "audio samples.")

for i in range(NUM_SAMPLES):
    style = styles[i % len(styles)]
    input(f"\nPress ENTER and speak sample {i+1} ({style})...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )
    sd.wait()

    audio = audio.flatten().astype(np.float32)

    true_text = input("Type what you spoke: ")

    print("Transcribing clean audio...")
    clean_result = model.transcribe(audio, fp16=False)
    clean_text_out = clean_result["text"]
    detected_lang = clean_result["language"]

    noisy_audio = add_noise(audio, NOISE_LEVEL).astype(np.float32)

    print("Transcribing noisy audio...")
    noisy_result = model.transcribe(noisy_audio, fp16=False)
    noisy_text_out = noisy_result["text"]

    results.append({
        "sample": i + 1,
        "style": style,
        "language": detected_lang,
        "ground_truth": true_text,
        "clean_text": clean_text_out,
        "noisy_text": noisy_text_out
    })

df_audio = pd.DataFrame(results)

print("\nSpeech-to-text completed.")
print(df_audio)


texts = []
labels = []

for intent, examples in INTENT_DATA.items():
    for ex in examples:
        texts.append(clean_text(ex))
        labels.append(intent)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)


def predict_intent(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]

df_audio["intent_clean"] = df_audio["clean_text"].apply(predict_intent)
df_audio["intent_noisy"] = df_audio["noisy_text"].apply(predict_intent)


df_audio["wer_clean"] = df_audio.apply(lambda x: wer(x["ground_truth"], x["clean_text"]), axis=1)
df_audio["wer_noisy"] = df_audio.apply(lambda x: wer(x["ground_truth"], x["noisy_text"]), axis=1)

intent_agreement = (df_audio["intent_clean"] == df_audio["intent_noisy"]).mean()

print("\nAverage Clean WER:", round(df_audio["wer_clean"].mean(), 3))
print("Average Noisy WER:", round(df_audio["wer_noisy"].mean(), 3))
print("Intent agreement:", round(intent_agreement, 3))



reliability = 1 - df_audio["wer_noisy"].mean()
print("System reliability:", round(reliability, 3))



df_audio.to_csv("speech_results.csv", index=False)

print("\nSaved: speech_results.csv")
print("\nPIPELINE COMPLETE.")
