import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import gdown

# Google Drive file IDs for model files
# These IDs correspond to files in the Google Drive folder: https://drive.google.com/drive/folders/1nXB7hPbSkn3zwCUT-zegXV0mAR80Wlv-
GOOGLE_DRIVE_FILE_IDS = {
    "model.safetensors": "1Fy6cWT7aXPfrraJWP9jfm5T8GtzbeIv-",
    "config.json": "1-ibDlfGL427TfPlXuomrt8VVsvwht90o",
    "generation_config.json": "1YDGJBDVUkv26EdqgEUhAtIfmVO44oo7q",
    "merges.txt": "1JCvmORZ0viVz4O-0pUNqmBw8YTZc0xDz",
    "special_tokens_map.json": "1fc91SRN9_afQYUH7vZiStgWiJVcCCGa3",
    "tokenizer_config.json": "1q8WIgHP0WEw3F0g6RR-osxEAkwPnirLd",
    "vocab.json": "1986ye9dofWUxLaxeoFVNnTF9-RkPtfKB"
}

# Model directory
MODEL_DIR = "./gpt2-rasa-finetuned"

# Pre-download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('cmudict', quiet=True)

# Initialize CMU dictionary
cmu_dict = cmudict.dict()

# Bhava-Chakra and Rasa mappings
BHAVA_RASA_MAP = {
    'happy': {'bhava': 'Rati (Love)', 'rasa': 'Shringara (Romantic)'},
    'sad': {'bhava': 'Shoka (Sorrow)', 'rasa': 'Karuna (Compassion)'},
    'angry': {'bhava': 'Krodha (Anger)', 'rasa': 'Raudra (Fury)'},
    'calm': {'bhava': 'Shanta (Peace)', 'rasa': 'Shanta (Tranquility)'},
    'fearful': {'bhava': 'Bhaya (Fear)', 'rasa': 'Bhayanaka (Horror)'},
    'excited': {'bhava': 'Utsaha (Enthusiasm)', 'rasa': 'Veera (Heroic)'},
}

# Phoneme to emotion mapping
PHONEME_EMOTION_MAP = {
    'AA': 'happy', 'AE': 'excited', 'AH': 'calm', 'AO': 'sad',
    'AW': 'happy', 'AY': 'excited', 'B': 'angry', 'CH': 'excited',
    'D': 'sad', 'DH': 'calm', 'EH': 'fearful', 'ER': 'sad',
    'EY': 'happy', 'F': 'fearful', 'G': 'angry', 'HH': 'calm',
    'IH': 'excited', 'IY': 'happy', 'JH': 'excited', 'K': 'angry',
    'L': 'calm', 'M': 'happy', 'N': 'calm', 'NG': 'sad',
    'OW': 'happy', 'OY': 'excited', 'P': 'angry', 'R': 'excited',
    'S': 'fearful', 'SH': 'calm', 'T': 'angry', 'TH': 'fearful',
    'UH': 'sad', 'UW': 'happy', 'V': 'happy', 'W': 'calm',
    'Y': 'excited', 'Z': 'fearful', 'ZH': 'calm'
}

# Download model files from Google Drive
def download_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for file_name, file_id in GOOGLE_DRIVE_FILE_IDS.items():
        file_path = os.path.join(MODEL_DIR, file_name)
        if not os.path.exists(file_path):
            try:
                st.write(f"Downloading {file_name} from Google Drive...")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
            except Exception as e:
                st.error(f"Failed to download {file_name}: {str(e)}")
                return False
    return True

# Load fine-tuned GPT-2 model and tokenizer
@st.cache_resource
def load_gpt2_model():
    if not download_model_files():
        return None, None
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, use_safetensors=True)
        if torch.cuda.is_available():
            model = model.cuda()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}. Ensure all model files are correctly downloaded.")
        return None, None

# Generate lore using fine-tuned GPT-2
def generate_gpt2_lore(name, emotion, bhava, rasa):
    try:
        tokenizer, model = load_gpt2_model()
        if tokenizer is None or model is None:
            return "Model loading failed. Please check the model files in Google Drive."
        prompt = f"In a mythical realm, {name} embodies {bhava}, radiating {rasa}. Their name evokes {emotion}. Craft a vivid lore about {name}, blending mysticism and cosmic wonder."
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        lore = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        if not lore.endswith("."):
            lore += "."
        return lore
    except Exception as e:
        return f"Error generating lore: {str(e)}"

def get_phonemes(word):
    word = word.lower()
    return cmu_dict.get(word, [[]])[0]

def analyze_name_phonemes(name):
    phonemes = []
    for word in word_tokenize(name):
        phonemes.extend(get_phonemes(word))
    emotions = [PHONEME_EMOTION_MAP.get(phoneme.split('-')[0], 'calm') for phoneme in phonemes]
    return max(set(emotions), key=emotions.count) if emotions else 'calm'

# Streamlit App
st.title("Bhava-Chakra-Rasa Lore Generator (A100 Fine-Tuned)")
st.markdown("""
Enter an English name such as Mahan H R Gowda to uncover its phoneme-based emotional essence, mapped to Bhava-Chakra and Rasa, and receive a lore crafted by a fine-tuned GPT-2 model loaded from Google Drive.
""")

# Input form
name_input = st.text_input("Enter a name:", placeholder="e.g., Aria, John, Seraphina")

if name_input:
    if not name_input.strip().isalpha():
        st.error("Please enter a valid name containing only letters.")
    else:
        try:
            dominant_emotion = analyze_name_phonemes(name_input)
            bhava_rasa = BHAVA_RASA_MAP.get(dominant_emotion, BHAVA_RASA_MAP['calm'])
            rasa = bhava_rasa['rasa']
            bhava = bhava_rasa['bhava']
            lore = generate_gpt2_lore(name_input, dominant_emotion, bhava, rasa)
            
            st.subheader(f"Lore of {name_input}")
            st.markdown(f"**Dominant Emotion**: {dominant_emotion.capitalize()}")
            st.markdown(f"**Bhava (Emotion)**: {bhava}")
            st.markdown(f"**Rasa (Aesthetic)**: {rasa}")
            st.markdown("---")
            st.markdown(f"**{name_input}'s Lore**:")
            st.write(lore)
            
            st.download_button(
                label="Download Lore",
                data=lore,
                file_name=f"{name_input}_lore.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}. Please check your setup.")

st.markdown("""
---
*Powered by Streamlit, NLTK, and fine-tuned GPT-2. Optimized for Streamlit Community Cloud.*
""")