# Bhava-Chakra-Rasa Lore Generator

This Streamlit web application generates mystical lore based on an input name, analyzing its phonemes to map emotions to Bhava-Chakra and Rasa aesthetics. It uses a fine-tuned GPT-2 model (`model.safetensors` and associated files) stored in Google Drive.

## Prerequisites

- Python 3.10
- A Google Drive folder containing the fine-tuned GPT-2 model files (`model.safetensors`, `config.json`, `generation_config.json`, `merges.txt`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.json`)
- Shareable Google Drive links for the model files (file IDs required)
- Internet access for downloading model files during the first run

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Google Drive File IDs**:
   - Update the `GOOGLE_DRIVE_FILE_IDS` dictionary in `app.py` with the file IDs from your Google Drive shareable links.
   - Example: For a file with URL `https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view`, the file ID is `1AbCdEfGhIjKlMnOpQrStUvWxYz`.

4. **Pre-download NLTK Data**:
   ```bash
   python setup_nltk.py
   ```

5. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Community Cloud

- Ensure `requirements.txt` and `runtime.txt` are included in the repository.
- Push the repository to GitHub.
- Connect Streamlit Community Cloud to your GitHub repository.
- The app will automatically download the model files from Google Drive during initialization.

## Usage

- Enter an English name in the web interface.
- The app analyzes the name's phonemes, maps them to emotions, Bhava, and Rasa, and generates a unique lore using the fine-tuned GPT-2 model.
- Download the generated lore as a text file.

## Directory Structure

```
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── runtime.txt              # Runtime environment (Python version)
├── setup_nltk.py            # Script to pre-download NLTK data
├── .gitattributes           # Git attributes for file handling
├── README.markdown          # Project documentation
└── gpt2-rasa-finetuned/     # Local directory for downloaded model files
```

## Notes

- The model files are downloaded from Google Drive to `gpt2-rasa-finetuned/` on the first run or if the files are missing.
- Ensure the Google Drive links are publicly accessible or shared appropriately.
- The app is optimized for Streamlit Community Cloud but can run locally with sufficient resources.

## License

MIT License