import nltk
import os

def download_nltk_data():
    """Download required NLTK data."""
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    required_packages = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'vader_lexicon'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package, download_dir=nltk_data_path)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")

if __name__ == "__main__":
    download_nltk_data() 