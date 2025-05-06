from transformers import pipeline
from keybert import KeyBERT
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter
import streamlit as st

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load transformer sentiment pipeline
sentiment_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Try to load KeyBERT model with fallback
kw_model = None
try:
    kw_model = KeyBERT('all-MiniLM-L6-v2')
except Exception as e:
    st.warning("Could not load 'all-MiniLM-L6-v2' for KeyBERT. Trying fallback model 'paraphrase-MiniLM-L6-v2'.")
    try:
        kw_model = KeyBERT('paraphrase-MiniLM-L6-v2')
    except Exception as e2:
        st.error("Could not load any KeyBERT model for keyphrase extraction. Keyphrases will be disabled.")
        kw_model = None

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]  # Truncate to 512 tokens for transformers
        label = result['label']
        score = result['score']
        if label == 'POSITIVE' and score > 0.85:
            return 'Very Positive'
        elif label == 'POSITIVE':
            return 'Positive'
        elif label == 'NEGATIVE' and score > 0.85:
            return 'Very Negative'
        elif label == 'NEGATIVE':
            return 'Negative'
        else:
            return 'Neutral'
    except Exception:
        return 'Neutral'

def get_keyphrases(text, top_n=5):
    if kw_model is None:
        return []
    try:
        keyphrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
        return [phrase for phrase, score in keyphrases]
    except Exception:
        return []

def extract_keywords(text):
    # For compatibility with old code, return top 5 keyphrases with dummy counts
    phrases = get_keyphrases(text, top_n=5)
    return [(phrase, 1) for phrase in phrases]

def classify_feedback(text):
    text = text.lower()
    # Enhanced categories with more keywords and context
    categories = {
        'Pain Point': {
            'keywords': ['crash', 'slow', 'bug', 'issue', 'worse', 'error', 'problem', 'broken', 'freeze', 'lag', 'fail', 'not working'],
            'context': ['fix', 'resolve', 'solve', 'address', 'handle']
        },
        'Feature Request': {
            'keywords': ['add', 'please', 'can we', 'would love', 'should', 'could', 'would like', 'suggestion', 'recommend', 'wish', 'hope'],
            'context': ['feature', 'function', 'option', 'capability']
        },
        'Positive Highlight': {
            'keywords': ['love', 'great', 'amazing', 'thanks', 'perfect', 'excellent', 'wonderful', 'fantastic', 'awesome', 'best', 'impressed'],
            'context': ['experience', 'app', 'tool', 'software']
        },
        'Usability': {
            'keywords': ['easy', 'simple', 'intuitive', 'user-friendly', 'interface', 'design', 'layout', 'experience', 'clear', 'straightforward'],
            'context': ['use', 'navigate', 'understand', 'learn']
        },
        'Performance': {
            'keywords': ['fast', 'speed', 'quick', 'responsive', 'efficient', 'smooth', 'lag', 'slow', 'performance', 'optimize'],
            'context': ['run', 'load', 'process', 'execute']
        },
        'Support': {
            'keywords': ['help', 'support', 'customer service', 'assistance', 'response', 'contact', 'team', 'service'],
            'context': ['issue', 'problem', 'question', 'concern']
        }
    }
    
    # Count matches for each category considering both keywords and context
    category_scores = {}
    for category, patterns in categories.items():
        keyword_matches = sum(1 for keyword in patterns['keywords'] if keyword in text)
        context_matches = sum(1 for context in patterns['context'] if context in text)
        category_scores[category] = keyword_matches + (context_matches * 0.5)
    
    # Get the category with the highest score
    if not any(category_scores.values()):
        return 'Other'
    return max(category_scores, key=category_scores.get)

def analyze_review(text):
    # Get sentiment
    sentiment = get_sentiment(text)
    
    # Get category
    category = classify_feedback(text)
    
    # Extract keywords
    keywords = extract_keywords(text)
    
    # Calculate review length
    sentences = sent_tokenize(text)
    word_count = len(word_tokenize(text))
    
    return {
        'sentiment': sentiment,
        'category': category,
        'keywords': keywords,
        'sentence_count': len(sentences),
        'word_count': word_count,
        'avg_sentence_length': word_count / len(sentences) if sentences else 0
    }
