import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from collections import Counter
import numpy as np

def generate_summary(df):
    # Basic sentiment and category counts
    sentiment_summary = df.groupby(['sentiment', 'category']).size().unstack(fill_value=0)
    
    # Calculate sentiment scores
    df['sentiment_score'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity_score'] = df['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Calculate average scores
    avg_sentiment = df.groupby('category')['sentiment_score'].mean()
    avg_subjectivity = df.groupby('category')['subjectivity_score'].mean()
    
    # Create visualizations
    # Sentiment distribution plot
    sentiment_plot = px.bar(sentiment_summary, 
                          title='Sentiment Distribution by Category',
                          labels={'value': 'Count', 'sentiment': 'Sentiment', 'category': 'Category'},
                          color_discrete_sequence=px.colors.qualitative.Set3)
    
    # Sentiment score distribution
    sentiment_score_plot = px.box(df, 
                                x='category', 
                                y='sentiment_score',
                                title='Sentiment Score Distribution by Category',
                                color='category',
                                color_discrete_sequence=px.colors.qualitative.Set3)
    
    # Subjectivity distribution
    subjectivity_plot = px.box(df,
                             x='category',
                             y='subjectivity_score',
                             title='Subjectivity Distribution by Category',
                             color='category',
                             color_discrete_sequence=px.colors.qualitative.Set3)
    
    # Word cloud data
    all_reviews = ' '.join(df['review'].tolist())
    words = all_reviews.lower().split()
    word_freq = Counter(words)
    top_words = dict(word_freq.most_common(20))
    
    # Word frequency plot
    word_freq_plot = px.bar(x=list(top_words.keys()),
                           y=list(top_words.values()),
                           title='Top 20 Most Common Words',
                           labels={'x': 'Word', 'y': 'Frequency'},
                           color_discrete_sequence=['#636EFA'])
    
    # Generate insights
    insights = {
        'total_reviews': len(df),
        'most_common_category': df['category'].mode()[0],
        'overall_sentiment': df['sentiment'].mode()[0],
        'avg_sentiment_score': df['sentiment_score'].mean(),
        'avg_subjectivity_score': df['subjectivity_score'].mean(),
        'top_positive_category': avg_sentiment.idxmax(),
        'top_negative_category': avg_sentiment.idxmin(),
        'most_subjective_category': avg_subjectivity.idxmax(),
        'least_subjective_category': avg_subjectivity.idxmin(),
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'category_distribution': df['category'].value_counts().to_dict(),
        'top_words': top_words
    }
    
    return {
        'summary_table': sentiment_summary,
        'sentiment_plot': sentiment_plot,
        'sentiment_score_plot': sentiment_score_plot,
        'subjectivity_plot': subjectivity_plot,
        'word_freq_plot': word_freq_plot,
        'insights': insights
    }
