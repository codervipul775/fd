import os
os.environ["NLTK_DATA"] = os.path.join(os.path.expanduser("~"), "nltk_data")
import streamlit as st
import pandas as pd
from analysis.analyzer import classify_feedback, extract_keywords
from utils.summarizer import generate_summary
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Initialize NLTK data
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
    except Exception as e:
        st.error(f"Error downloading NLTK package {package}: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { padding: 1rem 2rem; font-size: 1.1rem; }
    .stMetric { font-size: 1.2rem; }
    .pm-card { background: #fffbe6; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px #f0f0f0; }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Product Pulse: AI-Powered Feedback Analyzer")
st.markdown("""
    <span style='font-size:1.2rem;'>Upload your reviews or tweets and get actionable, weekly insights for Product Managers.</span>
""", unsafe_allow_html=True)

# --- CSV Upload Feature ---
st.sidebar.header("📥 Upload Your Reviews CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(10), use_container_width=True)

    # Let user select the review column
    text_col_candidates = [col for col in df.columns if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower()]
    if not text_col_candidates:
        text_col_candidates = [col for col in df.columns if df[col].dtype == object]
    review_col = st.selectbox("Select the column containing the review text:", options=text_col_candidates, help="Pick the column with the main review or tweet text.")

    # (Optional) Let user select a date column for weekly summaries
    date_col_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    date_col = st.selectbox("Select the date column (optional, for weekly summary):", options=[None] + date_col_candidates, help="Pick a date column for weekly grouping.")

    # Prepare working DataFrame
    work_df = df.copy()
    work_df['review'] = work_df[review_col].astype(str)
    if date_col:
        work_df['date'] = pd.to_datetime(work_df[date_col])

    # --- Speed Optimization: Limit Rows ---
    max_rows = st.sidebar.number_input("Max rows to analyze", min_value=500, max_value=len(work_df), value=min(5000, len(work_df)))
    work_df = work_df.head(max_rows)

    # Step 2: Analysis
    st.subheader("Step 2: Analysis")
    with st.spinner("Analyzing reviews..."):
        sia = SentimentIntensityAnalyzer()
        def fast_vader_sentiment(text):
            scores = sia.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.5:
                return 'Very Positive'
            elif compound >= 0.05:
                return 'Positive'
            elif compound <= -0.5:
                return 'Very Negative'
            elif compound <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        work_df['sentiment'] = [fast_vader_sentiment(text) for text in work_df['review']]
        work_df['category'] = [classify_feedback(text) for text in work_df['review']]
        work_df['keywords'] = [extract_keywords(text) for text in work_df['review']]
        work_df['word_count'] = work_df['review'].str.split().str.len()
        work_df['sentence_count'] = work_df['review'].apply(lambda x: len(nltk.sent_tokenize(x)))
        work_df['avg_sentence_length'] = work_df['word_count'] / work_df['sentence_count']
        work_df['avg_sentence_length'] = work_df['avg_sentence_length'].replace([float('inf'), float('nan')], 0)
        def flatten_column(col):
            return col.apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x)
        work_df['sentiment'] = flatten_column(work_df['sentiment']).astype(str)
        work_df['category'] = flatten_column(work_df['category']).astype(str)
        summary = generate_summary(work_df)

    # Main content
    if date_col:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analysis", "📅 Weekly Summary", "🔍 Details"])
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analysis", "🔍 Details"])

    with tab1:
        st.markdown("## 📋 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", summary['insights']['total_reviews'])
        with col2:
            st.metric("Most Common Category", summary['insights']['most_common_category'])
        with col3:
            st.metric("Overall Sentiment", summary['insights']['overall_sentiment'])
        with col4:
            st.metric("Avg Sentiment Score", f"{summary['insights']['avg_sentiment_score']:.2f}")
        st.markdown("---")
        st.markdown("### 📊 Sentiment Distribution")
        st.plotly_chart(summary['sentiment_plot'].update_layout(
            template='plotly_dark',
            title_font_size=20,
            xaxis_title="Category",
            yaxis_title="Count"
        ), use_container_width=True)
        st.markdown("---")
        st.markdown("### ☁️ Top Keywords Word Cloud")
        all_phrases = ' '.join([' '.join([kw for kw, _ in kws]) for kws in work_df['keywords']])
        if all_phrases.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_phrases)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No keywords to display in word cloud.")

    with tab2:
        st.markdown("### 📈 Detailed Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(summary['sentiment_score_plot'].update_layout(template='plotly_white'), use_container_width=True)
        with col2:
            st.plotly_chart(summary['subjectivity_plot'].update_layout(template='plotly_white'), use_container_width=True)
        st.markdown("---")
        st.plotly_chart(summary['word_freq_plot'].update_layout(template='plotly_white'), use_container_width=True)

    if date_col:
        with tab3:
            st.markdown("### 📅 Weekly Summary & PM Priorities")
            work_df['week'] = work_df['date'].dt.to_period('W').apply(lambda r: r.start_time)
            week_groups = work_df.groupby('week')
            prev_pain_points = set()
            prev_feature_requests = set()
            for week, group in week_groups:
                st.markdown(f"#### Week of {week.strftime('%Y-%m-%d')}")
                st.markdown("---")
                pain_points = group[group['category'] == 'Pain Point']['review'].tolist()
                feature_requests = group[group['category'] == 'Feature Request']['review'].tolist()
                positives = group[group['category'] == 'Positive Highlight']['review'].tolist()
                pain_keywords = Counter()
                for r in pain_points:
                    pain_keywords.update([w for w, _ in classify_feedback(r)['keywords']])
                feature_keywords = Counter()
                for r in feature_requests:
                    feature_keywords.update([w for w, _ in classify_feedback(r)['keywords']])
                top_pain = pain_keywords.most_common(2)
                top_feature = feature_keywords.most_common(2)
                st.markdown(f"<div class='pm-card'><b>Pain Points ({len(pain_points)}):</b>", unsafe_allow_html=True)
                for r in pain_points[:3]:
                    st.write(f"- {r[:120]}{'...' if len(r) > 120 else ''}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='pm-card'><b>Feature Requests ({len(feature_requests)}):</b>", unsafe_allow_html=True)
                for r in feature_requests[:3]:
                    st.write(f"- {r[:120]}{'...' if len(r) > 120 else ''}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='pm-card'><b>Positive Highlights ({len(positives)}):</b>", unsafe_allow_html=True)
                for r in positives[:3]:
                    st.write(f"- {r[:120]}{'...' if len(r) > 120 else ''}")
                st.markdown("</div>", unsafe_allow_html=True)
                sentiment_counts = group['sentiment'].value_counts()
                st.write("**Sentiment Distribution:**")
                st.plotly_chart(px.bar(sentiment_counts, labels={'index': 'Sentiment', 'value': 'Count'}, title='Sentiment Distribution').update_layout(template='plotly_dark'), use_container_width=True)
                st.markdown("**PM Priorities:**")
                pm_priorities = []
                new_pain = set([k for k, _ in top_pain]) - prev_pain_points
                for k, v in top_pain:
                    trend = " (new!)" if k in new_pain else ""
                    pm_priorities.append(f"Fix '{k}' issues ({v} mentions){trend}")
                prev_pain_points = set([k for k, _ in top_pain])
                new_feature = set([k for k, _ in top_feature]) - prev_feature_requests
                for k, v in top_feature:
                    trend = " (new!)" if k in new_feature else ""
                    pm_priorities.append(f"Prioritize '{k}' feature ({v} requests){trend}")
                prev_feature_requests = set([k for k, _ in top_feature])
                if pm_priorities:
                    for p in pm_priorities:
                        st.write(f"- {p}")
                else:
                    st.write("- No urgent priorities detected this week.")
                st.markdown("---")

    with tab4 if date_col else tab3:
        st.markdown("### 🔍 Review Details")
        selected_category = st.selectbox(
            "Filter by Category",
            options=['All'] + list(work_df['category'].unique())
        )
        selected_sentiment = st.selectbox(
            "Filter by Sentiment",
            options=['All'] + list(work_df['sentiment'].unique())
        )
        filtered_df = work_df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if selected_sentiment != 'All':
            filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
        for _, row in filtered_df.iterrows():
            with st.expander(f"Review: {row['review'][:50]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sentiment:**", row['sentiment'])
                    st.write("**Category:**", row['category'])
                    st.write("**Word Count:**", row['word_count'])
                with col2:
                    st.write("**Top Keywords:**")
                    for word, count in row['keywords']:
                        st.write(f"- {word} ({count})")

    # Additional insights
    st.sidebar.subheader("💡 Additional Insights")
    st.sidebar.write("**Most Subjective Category:**", summary['insights']['most_subjective_category'])
    st.sidebar.write("**Least Subjective Category:**", summary['insights']['least_subjective_category'])
    st.sidebar.write("**Average Subjectivity Score:**", f"{summary['insights']['avg_subjectivity_score']:.2f}")
    st.sidebar.download_button(
        label="📥 Download Analysis Results",
        data=work_df.to_csv(index=False).encode('utf-8'),
        file_name='review_analysis.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to get started.")
