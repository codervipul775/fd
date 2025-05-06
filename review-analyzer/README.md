# Review Analyzer

An AI-powered feedback analysis tool that helps Product Managers get actionable insights from customer reviews and feedback.

## Features

- Sentiment Analysis of reviews
- Keyword extraction and analysis
- Category classification
- Weekly trend analysis
- Interactive visualizations
- Word cloud generation

## How to Use

1. Upload a CSV file containing your reviews
2. Select the column containing the review text
3. (Optional) Select a date column for weekly analysis
4. View the analysis results in different tabs:
   - Overview: Key metrics and sentiment distribution
   - Analysis: Detailed sentiment and subjectivity analysis
   - Weekly Summary: Week-by-week insights (if date column provided)
   - Details: Individual review analysis

## Input Format

The CSV file should contain at least one column with review text. The column name should contain words like 'review', 'text', or 'comment'.

## Live Demo

Visit the live app at: [Your Streamlit Cloud URL will appear here after deployment]

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Dependencies

- Streamlit
- Pandas
- NLTK
- Plotly
- WordCloud
- Transformers
- PyTorch
- KeyBERT
- Sentence Transformers
