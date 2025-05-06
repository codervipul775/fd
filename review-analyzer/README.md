# Product Pulse: AI-Powered Feedback Analyzer

This tool fetches and analyzes user reviews or tweets about any product or app. It summarizes key pain points, requests, and positive feedback, providing a weekly summary and actionable insights for Product Managers.

## 🚀 Features

- Upload CSV files of reviews or tweets
- AI-powered sentiment analysis (VADER or DistilBERT)
- Keyphrase extraction (KeyBERT)
- Weekly summaries and PM priorities
- Interactive dashboards and word clouds

## 🟢 Deploy on Streamlit Cloud

1. **Push your code to GitHub** (already done)
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and click 'New app'
3. **Connect your GitHub repo**: `https://github.com/codervipul775/fd.git`
4. **Set the main file path**: `review-analyzer/app.py`
5. **Set the requirements file**: `review-analyzer/requirements.txt`
6. **(Optional) Add a sample CSV in `review-analyzer/data/` for demo**
7. **Click Deploy!**

## 📝 Requirements

All dependencies are listed in `requirements.txt` and will be installed automatically by Streamlit Cloud.

## 🖥️ Local Development

```bash
cd review-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Folder Structure

```
review-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── analysis/
│   └── analyzer.py
├── utils/
│   └── summarizer.py
├── data/
│   └── reviews.csv (optional sample)
```

## 📣 Contact

For questions or improvements, open an issue or pull request on GitHub!
