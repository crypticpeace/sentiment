import pandas as pd
import tkinter as tk
from tkinter import filedialog
import zipfile
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tabulate import tabulate  


DetectorFactory.seed = 0  

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text."""
    if pd.isna(text) or str(text).strip() == "":
        return "Neutral", 0.0  

    text = str(text).strip().lower()  

    
    if text in ["no", "nah", "not really", "never"]:
        return "Negative", -0.3
    if text in ["yes", "yeah", "sure", "definitely"]:
        return "Positive", 0.3
    
    try:
        lang = detect(text)
        if lang != 'en':
            translator = GoogleTranslator(source=lang, target='en')
            text = translator.translate(text)
    except LangDetectException:
        pass  

    analyzer = SentimentIntensityAnalyzer()
    vader_score = analyzer.polarity_scores(text)['compound']
    textblob_score = TextBlob(text).sentiment.polarity
    sentiment_score = (vader_score + textblob_score) / 2  

    if sentiment_score >= 0.7:
        return "Very Positive", sentiment_score
    elif 0.2 <= sentiment_score < 0.7:
        return "Positive", sentiment_score
    elif -0.2 < sentiment_score < 0.2:
        return "Neutral", sentiment_score
    elif -0.7 < sentiment_score <= -0.2:
        return "Negative", sentiment_score
    else:
        return "Very Negative", sentiment_score

def load_file(file_path):
    """Loads data from Excel, CSV, or ZIP files."""
    try:
        if file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_files = [f for f in zip_ref.namelist() if f.endswith(".xlsx") or f.endswith(".csv")]
                if not zip_files:
                    print("\n❌ No valid Excel or CSV file found in the ZIP archive.")
                    return None
                
                extracted_file = zip_files[0]  
                zip_ref.extract(extracted_file)
                file_path = extracted_file  

                if file_path.endswith(".xlsx"):
                    return pd.read_excel(file_path)
                else:
                    return pd.read_csv(file_path)
        else:
            print("\n❌ Unsupported file format. Please select an Excel, CSV, or ZIP file.")
            return None
    except Exception as e:
        print(f"\n❌ Error loading file: {str(e)}")
        return None

def process_feedback(file_path):
    """Processes feedback from a file and displays sentiment analysis results question-wise."""
    df = load_file(file_path)
    if df is None or df.empty:
        print("\n❌ Error: The file is empty or invalid.")
        return

    df_filtered = df.drop(columns=['Timestamp', 'Name', 'Roll No', 'Class'], errors='ignore')

    
    for column in df_filtered.columns:
        print(f"\n🔹 **Question: {column}** 🔹\n")

        results = []
        for text in df_filtered[column].dropna().astype(str): 
            sentiment, score = analyze_sentiment(text)
            results.append([text, sentiment, round(score, 2)])

        
        if results:
            print(tabulate(results, headers=["Feedback", "Sentiment", "Score"], tablefmt="fancy_grid", showindex=False))
        else:
            print("No feedback provided for this question.")

def select_file():
    """Opens file dialog to select an Excel, CSV, or ZIP file."""
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("Supported Files", "*.xlsx;*.csv;*.zip"), ("Excel Files", "*.xlsx"), ("CSV Files", "*.csv"), ("ZIP Files", "*.zip")]
    )
    
    if file_path:
        process_feedback(file_path)

def main():
    """Creates Tkinter GUI for file selection."""
    root = tk.Tk()
    root.withdraw()  
    
    print("\n📂 Please select an Excel, CSV, or ZIP file for sentiment analysis...\n")
    select_file()

if __name__ == "__main__":
    main()
