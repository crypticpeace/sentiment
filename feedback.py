import pandas as pd
import zipfile
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tabulate import tabulate

# Set seed for reproducibility
DetectorFactory.seed = 0

# Assume the file is in the same directory as the script
# This will work whether you're running from a local copy of the GitHub repository
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text."""
    if pd.isna(text) or str(text).strip() == "":
        return "Neutral", 0.0

    text = str(text).strip().lower()
    
    # Quick checks for common responses
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
        print(f"\n📂 Loading file from: {file_path}")
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
            print("\n❌ Unsupported file format. Please provide an Excel, CSV, or ZIP file.")
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

    print(f"\n✅ Successfully loaded file with {len(df)} records and {len(df.columns)} columns.")
    
    df_filtered = df.drop(columns=['Timestamp', 'Name', 'Roll No', 'Class'], errors='ignore')
    
    print(f"\n📊 Analyzing {len(df_filtered.columns)} questions...")

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

def main():
    """Main function with error handling and user feedback."""
    print("\n🔍 Starting Feedback Sentiment Analysis")
    print("=" * 50)
    
    try:
        # Try with the hardcoded path first
        if os.path.exists(FILE_PATH):
            print(f"\n✅ Found file at path: {FILE_PATH}")
            process_feedback(FILE_PATH)
        else:
            # Look for all potential feedback files in the current directory
            print(f"\n⚠️ File not found at '{FILE_PATH}'")
            print("🔍 Searching for feedback files in the current directory...")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            excel_files = [f for f in os.listdir(current_dir) if f.endswith('.xlsx') or f.endswith('.csv')]
            
            if excel_files:
                print(f"\n✅ Found {len(excel_files)} potential feedback file(s):")
                for i, file in enumerate(excel_files):
                    print(f"   {i+1}. {file}")
                
                file_to_use = os.path.join(current_dir, excel_files[0])
                print(f"\n🔍 Using file: {file_to_use}")
                process_feedback(file_to_use)
            else:
                print("\n❌ No Excel or CSV files found in the current directory.")
                print("📝 Please place your feedback file in the same folder as this script or update the FILE_PATH in the code.")
    
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    
    finally:
        print("\n" + "=" * 50)
        print("🏁 Analysis complete")
        print("\nPress Enter to exit...", end="")
        input()  # Wait for user input before closing

if __name__ == "__main__":
    main()
