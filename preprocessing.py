import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from textblob import TextBlob

# ================================
# SETUP NLP TOOLS
# ================================
stemmer = StemmerFactory().create_stemmer()
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())


# ================================
# CLEANING TEXT
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)     # hapus URL
    text = re.sub(r"[^\w\s]", " ", text)           # hapus simbol
    text = re.sub(r"\d+", "", text)                # hapus angka
    text = re.sub(r"\s+", " ", text).strip()       # rapikan spasi
    return text


# ================================
# TOKENIZATION
# ================================
def tokenize(text):
    return text.split()


# ================================
# STOPWORD REMOVAL
# ================================
def remove_stopwords(tokens):
    return [w for w in tokens if w not in stopwords]


# ================================
# STEMMING
# ================================
def stemming(tokens):
    return [stemmer.stem(w) for w in tokens]


# ================================
# SENTIMENT ANALYSIS
# ================================
def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_sentiment_label(score):
    if score > 0.1:
        return "Positif"
    elif score < -0.1:
        return "Negatif"
    return "Netral"


# ================================
# MAIN PREPROCESSING
# ================================
def preprocess_reviews(reviews):
    results = []

    for text in reviews:
        original = text
        
        # 1. Case Folding + Cleaning
        cleaned = clean_text(original)

        # 2. Tokenization
        tokens = tokenize(cleaned)

        # 3. Stopword Removal
        filtered = remove_stopwords(tokens)

        # 4. Stemming
        stemmed = stemming(filtered)

        # 5. Final Text untuk Sentiment
        final_text = " ".join(stemmed)

        # 6. Sentiment
        score = get_sentiment_score(final_text)
        label = get_sentiment_label(score)

        results.append({
            "original": original,
            "cleaned": cleaned,
            "tokens": tokens,
            "filtered": filtered,
            "stemmed": stemmed,
            "final_text": final_text,
            "sentiment_score": score,
            "sentiment_label": label
        })
    
    return results


# ================================
# STATISTIK TOKEN
# ================================
def get_statistics(results):
    total_reviews = len(results)

    total_tokens_original = sum(len(r["tokens"]) for r in results)
    total_tokens_final = sum(len(r["stemmed"]) for r in results)

    avg_original = total_tokens_original / total_reviews if total_reviews > 0 else 0
    avg_final = total_tokens_final / total_reviews if total_reviews > 0 else 0

    reduction = 100 - ((avg_final / avg_original) * 100) if avg_original > 0 else 0

    return {
        "total_reviews": total_reviews,
        "avg_tokens_original": round(avg_original, 2),
        "avg_tokens_final": round(avg_final, 2),
        "reduction_rate": round(reduction, 2)
    }


# ================================
# STATISTIK SENTIMENT
# ================================
def get_sentiment_statistics(results):
    positive = sum(1 for r in results if r["sentiment_label"] == "Positif")
    neutral  = sum(1 for r in results if r["sentiment_label"] == "Netral")
    negative = sum(1 for r in results if r["sentiment_label"] == "Negatif")

    total = len(results)

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_percentage": round((positive / total) * 100, 2) if total else 0,
        "neutral_percentage": round((neutral / total) * 100, 2) if total else 0,
        "negative_percentage": round((negative / total) * 100, 2) if total else 0
    }
