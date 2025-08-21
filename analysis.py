import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data():
    raw_df = pd.read_csv("raw_twitter_comments_data.csv", encoding="latin-1", header=None)
    df = raw_df[[0, 5]].copy()
    df.columns = ["polarity", "text"]
    print(df.head())
    return df

def filtering(df):
    """Keep only positive (4) and negative (0) tweets; drop neutral (2)."""
    df2 = df[df["polarity"].isin([0, 4])].copy()
    # ensure ints before mapping, in case they were strings
    df2["polarity"] = df2["polarity"].astype(int).map({0: 0, 4: 1})
    print(df2["polarity"].value_counts())
    return df2

def clean_text(text):
    """Standardise text to lowercase"""
    if not isinstance(text, str):
        return ""
    return text.lower()

def add_clean_text_column(df):
    """Add cleaned lowercase text column"""
    df2 = df.copy()
    df2["clean_text"] = df2["text"].astype(str).apply(clean_text)
    print(df2[["text", "clean_text"]].head())
    return df2

def data_split(df):
    """Split features/labels for modeling"""
    X = df["clean_text"]
    y = df["polarity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    return X_train, X_test, y_train, y_test

def vectorisation(X_train, X_test, max_features=5000, ngram_range=(1, 2)):
    """Convert text into numerical features"""
    vectoriser = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectoriser.fit_transform(X_train)
    X_test_tfidf = vectoriser.transform(X_test)
    print("TF-IDF shape (train):", X_train_tfidf.shape)
    print("TF-IDF shape (test):", X_test_tfidf.shape)
    return vectoriser, X_train_tfidf, X_test_tfidf

if __name__ == "__main__":
    # Data cleaning
    df  = load_data()
    df2 = filtering(df)
    df3 = add_clean_text_column(df2)
    df3.to_csv("data.csv", index=False, encoding="utf-8")

    # Model train and test
    X_train, X_test, y_train, y_test = data_split(df3)
    vectoriser, X_train_tfidf, X_test_tfidf = vectorisation(X_train, X_test)


