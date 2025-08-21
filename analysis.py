import pandas as pd

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

if __name__ == "__main__":
    # Data cleaning
    df  = load_data()
    df2 = filtering(df)
    df3 = add_clean_text_column(df2)
    df3.to_csv("data.csv", index=False, encoding="utf-8")


