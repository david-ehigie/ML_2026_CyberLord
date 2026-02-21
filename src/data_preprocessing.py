import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    return df

def clean_data(df):
    print("Cleaning dataset...")
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def preprocess_data(df):
    print("Separating features and labels...")

    # Drop URL column if present
    if 'url' in df.columns:
        df = df.drop(columns=['url'])

    # If first column is text (URL), drop it
    if df.iloc[:, 0].dtype == 'object':
        df = df.iloc[:, 1:]

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y):
    print("Splitting dataset...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    df = load_data("data/raw/phishing.csv")
    df = clean_data(df)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

    print("Preprocessing finished successfully!")