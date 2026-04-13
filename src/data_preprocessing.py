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

# ✅ Severity function
def assign_severity(row):
    score = 0

    if row['length_url'] > 75:
        score += 1
    if row['nb_dots'] > 3:
        score += 1
    if row['nb_subdomains'] > 2:
        score += 1

    if row['ip'] == 1:
        score += 2
    if row['https_token'] == 1:
        score += 1
    if row['phish_hints'] > 0:
        score += 2

    if score <= 1:
        return 0  # Legitimate
    elif score <= 3:
        return 1  # Suspicious
    else:
        return 2  # Phishing

def preprocess_data(df):
    print("Creating severity labels...")

    df['severity'] = df.apply(assign_severity, axis=1)

    print("Separating features and labels...")

    # Drop URL column
    if 'url' in df.columns:
        df = df.drop(columns=['url'])

    # Drop any remaining object/text columns (VERY IMPORTANT)
    df = df.select_dtypes(exclude=['object'])

    # ✅ FIX: remove BOTH labels from features
    y = df['severity']
    X = df.drop(columns=['severity'])

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

    print("Columns:", df.columns)

    X, y = preprocess_data(df)

    print("Severity distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = split_data(X, y)

    label_map = {0: "Legitimate", 1: "Suspicious", 2: "Phishing"}

    print("Severity distribution (readable):")
    print(y.map(label_map).value_counts())


    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

    print("Preprocessing finished successfully!")