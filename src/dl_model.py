import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.metrics import Precision, Recall

print("Loading dataset...")
df = pd.read_csv("data/raw/phishing.csv")

# ---- Extract URL column ----
urls = df.iloc[:, 0].astype(str)

# ---- Extract labels ----
labels = df.iloc[:, -1]

# Convert string labels to numeric
labels = labels.replace({
    "legitimate": 0,
    "phishing": 1
})

labels = labels.astype("int32")

print("Number of URLs:", len(urls))

# ---- Train test split ----
X_train, X_test, y_train, y_test = train_test_split(
    urls, labels, test_size=0.2, random_state=42
)

# ---- Tokenization (character level) ----
print("Tokenizing URLs...")
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 150
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size:", vocab_size)

# ---- Build LSTM Model ----
print("Building LSTM model...")
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall()]
)

model.summary()

# ---- Train ----
print("Training model...")
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# ---- Evaluate ----
print("Evaluating model...")
loss, acc, prec, rec = model.evaluate(X_test_pad, y_test)

print("\nDeep Learning Results")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)