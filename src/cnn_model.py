import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.metrics import Precision, Recall

print("Loading dataset...")
df = pd.read_csv("data/raw/phishing.csv")

# Extract URL and labels
urls = df.iloc[:, 0].astype(str)
labels = df.iloc[:, -1]

labels = labels.replace({"legitimate": 0, "phishing": 1})
labels = labels.astype("int32")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    urls, labels, test_size=0.2, random_state=42
)

# Tokenization
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 150
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size:", vocab_size)

# Build CNN model
print("Building CNN model...")
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall()]
)

model.summary()

# Train
print("Training CNN...")
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
print("Evaluating CNN...")
loss, acc, prec, rec = model.evaluate(X_test_pad, y_test)

print("\nCNN Results")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)