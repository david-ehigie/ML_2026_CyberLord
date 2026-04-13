import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# LOAD DATA
# -------------------------------
print("Loading processed datasets...")

X_train = pd.read_csv("data/processed/X_train.csv").values
X_test = pd.read_csv("data/processed/X_test.csv").values

y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# -------------------------------
# LOAD URL DATA (for CNN)
# -------------------------------
df = pd.read_csv("data/raw/phishing.csv")
urls = df.iloc[:, 0].astype(str)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(urls)

X_url = tokenizer.texts_to_sequences(urls)
X_url = pad_sequences(X_url, maxlen=150)

# Split URL same way
X_url_train = X_url[:len(X_train)]
X_url_test = X_url[len(X_train):]

# -------------------------------
# MODEL ARCHITECTURE
# -------------------------------

# URL input (CNN)
url_input = Input(shape=(150,))
embedding = Embedding(input_dim=5000, output_dim=64)(url_input)
conv = Conv1D(128, 5, activation='relu')(embedding)
pool = GlobalMaxPooling1D()(conv)

# Feature input (RF features)
feature_input = Input(shape=(X_train.shape[1],))

# Combine both
combined = Concatenate()([pool, feature_input])

dense = Dense(64, activation='relu')(combined)
dropout = Dropout(0.5)(dense)

# ✅ MULTI-CLASS OUTPUT
output = Dense(3, activation='softmax')(dropout)

model = Model(inputs=[url_input, feature_input], outputs=output)

# -------------------------------
# COMPILE MODEL
# -------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# TRAIN MODEL
# -------------------------------
print("Training hybrid model...")

model.fit(
    [X_url_train, X_train],
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_url_test, X_test], y_test)
)

# -------------------------------
# EVALUATE MODEL
# -------------------------------
print("Evaluating hybrid model...")

loss, acc = model.evaluate([X_url_test, X_test], y_test)
print("\nHybrid Model Accuracy:", acc)

# Predictions
y_pred = model.predict([X_url_test, X_test])
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))