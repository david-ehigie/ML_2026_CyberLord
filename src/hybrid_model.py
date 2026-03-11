import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
from tensorflow.keras.models import Model

print("Loading dataset...")

df = pd.read_csv("data/raw/phishing.csv")

# URLs
urls = df.iloc[:,0].astype(str)

# labels
labels = df.iloc[:,-1]
labels = labels.replace({"legitimate":0,"phishing":1})
labels = labels.astype("float32")

# engineered features
X_features = pd.read_csv("data/processed/X_train.csv")
X_features = pd.concat([X_features, pd.read_csv("data/processed/X_test.csv")])

# select top 20 features
selected_features = [
85,86,56,83,20,
82,58,50,46,74,
57,25,0,62,1,
44,49,67,39,78
]

X_features = X_features.iloc[:,selected_features]

# force numeric
X_features = X_features.apply(pd.to_numeric, errors="coerce").fillna(0)

# consistent split
url_train, url_test, feat_train, feat_test, y_train, y_test = train_test_split(
    urls,
    X_features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# tokenize URLs
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(url_train)

train_seq = tokenizer.texts_to_sequences(url_train)
test_seq = tokenizer.texts_to_sequences(url_test)

max_length = 150

train_pad = pad_sequences(train_seq,maxlen=max_length)
test_pad = pad_sequences(test_seq,maxlen=max_length)

vocab_size = len(tokenizer.word_index)+1

# CNN branch
url_input = Input(shape=(max_length,))

x = Embedding(vocab_size,64)(url_input)
x = Conv1D(128,5,activation="relu")(x)
x = GlobalMaxPooling1D()(x)

# engineered feature branch
feature_input = Input(shape=(20,))

# merge
combined = Concatenate()([x,feature_input])

z = Dense(64,activation="relu")(combined)
z = Dropout(0.3)(z)
z = Dense(1,activation="sigmoid")(z)

model = Model(inputs=[url_input,feature_input],outputs=z)

model.compile(
optimizer="adam",
loss="binary_crossentropy",
metrics=["accuracy"]
)

model.summary()

print("Training hybrid model...")

model.fit(
[train_pad,feat_train],
y_train,
epochs=10,
batch_size=64,
validation_split=0.2
)

print("Evaluating hybrid model...")

loss,acc = model.evaluate([test_pad,feat_test],y_test)

print("\nHybrid Model Accuracy:",acc)