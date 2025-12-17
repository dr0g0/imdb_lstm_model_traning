import numpy as np
import pandas as pd
import pickle, sys
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

def train_and_export():
    if len(sys.argv) < 2:
        print("Usage: python train.py 'dataset file path'")
        return
    df = pd.read_csv(sys.argv[1])
    df['review'] = df['review'].str.replace('<br />', ' ')
    df['review'] = df['review'].str.lower()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    reviews = df['review'].values
    labels = df['sentiment'].values

    max_words = 10000
    max_len = 200
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    X = pad_sequences(sequences, maxlen=max_len)
    y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    model.save('imdb_lstm_model.h5')

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training complete. Model and Tokenizer have been exported.")

if __name__ == "__main__":
    train_and_export()