import sys
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



def predict_from_cli():
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'Your movie review here'")
        return
    model = load_model('imdb_lstm_model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    user_review = sys.argv[1]

    clean_text = user_review.lower().replace('<br />', ' ')

    sequences = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequences, maxlen=200)

    prediction = model.predict(padded, verbose=0)[0][0]

    label = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
    print(f"\nReview: {user_review}")
    print(f"Sentiment: {label} (Confidence: {prediction:.4f})")

if __name__ == "__main__":
    predict_from_cli()