import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam

with open("LSTM.txt", 'r', encoding='utf-8') as f:
    text_data = f.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text_data.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]

model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_length - 1),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(X, y, epochs=1, batch_size=32, verbose=1)

def predict_next_word(model, input_text, tokenizer, max_sequence_length):
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]
    input_sequence = pad_sequences([input_sequence], maxlen=max_sequence_length - 1, padding='pre')
    predicted_word_index = np.argmax(model.predict(input_sequence), axis=1)[0]
    return tokenizer.index_word.get(predicted_word_index, "<unknown>")

input_text = "I am a little"
predicted_word = predict_next_word(model, input_text, tokenizer, max_sequence_length)
print(f"Predicted next word: {predicted_word}")

train_loss, train_accuracy = model.evaluate(X, y, verbose=0)
print(f"Training Accuracy: {train_accuracy:.2f}")
