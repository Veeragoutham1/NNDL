from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load a smaller dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
text = "\n".join(dataset["train"]["text"][:1000])  # First 1000 lines

# Tokenize and limit vocabulary size
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Prepare sequences
input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = 20
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Predictors and labels
predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define a smaller model
model = Sequential([
    Embedding(total_words, 50, input_length=max_sequence_len - 1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train with fewer epochs
history = model.fit(predictors, labels, epochs=10, verbose=1)

import numpy as np

# Function to generate predictions
def generate_next_word(model, tokenizer, input_text, max_sequence_len=10):
    # Step 1: Tokenize and convert input_text to sequences
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]

    # Step 2: If the sequence is too long, truncate it to fit the model's input length
    if len(input_sequence) > max_sequence_len - 1:
        input_sequence = input_sequence[-(max_sequence_len - 1):]

    # Step 3: Pad the sequence if it's shorter than max_sequence_len - 1
    input_sequence = np.pad(input_sequence, (max_sequence_len - 1 - len(input_sequence), 0), mode='constant')

    # Step 4: Reshape input_sequence for LSTM model (as batch_size, time_steps)
    input_sequence = np.array(input_sequence).reshape(1, max_sequence_len - 1)

    # Step 5: Make prediction
    prediction = model.predict(input_sequence)

    # Step 6: Get the index of the word with the highest probability
    predicted_index = np.argmax(prediction)

    # Step 7: Convert the index back to a word
    predicted_word = tokenizer.index_word[predicted_index]

    return predicted_word

# Example Usage:
input_text = "The quick brown fox"
predicted_word = generate_next_word(model, tokenizer, input_text)
print(f"Input: {input_text}")
print(f"Predicted next word: {predicted_word}")
