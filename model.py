import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["I am happy to meet my friends. We are planning to go a party.", 
            "I had a bad day at school. i got hurt while playing football"]

# Tokenization
tokeniser = Tokenizer(num_words = 13945, oov_token= "<OOV>")
tokeniser.fit_on_texts(sentence)
# Create a word_index dictionary
word_index = tokeniser.word_index
sequence = tokeniser.texts_to_sequence(sentence)
# Padding the sequence
padded = pad_sequences(sequence, maxlen=100, padding="pre", truncating="post")
# Define the model using .h5 file
model = tensorflow.keras.models.load_model("Text_Emotion.h5")
# Test the model
result = model.predict(padded)
predict_class = np.argmax(result, axis=1)
# Print the result
print(predict_class)
