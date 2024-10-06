import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Flatten, TextVectorization, Dropout
import tensorflow_datasets as tfds
import tensorflow as tf

import string
import re
from com.iqvia.Embeddings import Embeddings
from com.iqvia.TransformerEncoder import TransformerEncoder

VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
num_heads = 2
num_layers = 1
dense_dim = 1024
SEQUENCE_LENGTH = 250
BATCH_SIZE = 64

## data cleanup
def clean_text(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "<[^>]", "")
    text = tf.strings.regex_replace(text, "[%s]" % re.escape(string.punctuation), "")
    return text

### loading data
train_ds, val_ds, test_ds = tfds.load("imdb_reviews", split=["train", "test[:50%]", "test[50%:]"], as_supervised=True)

## convert text into numerical vectors
vectorizer_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize=clean_text,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    name="vectorizer_layer"
)
train_dataset = train_ds.map(lambda x, y: x)
vectorizer_layer.adapt(train_dataset)

def vectorizer(review, label):
    return vectorizer_layer(review), label

train_dataset = train_ds.map(vectorizer)
val_dataset = val_ds.map(vectorizer)

## shuffle data
train_dataset = train_dataset.shuffle(25000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(25000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

################### Building Model ##########################
encoder_input = keras.layers.Input(shape=(None,))
emb = Embeddings(VOCAB_SIZE, EMBEDDING_DIM, sequence_length=SEQUENCE_LENGTH)
x = emb(encoder_input)
enc_mask = emb.compute_mask1(encoder_input)

for i in range(num_layers):
    x = TransformerEncoder(num_heads, dense_dim, EMBEDDING_DIM)(x)

x = Flatten()(x)
x = Dropout(0.4)(x)
output = keras.layers.Dense(1, activation="softmax")(x)
model = keras.models.Model(inputs=encoder_input, outputs=output)

# save model
checkpoint_dir = "./sentiment_anaysis_with_transformer.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# compile and fit the model
adamOpt = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=adamOpt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10,callbacks=[model_checkpoint_callback],batch_size=32,shuffle=True)

## Plotting train loss vs validation loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("model loss")
plt.legend(["train loss", "validation loss"])
plt.show()

# Plotting train loss vs validation loss
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("model loss")
plt.legend(["train accuracy", "validation accuracy"])
plt.show()