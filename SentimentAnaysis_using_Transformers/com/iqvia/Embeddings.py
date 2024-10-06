import tensorflow as tf
import keras
from keras.layers import Input
from com.iqvia.Positional_Embedding import positional_embedding


class Embeddings(keras.layers.Layer):
    def __init__(self, vocab_size, emb_dim, sequence_length):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.sequence_length = sequence_length
        self.token_embeddings = keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim)

    def call(self, inputs, *args, **kwargs):
        x = self.token_embeddings(inputs)
        y = positional_embedding(emb_dim=self.emb_dim, sequence_length=self.sequence_length)
        return x + y

    def compute_mask1(self, input):
        return tf.math.not_equal(input, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "emb_dim": self.emb_dim,
            "sequence_length": self.sequence_length
        })
        return config
