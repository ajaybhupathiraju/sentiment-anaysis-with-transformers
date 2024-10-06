import tensorflow as tf
import keras
from keras.layers import Input, MultiHeadAttention, LayerNormalization
from tensorflow import newaxis


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, num_heads, dense_dim, emd_dim):
        super(TransformerEncoder, self).__init__()
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.emd_dim = emd_dim

        self.attention1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.emd_dim)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.linear_projection = keras.models.Sequential(
            [
                keras.layers.Dense(units=self.dense_dim, activation="relu"),
                keras.layers.Dense(units=self.emd_dim)
            ]
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask[:, newaxis, :], dtype="int32")
            T = tf.shape(mask)[2]
            mask = tf.repeat(mask, T, axis=1)

        attention_output1 = self.attention1(query=inputs, key=inputs, value=inputs, attention_mask=mask)

        norm1 = self.layernorm1(attention_output1 + inputs)
        linear_proj = self.linear_projection(norm1)
        return self.layernorm2(linear_proj + norm1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
            "emd_dim": self.emd_dim
        })
        return config
