import tensorflow as tf
import math

from model.utils.dense import Dense


class SelfAttention:
    def __init__(self, config):
        self.num_attention_head = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_head * self.attention_head_size

        self.query = Dense(config.hidden_size, self.all_head_size, name="query/")
        self.key = Dense(config.hidden_size, self.all_head_size, name="key/")
        self.value = Dense(config.hidden_size, self.all_head_size, name="value/")

        self.dropout_prob = config.dropout_prob

    @tf.function
    def transpose_for_scores(self, x):
        new_shape = tf.shape(x)[:-1] + (self.num_attention_head, self.attention_head_size)
        x = tf.reshape(x, new_shape)
        x = tf.transpose(x, [0, 2, 1, 3])
        return x

    @tf.function
    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query.forward(hidden_states)
        mixed_key_layer = self.key.forward(hidden_states)
        mixed_value_layer = self.key.forward(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, [-1, -2]))
        attention_scores = tf.math.divide(attention_scores, math.sqrt(self.attention_head_size))
        attention_scores = tf.add(attention_scores, attention_mask)

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = tf.nn.dropout(attention_probs, self.dropout_prob)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        new_context_layer_shape = tf.shape(context_layer)[:-2] + (self.all_head_size,)
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        return context_layer
