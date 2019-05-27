import tensorflow as tf

from model.utils import LayerNorm, Dense


class BertOutput:
    def __init__(self, config):
        self.dense = Dense(config.intermediate_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout_prob = config.hidden_dropout_prob

    @tf.function
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense.forward(hidden_states)
        hidden_states = tf.nn.dropout(hidden_states, self.dropout_prob)
        hidden_states = self.layer_norm.forward(hidden_states + input_tensor)
        return hidden_states
