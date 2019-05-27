import tensorflow as tf

from model.utils import Dense, LayerNorm


class BERTSelfOutput:
    def __init__(self, config):
        self.dense = Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout_rate = config.hidden_dropout_prob

    @tf.function
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense.forward(hidden_states)
        hidden_states = tf.nn.dropout(hidden_states, self.dropout_rate)
        hidden_states = self.layer_norm.forward(hidden_states + input_tensor)
        return hidden_states
