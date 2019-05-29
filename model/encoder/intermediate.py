import tensorflow as tf

from ..utils import Dense, GELU


class BERTIntermediate:
    def __init__(self, config):
        self.dense = Dense(config.hidden_size, config.intermediate_size)
        self.activation = GELU()

    @tf.function
    def forward(self, hidden_states):
        hidden_states = self.dense.forward(hidden_states)
        hidden_states = self.activation.forward(hidden_states)
        return hidden_states

