import tensorflow as tf

from .self_attention import SelfAttention
from ..utils import Dense, LayerNorm


class BertAttention:
    def __init__(self, config):
        self.attention = SelfAttention(config)
        self.output = BERTSelfOutput(config)

    @tf.function
    def forward(self, input_tensor, attention_mask):
        self_output = self.attention.forward(input_tensor, attention_mask)
        attention_output = self.output.forward(self_output, input_tensor)
        return attention_output


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
