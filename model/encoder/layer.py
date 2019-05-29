import tensorflow as tf

from .intermediate import BERTIntermediate
from ..attention import BertAttention
from ..utils import LayerNorm, Dense


class BertLayer:
    def __init__(self, config):
        self.attention = BertAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BertLayerOutput(config)

    @tf.function
    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention.forward(hidden_states, attention_mask)
        intermediate_output = self.intermediate.forward(attention_output)
        layer_output = self.output.forward(intermediate_output, attention_output)
        return layer_output


class BertLayerOutput:
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
