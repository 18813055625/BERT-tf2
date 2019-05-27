import tensorflow as tf

from .attention import BertAttention
from .intermediate import BERTIntermediate
from .output import BertOutput


class BertLayer:
    def __init__(self, config):
        self.attention = BertAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BertOutput(config)

    @tf.function
    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention.forward(hidden_states, attention_mask)
        intermediate_output = self.intermediate.forward(attention_output)
        layer_output = self.output.forward(intermediate_output, attention_output)
        return layer_output
