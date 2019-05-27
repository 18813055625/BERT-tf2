import tensorflow as tf

from .self_attention import SelfAttention
from .self_output import BERTSelfOutput


class BertAttention:
    def __init__(self, config):
        self.attention = SelfAttention(config)
        self.output = BERTSelfOutput

    @tf.function
    def forward(self, input_tensor, attention_mask):
        self_output = self.attention.forward(input_tensor, attention_mask)
        attention_output = self.output.forward(self_output, input_tensor)
        return attention_output
