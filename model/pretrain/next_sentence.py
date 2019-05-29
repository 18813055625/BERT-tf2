import tensorflow as tf

from model.utils import Dense


class NextSentencePredictionModel:
    def __init__(self, config):
        self.projection = Dense(config.hidden_size, 2)

    @tf.function
    def forward(self, bert_output):
        projected_output = self.projection.forward(bert_output)
        softmax_output = tf.nn.softmax(projected_output, axis=-1)
        return softmax_output
