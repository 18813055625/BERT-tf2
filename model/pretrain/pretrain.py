import tensorflow as tf

from .masked_lm import MaskedLMModel
from .next_sentence import NextSentencePredictionModel


class BertPreTrainModel:
    def __init__(self, config):
        self.masked_lm = MaskedLMModel(config)
        self.next_sentence = NextSentencePredictionModel(config)

    @tf.function
    def forward(self, bert_output):
        masked_lm_output = self.masked_lm.forward(bert_output)
        next_sentence_output = self.next_sentence.forward(bert_output)
        return masked_lm_output, next_sentence_output
