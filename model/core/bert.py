import tensorflow as tf

from ..encoder import BertEncoder
from ..embedding import BERTEmbedding

from ..utils import Dense


class BertModel:
    def __init__(self, config):
        self.embedding = BERTEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pool_projection = Dense(config.hidden_size, config.hidden_size)

    def forward(self, input_tokens, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True):
        embedding_output = self.embedding.forward(input_tokens, token_type_ids)
        encoded_layers = self.encoder.forward(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_all_encoded_layers=output_all_encoded_layers
        )

        cls_embed = tf.squeeze(encoded_layers[:, 0:1, :], axis=1)
        projected_cls = self.pool_projection.forward(cls_embed)
