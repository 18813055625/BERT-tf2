import tensorflow as tf

from ..utils import LayerNorm


class BERTEmbedding:
    def __init__(self, config):
        token_embed_init = tf.random.normal(config.vocab_size, config.hidden_size)
        position_embed_init = tf.random.normal(config.max_seq_length, config.embed_dim)
        segment_embed_init = tf.random.normal(config.segment_size, config.embed_dim)

        self.token_embed = tf.Variable(token_embed_init, name="token_embed")
        self.position_embed = tf.Variable(position_embed_init, name="position_embed")
        self.segment_embed = tf.Variable(segment_embed_init, name="segment_embed")

        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout_prob = config.hidden_dropout_prob
        self.max_seq_length = config.max_seq_length

    @tf.function
    def forward(self, token_input, segment_input=None):
        position_input = tf.expand_dims(tf.range(0, self.max_seq_length), 0)
        position_input = tf.tile(position_input, [tf.shape(token_input)[0], 1])

        if segment_input is None:
            segment_input = tf.zeros(token_input.shape)

        token_embed_output = tf.nn.embedding_lookup(self.token_embed, token_input)
        position_embed_output = tf.nn.embedding_lookup(self.position_embed, position_input)
        segment_embed_output = tf.nn.embedding_lookup(self.segment_embed, segment_input)

        embedding = token_embed_output + position_embed_output + segment_embed_output
        embedding = self.layer_norm.forward(embedding)
        embedding = tf.nn.dropout(embedding, self.dropout_prob)

        return embedding
