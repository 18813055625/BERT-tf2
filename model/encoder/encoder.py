import tensorflow as tf

from .layer import BertLayer


class BertEncoder:
    def __init__(self, config):
        self.layers = [
            BertLayer(config)
            for _ in range(config.num_hidden_layers)
        ]

    @tf.function
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
