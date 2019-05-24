import tensorflow as tf


class LayerNorm:
    def __init__(self, hidden_size, eps=1e-12):
        self.weight = tf.Variable(tf.ones(hidden_size), name="layer_norm_w")
        self.bias = tf.Variable(tf.zeros(hidden_size), name="layer_norm_b")
        self.variance_epsilon = eps

    @tf.function
    def forward(self, x):
        u = tf.reduce_mean(x, keepdims=True)
        s = tf.reduce_mean(tf.pow(x - u, 2), axis=-1, keepdims=True)
        x = (x - u) / tf.sqrt(s + self.variance_epsilon)
        x = tf.multiply(self.weight, x) + self.bias
        return x
