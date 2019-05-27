import tensorflow as tf


class GELU:
    @tf.function
    def forward(self, x):
        return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
