import tensorflow as tf


class Dense:
    def __init__(self, input_dim, hidden_dim, name="dense/"):
        w_init = tf.random.normal(input_dim, hidden_dim)
        b_init = tf.random.normal(hidden_dim)
        self.w = tf.Variable(w_init, name=f"{name}dense_w")
        self.b = tf.Variable(b_init, name=f"{name}dense_b")

    @tf.function
    def forward(self, x):
        return tf.add(tf.matmul(x, self.w), self.b)
