"""
Clean standalone implementation of Membership Function activations
and the MembershipLSTMSNPCell for LSTM-SNP models.

Activation replacements for Consumption gate c(t) and Generation gate o(t):
  - Difference of Gaussians (DoG)
  - Signed Gaussian (SG)
  - Generalized Bell Function (GB)

Reset gate r(t) always uses tanh.
Spike generation a(t) always uses tanh.
"""

import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# 1. Modular Activation Functions
# ============================================================

def dog(x, mu1=-1.0, mu2=1.0, sigma1=0.5, sigma2=0.5):
    """Difference of Gaussians (DoG) activation function.
    
    DoG(x) = exp(-(x-mu1)^2 / (2*sigma1^2)) - exp(-(x-mu2)^2 / (2*sigma2^2))
    """
    term1 = tf.exp(-tf.square(x - mu1) / (2.0 * sigma1**2))
    term2 = tf.exp(-tf.square(x - mu2) / (2.0 * sigma2**2))
    return term1 - term2


def signed_gaussian(x, sigma=0.5):
    """Signed Gaussian activation function.
    
    SG(x) = x * exp(-x^2 / (2*sigma^2))
    """
    return x * tf.exp(-tf.square(x) / (2.0 * sigma**2))


def generalized_bell(x, a=1.0, b=2.0, c_val=0.0):
    """Generalized Bell Membership Function.
    
    GB(x) = 1 / (1 + |((x-c)/a)|^(2b))
    """
    abs_term = tf.abs((x - c_val) / a)
    denom = 1.0 + tf.pow(abs_term, 2.0 * b)
    return 1.0 / denom


# ============================================================
# 2. MembershipLSTMSNPCell
# ============================================================

@tf.keras.utils.register_keras_serializable()
class MembershipLSTMSNPCell(layers.Layer):
    """
    LSTM-SNP Cell with Membership Function gate replacement.

    Gates:
      r(t) = tanh(W_r x(t) + U_r u(t-1) + b_r)            [reset — unchanged]
      c(t) = clip(MF(W_c x(t) + U_c u(t-1) + b_c), -1, 1) [consumption]
      o(t) = clip(MF(W_o x(t) + U_o u(t-1) + b_o), -1, 1) [generation/output]
      a(t) = tanh(W_a x(t) + U_a u(t-1) + b_a)             [spikes — unchanged]

    State Update:
      u(t) = r(t) * u(t-1) - c(t) * a(t)
      h(t) = o(t) * a(t)

    Supports mf_type: 'dog', 'sg', 'gb'
    state_size is a tuple (u, c, o) so return_state=True exposes gate values.
    """
    def __init__(self, units, mf_type='dog', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.mf_type = mf_type
        self.state_size = (units, units, units)
        self.output_size = units

        if mf_type == 'dog':
            self.mf = dog
        elif mf_type == 'sg':
            self.mf = signed_gaussian
        elif mf_type == 'gb':
            self.mf = generalized_bell
        else:
            raise ValueError(f"Unknown mf_type: {mf_type}. Choose from 'dog', 'sg', 'gb'.")

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer='orthogonal', name='recurrent_kernel')
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer='zeros', name='bias')

    def call(self, inputs, states):
        u_tm1 = states[0]

        z = (tf.matmul(inputs, self.kernel)
             + tf.matmul(u_tm1, self.recurrent_kernel)
             + self.bias)

        z0 = z[:, :self.units]
        z1 = z[:, self.units:2*self.units]
        z2 = z[:, 2*self.units:3*self.units]
        z3 = z[:, 3*self.units:]

        r = tf.tanh(z0)
        c = tf.clip_by_value(self.mf(z1), -1.0, 1.0)
        o = tf.clip_by_value(self.mf(z2), -1.0, 1.0)
        a = tf.tanh(z3)

        u = r * u_tm1 - c * a
        h = o * a

        return h, [u, c, o]

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'mf_type': self.mf_type})
        return config
