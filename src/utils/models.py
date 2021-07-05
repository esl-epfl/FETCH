import tensorflow as tf
from utils.Encoder import Encoder


class EpilepsyEncoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_size, pe_input, rate=0.1):
        super(EpilepsyEncoder, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                 pe_input, rate)

        self.final_layer = tf.keras.layers.Dense(target_size, activation='softmax')

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output
