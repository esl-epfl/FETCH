import numpy as np
import tensorflow as tf
from utils.models import EpilepsyEncoder


def main():
    sample_epilepsy_encoder = EpilepsyEncoder(num_layers=8, d_model=108, num_heads=12, dff=512,
                                              target_size=2, pe_input=3600)

    temp_input = tf.random.uniform((64, 899, 108), dtype=tf.float64, minval=0, maxval=200)

    fn_out = sample_epilepsy_encoder(temp_input, training=False,
                                        enc_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)


if __name__ == '__main__':
    main()
