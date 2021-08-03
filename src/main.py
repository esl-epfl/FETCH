import numpy as np
import tensorflow as tf
from utils.models import EpilepsyEncoder
import utils.data as dt


def main():
    sample_epilepsy_encoder = EpilepsyEncoder(num_layers=8, d_model=108, num_heads=12, dff=512,
                                              target_size=2, pe_input=3600)

    X, y = dt.get_eglass_features(test_patient=1)
    print("X: {}".format(X.shape))

    fn_out = sample_epilepsy_encoder(X[:32,:,:], training=False,
                                     enc_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)


if __name__ == '__main__':
    main()
