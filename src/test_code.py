import unittest

from bio_transformer_training import get_data


class TestGetData(unittest.TestCase):
    def test_datasets(self):
        X, labels, valid_labels, pat_start_end = get_data(pretrain_mode=False, dataset="TUSZ")
        for mode in  ["train", "test", "val"]:
            self.assertEqual(X[mode].shape[0], labels[mode].shape[0])


if __name__ == '__main__':
    unittest.main()