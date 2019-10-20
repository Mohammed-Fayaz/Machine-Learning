import unittest

from Templates import regression_pipeline


class TestRead(unittest.TestCase):
    """
    Tests the read function
    """

    def setUp(self):
        self.TRAIN_PATH = 'C:\\Users\\fayaz\\PycharmProjects\\Machine-Learning\\Tests\\train.csv'
        self.TEST_PATH = 'C:\\Users\\fayaz\\PycharmProjects\\Machine-Learning\\Tests\\test.csv'

    def tearDown(self):
        pass

    def test_if_read_works_alright(self):
        X_train, y_train, X_test = regression_pipeline.read(train_path=self.TRAIN_PATH, test_path=self.TEST_PATH,
                                                            label_name='d')


if __name__ == '__main__':
    unittest.main()
