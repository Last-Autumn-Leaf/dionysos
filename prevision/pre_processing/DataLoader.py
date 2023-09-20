from .dataGenerator import get_all_data, get_data_filtered_data, addDates
from .. import timeThis

import torch
from torch.utils.data import Dataset


class WindowGenerator(Dataset):
    def __init__(self, X, Y, input_sequence_length, output_sequence_length):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        assert (len(self.X) - self.input_sequence_length - self.output_sequence_length + 1 > 0), \
            f'Dataset size too small for current options\ndata:{len(self.X)}, ' \
            f'input_sequence_length:{input_sequence_length}, output_sequence_length:{output_sequence_length}'

    def __len__(self):  # number of windows we can generate to work it must be at least 1
        return len(self.X) - self.input_sequence_length - self.output_sequence_length + 1

    def __getitem__(self, idx):
        x = torch.cat(
            (self.X[idx:idx + self.input_sequence_length], self.Y[idx:idx + self.input_sequence_length][:, None]),
            1)
        y = self.Y[
            idx + self.input_sequence_length:idx + self.input_sequence_length + self.output_sequence_length]
        return x, y

    def getInOutSize(self):
        x, y = self.__getitem__(0)
        in_size = x.shape[1] if len(x.shape) > 1 else 1
        out_size = y.shape[1] if len(y.shape) > 1 else 1
        return in_size, out_size


class DataLoader():
    def __init__(self, options):
        self.options = options
        X, Y = get_all_data() if options.targetFeatures is None else get_data_filtered_data(options.targetFeatures)
        self.setData(X, Y)

    def getData(self):
        return self.X, self.Y

    @timeThis("Data set in : ")
    def setData(self, X, Y):
        self.X = X.to_numpy(dtype='float64')
        self.Y = Y.to_numpy(dtype='float64')

        self.split_dataset()
        self.train_dataset = WindowGenerator(self.x_train, self.y_train, self.options.input_sequence_length,
                                             self.options.output_sequence_length)
        self.test_dataset = WindowGenerator(self.x_test, self.y_test, self.options.input_sequence_length,
                                            self.options.output_sequence_length)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.options.batch_size,
                                                        shuffle=self.options.shuffle)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.options.batch_size,
                                                       shuffle=self.options.shuffle)

    def split_dataset(self):
        n = len(self.X)
        dataset_split = self.options.dataset_split
        self.x_train = self.X[:int(n * dataset_split)]
        self.y_train = self.Y[:int(n * dataset_split)]
        self.x_test = self.X[int(n * dataset_split):]
        self.y_test = self.Y[int(n * dataset_split):]

        window_size = self.options.input_sequence_length + self.options.output_sequence_length
        assert len(
            self.y_train) - window_size + 1 > 0, f"train dataset size too small for current options\n Try changing split, window size or increase the dataset size (currently:{len(self.y_train)})"
        assert len(
            self.y_test) - window_size + 1 > 0, f"test dataset size too small for current options\n Try changing split, window size or increase the dataset size (currently:{len(self.y_test)})"

    def getTrainDataLoader(self):
        return self.train_loader

    def getTestDataLoader(self):
        return self.test_loader

    def getData(self):
        return self.X, self.Y

    # TODO create a DB using training data but more importantly TEST DATA
