from .DataGenerator import get_all_data, get_data_filtered_data, getAllDataFromCSV, addDates, addSportBroadcast
from .. import timeThis

import torch
from torch.utils.data import Dataset


class WindowGenerator(Dataset):
    "We watch the future of X but not y"

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
        addToX = self.Y[idx:idx + self.input_sequence_length] if \
            len(self.Y.shape) == 2 else self.Y[idx:idx + self.input_sequence_length][:, None]
        temp_ = torch.zeros((self.output_sequence_length, addToX.shape[1]))
        addToX = torch.concatenate((addToX, temp_), 0)
        if len(addToX) < self.input_sequence_length + self.output_sequence_length:
            addToX = torch.cat((addToX, torch.zeros(
                (self.input_sequence_length + self.output_sequence_length - len(addToX), addToX.shape[1]))))
        x = torch.cat(
            (self.X[idx:idx + self.input_sequence_length + self.output_sequence_length], addToX), 1)
        y = self.Y[
            idx + self.input_sequence_length:idx + self.input_sequence_length + self.output_sequence_length]
        return x, y

    def getInOutSize(self):
        x, y = self.__getitem__(0)
        in_size = x.shape[1] if len(x.shape) > 1 else 1
        out_size = y.shape[1] if len(y.shape) > 1 else 1
        return in_size, out_size

    def getWithoutFuture(self, idx):
        x = torch.cat(
            (self.X[idx:idx + self.input_sequence_length], self.Y[idx:idx + self.input_sequence_length][:, None]),
            1)
        y = self.Y[
            idx + self.input_sequence_length:idx + self.input_sequence_length + self.output_sequence_length]
        return x, y


class DataLoader():
    def __init__(self, options, customData=None):
        self.X = None
        self.Y = None

        self.train_dataset = None
        self.test_dataset = None
        self.full_dataset = None

        self.train_loader = None
        self.test_loader = None
        self.full_loader = None

        self.featuresNames = None
        self.options = options
        if customData is None:
            X, Y = get_all_data(hourly=options.hourly) if options.targetFeatures is None else \
                get_data_filtered_data(options.targetFeatures, hourly=options.hourly)
        else:
            X, Y = customData
        X.sort_index(ascending=True, inplace=True)
        Y.sort_index(ascending=True, inplace=True)

        self.dfX = X
        self.dfY = Y

        self.setData(X, Y)
        self.verif()

    def verif(self):
        if self.options.hourly:
            assert len(self.Y.shape) == 2, f"Output {len(self.Y.shape)} instead of 2"
            assert self.options.output_sequence_length == 1, f"output_sequence_length should be 1" \
                                                             f" instead of {self.options.output_sequence_length}"
        if self.options.recursif:
            assert not self.options.shuffle, "Shuffle should not be used with recursif on"
            assert self.options.output_sequence_length == 1, "output_sequence_length should be set to 1" \
                                                             f" instead of {self.options.output_sequence_length}"

        assert len(self.Y.shape) == 2 and self.Y.shape[1] == self.options.output_size, "output_size should be " \
                                                                                       f"{self.Y.shape[1]}, and not " \
                                                                                       f"{self.options.output_size}"
        assert len(self.X.shape) == 2 and self.X.shape[1] == self.options.input_size, "input_size should be " \
                                                                                      f"{self.X.shape[1]}, and not " \
                                                                                      f"{self.options.input_size}"

    def getData(self):
        return self.X, self.Y

    @timeThis("Data set in : ")
    def setData(self, X, Y):
        self.featuresNames = list(X.columns) + list(Y.columns) if len(Y.shape) == 2 else list(X.columns) + [
            'ventes passées']
        self.X = X.to_numpy(dtype='float64')
        self.Y = Y.to_numpy(dtype='float64')

        if self.options.fullTraining:
            self.full_dataset = WindowGenerator(self.X, self.Y, self.options.input_sequence_length,
                                                self.options.output_sequence_length)
            self.full_loader = torch.utils.data.DataLoader(self.full_dataset, batch_size=self.options.batch_size,
                                                           shuffle=self.options.shuffle)
        else:
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
        if not self.options.fullTraining:
            assert len(
                self.y_train) - window_size + 1 > 0, f"train dataset size too small for current options\n " \
                                                     f"Try changing split, window size or increase the dataset size (currently:{len(self.y_train)})"
            assert len(
                self.y_test) - window_size + 1 > 0, f"test dataset size too small for current options\n " \
                                                    f"Try changing split, window size or increase the dataset size (currently:{len(self.y_test)})"

    def getTrainDataLoader(self):
        return self.train_loader

    def getTestDataLoader(self):
        return self.test_loader

    def getData(self):
        return self.X, self.Y

    def getDF(self):
        return self.dfX, self.dfY

    def getXDimension(self):
        return self.options.getXDimension()

    def getFeatureNames(self):
        return self.featuresNames

    # TODO create a DB using training data but more importantly TEST DATA
