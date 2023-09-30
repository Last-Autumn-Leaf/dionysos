from sklearn.model_selection import RandomizedSearchCV

from .utils import XGBOOST_TYPE, RNN_TYPE, DEFAULT_OPTIONS, MAE, MSE, SMOOTH, ADAM, SGD, resultsDir
from .options import Options
from .rnn import RNN
from .xgboost import Xgboost

from .. import timeThis
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Model():
    modeltype2model = {
        RNN_TYPE: RNN,
        XGBOOST_TYPE: Xgboost
    }
    loss_functions = {
        MSE: torch.nn.MSELoss(),
        MAE: torch.nn.L1Loss(),
        SMOOTH: torch.nn.SmoothL1Loss()
    }
    showCurrentOptions = lambda self, currentOptions: print(f"\t", *[(k, v) for k, v in currentOptions.items()],
                                                            sep='\n\t')

    def __init__(self, options=None):
        self.model = None
        print("setting model with options:", end='')
        self.options = Options(DEFAULT_OPTIONS[RNN_TYPE]) if options is None else options
        self.showCurrentOptions(self.options.getModelOptions())
        self.showCurrentOptions(self.options.getDataOptions())
        self.setModel()

    def setModel(self):
        self.model = self.modeltype2model[self.options.model_type](**self.options.getModelOptions())

    @timeThis("Traning time: ")
    def train(self, dataLoader, test=True, plot=False, n=3, overfiting=False):
        if self.options.model_type == RNN_TYPE:
            self.trainRNN(dataLoader, plot)
        elif self.options.model_type == XGBOOST_TYPE:
            self.trainXGBoost(dataLoader, plot)
        if test:
            self.test(dataLoader, n, overfiting)

        saveModel(self)

    def test(self, dataLoader, n=3, overfitting=False):
        if self.options.model_type == RNN_TYPE:
            x_test, y_test, predicted_sequence = self.testRNN(dataLoader, overfitting)
        elif self.options.model_type == XGBOOST_TYPE:
            x_test, y_test, predicted_sequence = self.testXGBoost(dataLoader, overfitting)
        else:
            raise f"model_type non reconnu {self.options.model_type}"

        target_sequence = y_test.numpy()
        if len(predicted_sequence.shape) == 1:
            predicted_sequence = predicted_sequence[None, :]
        n_test = min(n, len(predicted_sequence))
        for i in range(n_test):
            history_sequence = x_test[i, :, -1].numpy()
            plt.plot(range(self.options.input_sequence_length + self.options.output_sequence_length),
                     np.concatenate((history_sequence, target_sequence[i])), label='True sequence',
                     linestyle="-", marker="o")
            plt.plot(range(self.options.input_sequence_length,
                           self.options.input_sequence_length + self.options.output_sequence_length),
                     predicted_sequence[i], linestyle="-", marker="o", label='Predicted sequence')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title('Sequence and Predicted Sequence')
            plt.legend()
            plt.show()

    def setTransform(self, dataLoader):
        X, Y = dataLoader.getData()
        self.transform_params = {}
        try:
            ncols = X.shape[1]
            one_hot = set()
            for i in range(ncols):
                maxV = X[:, i].max()
                minV = X[:, i].min()
                if maxV != 1:
                    # self.x[:,i]=(self.x[:,i]-minV)/maxV
                    self.transform_params[i] = (maxV, minV)
                else:
                    one_hot.add(i)
            print('tensors', one_hot, 'found as one-hot encoded')

        except:
            print('no scaling done')

    def transform(self, X, Y):
        for i, (maxV, minV) in self.transform_params.items():
            X[:, :, i] = (X[:, :, i] - minV) / maxV
        return X, Y / 10000

    def reverseTransform(self, X, Y=None):
        if Y is None:
            return X * 10000
        for i, (maxV, minV) in self.transform_params.items():
            X[:, :, i] = (X[:, :, i] * maxV + minV)
        return X, Y * 10000

    def trainRNN(self, dataLoader, plot=False):
        self.setTransform(dataLoader)
        trainLoader = dataLoader.getTrainDataLoader()
        testLoader = dataLoader.getTestDataLoader()

        model = self.model

        criterion = self.loss_functions.get(self.options.lossFunction)
        assert criterion is not None, "Unrecognized loss function"

        optimizer = None
        if self.options.optimizer == ADAM:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.options.learning_rate)
        elif self.options.optimizer == SGD:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.options.learning_rate,
                                        momentum=self.options.momentum, weight_decay=self.options.weight_decay)
        assert optimizer is not None, f"optimizer {self.options.optimizer} not found !"
        best_test_loss = float('inf')
        best_model_state = None
        best_test_loss_epoch = None

        train_losses = []
        test_losses = []
        counter = -1
        for epoch in range(self.options.epochs):
            model.train()
            train_loss = 0.0
            for x, y in trainLoader:
                optimizer.zero_grad()
                x, y = self.transform(x, y)
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(trainLoader)
            train_losses.append(train_loss)

            model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for x, y in testLoader:
                    x, y = self.transform(x, y)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    test_loss += loss.item()

                test_loss /= len(testLoader)
                test_losses.append(test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_loss_epoch = epoch
                best_model_state = model.state_dict()

            if self.options.verbose:
                counter = (counter + 1) % self.options.verbose_mod
                if not counter or epoch == self.options.epochs - 1:
                    print(
                        f"Epoch {epoch + 1}/{self.options.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Best Loss :{best_test_loss} at {best_test_loss_epoch}")
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss Evolution for {}'.format(self.options.model_type))
            plt.legend()
            plt.show()
        return model

    def preprocessDataXGBoost(self, dataLoader):
        n_train = len(dataLoader.train_dataset)
        n_test = len(dataLoader.test_dataset)
        X_train = torch.zeros((n_train, self.options.input_sequence_length * self.options.input_size))
        Y_train = torch.zeros((n_train, self.options.output_sequence_length))
        X_test = torch.zeros((n_test, self.options.input_sequence_length * self.options.input_size))
        Y_test = torch.zeros((n_test, self.options.output_sequence_length))

        for i in range(n_train):
            x, y = dataLoader.train_dataset[i]
            X_train[i] = x.t().flatten()
            Y_train[i] = y
        for i in range(n_test):
            x, y = dataLoader.test_dataset[i]
            X_test[i] = x.t().flatten()
            Y_test[i] = y
        return X_train, Y_train, X_test, Y_test

    def trainXGBoost(self, dataLoader, plot):

        X_train, Y_train, X_test, Y_test = self.preprocessDataXGBoost(dataLoader)

        model = self.model.model
        # Entraîner le modèle en affichant la courbe d'apprentissage
        model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)],
                  verbose=self.options.verbose and self.options.verbose_mod)

        # Afficher la courbe d'apprentissage
        if plot:
            results = model.evals_result()
            train_loss = results['validation_0'][self.options.eval_metric]
            val_loss = results['validation_1'][self.options.eval_metric]

            plt.figure(figsize=(15, 8))
            plt.plot(train_loss, label='Train')
            plt.plot(val_loss, label='Validation')
            plt.xlabel('Boosting Round')
            plt.ylabel(self.options.eval_metric)
            plt.title('XGBoost Loss Evolution')
            plt.legend()
            plt.show()

        return model

    def testRNN(self, dataLoader, overfiting=False):
        if len(self.transform_params) != 0:
            self.setTransform(dataLoader)

        model = self.model
        model.eval()
        with torch.no_grad():
            # Get a batch of data from the test loader
            x_test, y_test = next(
                iter(dataLoader.getTrainDataLoader() if overfiting else dataLoader.getTestDataLoader()))
            # Forward pass through the model
            x_test, y_test = self.transform(x_test, y_test)
            predicted_sequence = model(x_test)
            x_test, y_test = self.reverseTransform(x_test, y_test)
            predicted_sequence = self.reverseTransform(predicted_sequence)
            # Convert the predicted sequence and target sequence to numpy arrays
            predicted_sequence = predicted_sequence.squeeze().numpy()
        return x_test, y_test, predicted_sequence

    def testXGBoost(self, dataLoader, overfiting=False):

        model = self.model.model

        X_train, Y_train, X_test, Y_test = self.preprocessDataXGBoost(dataLoader)
        x_test, y_test = (X_train, Y_train) if overfiting else (X_test, Y_test)

        predicted_sequence = model.predict(x_test)

        x_test = x_test.reshape((16, self.options.input_sequence_length, self.options.input_size))
        return x_test, y_test, predicted_sequence

    def fineTuneXGBoostRandomSearch(self, dataLoader, param_distributions, n_iter=100, cv=3):
        X_train, Y_train, X_test, Y_test = self.preprocessDataXGBoost(dataLoader)
        eval_results, res = self.model.fineTune(X_train, Y_train, X_test, Y_test, param_distributions, n_iter, cv,
                                                self.options.verbose)
        plot_eval_result_XGBOOST(eval_results)
        show_test(X_test, Y_test, res, self.options.input_sequence_length, self.options.output_sequence_length)

        # TODO : handle the saving of the models
        saveModel(self)

    def fineTuneRnn(self, dataLoader, param_distributions=None, n_iter=10):
        X_train, Y_train, X_test, Y_test = self.preprocessDataXGBoost(dataLoader)
        eval_results, res = self.model.rayTune(X_train, Y_train, X_test, Y_test, param_distributions, n_iter)
        plot_eval_result_XGBOOST(eval_results)
        show_test(X_test, Y_test, res, self.options.input_sequence_length, self.options.output_sequence_length)

        # TODO : handle the saving of the models
        saveModel(self)


# --------- saving and loading models
def saveModel(obj, suffix=''):
    modelType = obj.options.model_type
    dir = resultsDir / modelType / "saves"
    if not dir.exists():
        dir.mkdir(parents=True)
    files_list = list(dir.glob('*'))
    filename = dir / f"{modelType}_{len(files_list)}{suffix if suffix else ''}.pkl"
    with open(filename, "wb") as file:
        pickle.dump(obj, file)
    print(f"file saved at {filename}")


def loadModel(filename):
    with open(filename, "rb") as file:
        obj = pickle.load(file)
    return obj


# ----------- Visualization
def plot_eval_result_XGBOOST(eval_results):
    # Extract training and validation losses
    train_loss = eval_results['validation_0']['rmse']
    val_loss = eval_results['validation_1']['rmse']

    # Plot the loss
    epochs = len(train_loss)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def show_test(X_test, Y_test, res, input_sequence_length, output_sequence_length, ntest=3):
    for i in range(ntest):
        history_sequence = X_test[i][-input_sequence_length:]
        target_sequence = Y_test[i]
        predicted_sequence = res[i]
        plt.plot(range(input_sequence_length + output_sequence_length),
                 np.concatenate((history_sequence, target_sequence)), label='True sequence',
                 linestyle="-", marker="o")
        plt.plot(range(input_sequence_length, input_sequence_length + output_sequence_length),
                 predicted_sequence, linestyle="-", marker="o", label='Predicted sequence')
        plt.legend()
        plt.title(f'Sequence and Predicted Sequence n°{i}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.show()
