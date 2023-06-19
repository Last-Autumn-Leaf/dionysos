'''
Cette classe permet de créer un modèle RNN et de l'entraîner sur les données

Date : 06/06/2023
@Auteur : Carl-André Gasette
'''

# Librairie
# Our lib
import os
import sys

from prevision.pre_processing.utils import timeThis

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from prevision.pre_processing.pre_process import pre_process

# Data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from prevision.pre_processing.constante import GRU, LSTM, SimpleRNN, SMOOTH, MSE, MAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log
from datetime import datetime
import logging as lg
lg.basicConfig(level=lg.INFO)

class WindowGenerator(Dataset):
    def __init__(self, X, Y, input_sequence_length, output_sequence_length):
        self.X = X
        self.Y = Y
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

    def __len__(self):
        return len(self.X) - self.input_sequence_length - self.output_sequence_length + 1

    def __getitem__(self, idx):
        # x = self.X[idx:idx + self.input_sequence_length]
        # x=torch.cat((self.X[idx:idx + self.input_sequence_length],self.Y[idx:idx + self.input_sequence_length][:,None]),1)
        x=self.X[idx:idx + self.input_sequence_length]
        y = self.Y[
            idx + self.input_sequence_length:idx + self.input_sequence_length + self.output_sequence_length]
        return x, y

    def getInOutSize(self):
        x,y=self.__getitem__(0)
        in_size=x.shape[1] if len(x.shape) > 1 else 1
        out_size=y.shape[1] if len(y.shape) > 1 else 1
        return in_size,out_size


class RNN(nn.Module):
    model_types = {
        GRU: nn.GRU,
        LSTM: nn.LSTM,
        SimpleRNN: nn.RNN
    }
    def __init__(self, input_size, model_type='LSTM', hidden_size=5, num_layers=1, output_size=1, output_sequence_length=1,dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_sequence_length = output_sequence_length

        if model_type in self.model_types :
            self.rnn_layer = self.model_types[model_type](input_size=input_size, hidden_size=hidden_size,
                                        dropout=dropout,num_layers=num_layers, batch_first=True)
        else:
            raise Exception('Model not implemented')

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        #  TODO : the problem is simple and severals non-linears layers should not be needed ! (delete this after test)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, x, h=None):
        x, h = self.rnn_layer(x, h)
        out = self.fc1(x)
        return out[:, -self.output_sequence_length:, :] if self.output_size != 1 else out[:, -self.output_sequence_length:, 0]

class modele_rnn():
    '''
    Cette classe permet de créer un modèle RNN et de l'entrainer sur les données
    '''
    loss_functions = {
        MSE: nn.MSELoss(),
        MAE: nn.L1Loss(),
        SMOOTH: nn.SmoothL1Loss()
    }

    # TODO : the Option class can be used for all model so it should not be declared here ! but in constante !!
    class Options:
        def __init__(self, input_sequence_length=10, output_sequence_length=5,model_type = 'LSTM',hidden_size=128, num_layers=2,
                     output_size=1, batch_size=64, epochs=50, learning_rate=0.01, dataset_split=0.7,optimizer='ADAM',
                     momentum=0.9, weight_decay=1e-5,verbose=True,verbose_mod=20,lossFunction='MSE'):
            self.input_sequence_length = input_sequence_length
            self.output_sequence_length = output_sequence_length
            self.hidden_size = hidden_size
            self.model_type = model_type
            self.num_layers = num_layers
            self.output_size = output_size
            self.batch_size = batch_size
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.dataset_split = dataset_split
            self.optimizer=optimizer
            self.momentum=momentum
            self.weight_decay=weight_decay
            self.verbose=verbose
            self.verbose_mod=verbose_mod
            self.lossFunction=lossFunction

        def getModelOptions(self):
            return {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_sequence_length': self.output_sequence_length,
                'model_type' : self.model_type
            }

    @timeThis("Initialisation finished in :")
    def __init__(self, x, y, options=None):
        self.options = options if options else self.Options()

        self.x = torch.tensor(x.to_numpy(dtype=float)).float()
        if len(self.x.shape)==1 : self.x=self.x[:,None]
        self.y = torch.tensor(y.to_numpy(dtype=float)).float()
        # self.x = torch.tensor(MinMaxScaler().fit_transform(self.x))
        self.custom_transform()
        self.split_dataset()

        train_dataset = WindowGenerator(self.x_train, self.y_train, self.options.input_sequence_length,
                                        self.options.output_sequence_length)
        test_dataset = WindowGenerator(self.x_test, self.y_test, self.options.input_sequence_length,
                                       self.options.output_sequence_length)

        self.input_size,self.output_size = train_dataset.getInOutSize()

        self.train_loader = DataLoader(train_dataset, batch_size=self.options.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.options.batch_size, shuffle=True)

    def custom_transform(self):
        # TODO : try using mean and std instead of max and min
        try :
            ncols=self.x.size(1)
            self.transform_params={}
            one_hot=set()
            for i in range (ncols):
                maxV=self.x[:,i].max()
                minV=self.x[:,i].min()
                if maxV !=1 :
                    self.x[:,i]=(self.x[:,i]-minV)/maxV
                else :
                    one_hot.add(i)
                self.transform_params[i]={'max':maxV,'min':minV}
            print('tensors', one_hot, 'found as one-hot encoded')
            self.y/=10000
        except  :
            print('no scaling done')


    def split_dataset(self):
        n = len(self.x)
        dataset_split = self.options.dataset_split
        self.x_train = self.x[:int(n * dataset_split)]
        self.y_train = self.y[:int(n * dataset_split)]
        self.x_test = self.x[int(n * dataset_split):]
        self.y_test = self.y[int(n * dataset_split):]

    @timeThis("Training finished in ")
    def train(self, plot=False):
        model = RNN(input_size=self.input_size,
                    output_size=self.output_size,
                    **self.options.getModelOptions())

        criterion = self.loss_functions.get(self.options.lossFunction)
        if criterion is None:
            print("Unrecognized loss function")

        #TODO : the ADAM string should be in a macro !
        if self.options.optimizer=='ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.options.learning_rate)
        else :
            optimizer = torch.optim.SGD(model.parameters(), lr=self.options.learning_rate,
                    momentum=self.options.momentum, weight_decay=self.options.weight_decay)

        best_test_loss = float('inf')
        best_model_state = None

        train_losses = []
        test_losses = []
        counter = -1
        for epoch in range(self.options.epochs):
            model.train()
            train_loss = 0.0
            for x, y in self.train_loader:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for x, y in self.test_loader:
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    test_loss += loss.item()

                test_loss /= len(self.test_loader)
                test_losses.append(test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict()

            if self.options.verbose:
                counter = (counter + 1) % self.options.verbose_mod
                if not counter or epoch == self.options.epochs - 1:
                    print(f"Epoch {epoch + 1}/{self.options.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss Evolution for {}'.format(self.options.model_type))
            plt.legend()
            plt.show()

        # TODO : put correct path & use options
        # torch.save(best_model_state, 'best_model.pkl')

        return model,best_model_state


    def test(self, model,best_model_state=None,n_test=1,overfiting=False):
        if best_model_state:
            model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            # Get a batch of data from the test loader
            x_test, y_test = next(iter(self.train_loader if overfiting else self.test_loader))

            # Forward pass through the model
            predicted_sequence = model(x_test)
            # Convert the predicted sequence and target sequence to numpy arrays
            predicted_sequence = predicted_sequence.squeeze().numpy()
            target_sequence = y_test.numpy()
            for i in range(n_test):
                history_sequence = x_test[i, :, -1].numpy()

                plt.plot(range(self.options.input_sequence_length + self.options.output_sequence_length),
                        np.concatenate((history_sequence, target_sequence[i])), label='True sequence',
                        linestyle="-", marker="o")

                plt.plot(range(self.options.input_sequence_length, self.options.input_sequence_length + self.options.output_sequence_length),
                        predicted_sequence[i], linestyle="-", marker="o", label='Predicted sequence')

                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.title('Sequence and Predicted Sequence')
                plt.legend()
                plt.show()


    # @staticmethod
    # def hyperparametres_random_search(X_train, X_test, y_train, y_test, input_sequence_length, output_sequence_length, output_size, batch_size, epochs, learning_rate, hidden_size_range, num_layers_range, param_grid, num_iterations=20):
    #     # Adapter les données pour le modèle
    #     X_train, X_test, y_train, y_test = modele_rnn.get_data(X_train, X_test, y_train, y_test)
    #
    #     # Créer les datasets et les dataloaders
    #     train_dataset = WindowGenerator(X_train, y_train, input_sequence_length, output_sequence_length)
    #     test_dataset = WindowGenerator(X_test, y_test, input_sequence_length, output_sequence_length)
    #
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #     # Définir les paramètres de la recherche aléatoire
    #     best_params = {}
    #     best_model = None
    #     best_rmse = float('inf')
    #
    #     # Generate all possible combinations of hyperparameters
    #     hyperparameter_combinations = list(itertools.product(hidden_size_range, num_layers_range, *param_grid.values()))
    #     #
    #     for _ in range(num_iterations):
    #         # Randomly select hyperparameter combination
    #         hyperparameters = random.choice(hyperparameter_combinations)
    #         hidden_size, num_layers = hyperparameters[:2]
    #         other_hyperparameters = hyperparameters[2:]
    #
    #         model = RNN(X_train.shape[1], hidden_size, num_layers, output_size, output_sequence_length).to(device)
    #         model.double()
    #
    #         criterion = nn.MSELoss()
    #         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #
    #         for epoch in range(epochs):
    #             model.train()
    #             train_loss = 0.0
    #
    #             for x, y in train_loader:
    #                 x = x.to(device).double()
    #                 y = y.to(device).double()
    #
    #                 optimizer.zero_grad()
    #                 outputs = model(x)
    #                 loss = criterion(outputs, y)
    #                 loss.backward()
    #                 optimizer.step()
    #
    #                 train_loss += loss.item()
    #
    #             train_loss /= len(train_loader)
    #
    #             model.eval()
    #             test_loss = 0.0
    #
    #             with torch.no_grad():
    #                 for x, y in test_loader:
    #                     x = x.to(device).double()
    #                     y = y.to(device).double()
    #                     outputs = model(x)
    #                     loss = criterion(outputs, y)
    #                     test_loss += loss.item()
    #
    #                 test_loss /= len(test_loader)
    #
    #             if test_loss < best_rmse:
    #                 best_rmse = test_loss
    #                 best_params = {'hidden_size': hidden_size, 'num_layers': num_layers, **dict(zip(param_grid.keys(), other_hyperparameters))}
    #                 best_model = model
    #
    #     print("Meilleurs hyperparamètres :", best_params)
    #     print("RMSE sur les données de test :", np.sqrt(best_rmse))
    #
    #     with open('prevision/model/rnn.pkl', 'wb') as f:
    #         pickle.dump(best_model, f)
    #
    #     return best_params, best_model
    #
    # @staticmethod
    # def load_model(path):
    #     # Charger le modèle à partir du fichier pickle
    #     with open(path, 'rb') as f:
    #         model = pickle.load(f)
    #     return model



def fixSeed(seed=1999):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

fixSeed()
if __name__ == '__main__':

    parameters = {
        "input_sequence_length": 14,
        "output_sequence_length": 7,
        "model_type" : LSTM,
        "hidden_size": 32,
        "num_layers": 4,
        "output_size": 1,
        "batch_size": 32,
        "epochs": 1000,
        "learning_rate": 0.1,
        'dataset_split': 0.7,
        'optimizer':'SGD',
        'momentum' :0.9,
        'weight_decay':1e-5,
        'lossFunction':MSE,
        'verbose_mod':100
    }
    # TODO : try with different options but we should fix the input and output length !
    # TODO : the network might need to be deeper when we will have more values, right now the default Options value can be considered overkill
    options = modele_rnn.Options(**parameters)

    X, y = pre_process.get_data()

    rnn_model = modele_rnn(X, y, options=options)

    # Entraînement du modèle
    model,best_model_state = rnn_model.train(plot=True)
    rnn_model.test(model,best_model_state ,3)
