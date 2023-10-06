import torch
import torch.nn as nn
from sklearn.base import BaseEstimator

from .utils import GRU, LSTM, SimpleRNN


class RNN(nn.Module):
    model_types = {
        GRU: nn.GRU,
        LSTM: nn.LSTM,
        SimpleRNN: nn.RNN
    }

    def __init__(self, input_size=25, cell_type='LSTM', hidden_size=5, num_layers=2, output_size=1,
                 output_sequence_length=1, input_sequence_length=1, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_sequence_length = output_sequence_length
        self.input_sequence_length = input_sequence_length

        if cell_type in self.model_types:
            self.rnn_layer = self.model_types[cell_type](input_size=input_size, hidden_size=hidden_size,
                                                         dropout=dropout, num_layers=num_layers, batch_first=True)
        else:
            raise Exception('Model not implemented')

        self.fc1 = nn.Linear(hidden_size * input_sequence_length, output_sequence_length)
        self.dropout_layer = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
        self.double()

    def forward(self, x, h=None):
        bsize = x.size(0)
        x, h = self.rnn_layer(x, h)
        # TODO : fix this here
        out = self.fc1(x.reshape((bsize, self.hidden_size * self.input_sequence_length)))[:, :, None]
        return out[:, -self.output_sequence_length:, :] if self.output_size != 1 else out[:,
                                                                                      -self.output_sequence_length:, 0]

    def featureImportance(self, features_names, feature_threshold=0.001):
        print("Not implemented yet for the RNN")


if __name__ == '__main__':

    batchSize = 1
    input_size = 25
    output_size = 1
    input_sequence_length = 7
    output_sequence_length = 7

    rnn = RNN(
        input_size=input_size,
        output_size=output_size,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
    )
    # B,Tin,N
    # B batch size, Tin input sequence length,N numbers of input
    input = torch.ones((batchSize, input_sequence_length, input_size))
    output = rnn(input)
    # B batch size, Tout output sequence length
    # B batch size, Tout output sequence length,N numbers of output
    shape = output.shape
    assert 1 < len(shape) <= 3, f'Wrong output shape {shape}'

    assert shape[0] == batchSize, f'Error expected batch size {batchSize} got {shape[0]}'
    assert shape[1] == output_sequence_length, f'Error expected output sequence' \
                                               f' length {output_sequence_length} got {shape[1]}'
    if len(shape) == 3:
        assert shape[2] == output_size, f'Error expected output size  {output_size} got {shape[2]}'
    print('Everything works !')
