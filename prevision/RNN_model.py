import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from prevision.constants import affluencePath, dataVentePath, meteoPath
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,output_sequence_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.outlSeqLen=output_sequence_length
    def forward(self, x):
        bsize=x.size(0)
        h0 = torch.zeros(self.num_layers, bsize,self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -self.outlSeqLen:, :])  # Select the last 5 time steps and pass through the linear layer
        return out.view(bsize, self.outlSeqLen)


class WindowGenerator (Dataset):
    def __init__(self, X, Y, input_sequence_length, output_sequence_length):
        self.X = X
        self.Y = Y
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

    def __len__(self):
        return len(self.X) - self.input_sequence_length - self.output_sequence_length + 1

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.input_sequence_length]
        y = self.Y[idx + self.input_sequence_length:idx + self.input_sequence_length + self.output_sequence_length]
        return x, y




def getData():
    selected_attendance=['date', 'phq_attendance_stats_sum']
    attendanceDf = pd.read_csv(affluencePath)[selected_attendance].rename(
        columns={'phq_attendance_stats_sum': 'attendance'})
    prevSellsDf = pd.read_csv(dataVentePath, sep=';')
    meteoDf = pd.read_csv(meteoPath).iloc[3:].rename(columns={'time': 'date'})

    meteoDf['date'] = pd.to_datetime(meteoDf['date'], format='%Y-%m-%d')
    attendanceDf['date'] = pd.to_datetime(attendanceDf['date'], format='%Y-%m-%d')
    prevSellsDf['date'] = pd.to_datetime(prevSellsDf['date'], format='%d-%m-%Y')

    df = pd.merge(pd.merge(attendanceDf, prevSellsDf, on='date'), meteoDf, on='date')
    df['day'] = df['date'].apply(lambda x: x.weekday())
    df = pd.get_dummies(df, columns=['day'])
    X = df.drop(['vente', 'date'], axis=1).to_numpy(dtype='float64')
    Y = df['vente'].to_numpy(dtype='float64')
    return torch.tensor(X, dtype=torch.float32).to(device),torch.tensor(Y, dtype=torch.float32).to(device)

if __name__=="__main__":

    X,Y = getData()
    input_sequence_length = 10  # Length of input sequence
    output_sequence_length = 5  # Length of output sequence
    batch_size = 8  # Number of samples in each batch
    num_epochs = 1

    dataset = WindowGenerator(X, Y, input_sequence_length, output_sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define the hyperparameters
    input_size = X.shape[1]
    hidden_size =10
    num_layers = 3
    output_size = 1

    model = RNN(input_size, hidden_size, num_layers, output_size,output_sequence_length).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adjust the learning rate as needed

    # Train the model
    model.train()
    losses = []
    val_losses = []

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # inputs shape: (batch_size, sequence_length)
            # targets shape: (batch_size,)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}")

            # # Validate the model
            # model.eval()
            # with torch.no_grad():
            #     val_outputs = model(X_test)
            #     val_loss = criterion(val_outputs, y_test)
            # val_losses.append(val_loss.item())
            #
            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss.item()}")

    # Plot the training and validation loss curves
    plt.plot(losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # # Convert the test data to a PyTorch tensor and move it to the device
    # X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    #
    # # Make predictions
    # model.eval()
    # with torch.no_grad():
    #     predictions = model(X_test).cpu().numpy()
    #
    # # Evaluate the model
    # mse = mean_squared_error(y_test, predictions)
    # mae = mean_absolute_error(y_test, predictions)
    #
    # print("mse: ",mse)
    # print("mae: ",mae)