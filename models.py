import torch
import torch.nn as nn
from torch.autograd import Variable

#Chesney's models
if torch.cuda.is_available():  
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    dev = torch.device("cpu")

class RNN(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.init_weights()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, input):
        h_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        output, hidden = self.rnn(input, h_0.detach())
        out = output[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out


#Chesney's models
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_weights()
        
        self.LSTM = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(dev)
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(dev)
        output, (hidden, _) = self.LSTM(x, (h_0, c_0))
        out = output[:, -1, :]
        #out = hidden[-1]
        out = self.dropout(out)

        return self.fc(out)


class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.init_weights()
        
        self.LSTM = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size)
        c_0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size)
        output, (hidden, _) = self.LSTM(x, (h_0, c_0))
        out = output[:, -1, :]
        out = self.dropout(out)

        return self.fc(out)

# Morgan's models

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.do = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        x = self.do(x)
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class CNN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, kernel_size, dropout):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=kernel_size),
            nn.Flatten(),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(8*(input_size-2*kernel_size+2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )
        self.init_weights()
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        out = self.main(x)
        return out

from tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        #x = torch.swapaxes(x, 1, 2)
        y1 = torch.relu(self.tcn(x))
        
        return self.linear(y1[:, :, -1])

class TreNet(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_weights()
        
        self.LSTM = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
        )
        
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=2),
            nn.Flatten(),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(8*(seq_len-2*2+2), hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fusion = nn.Sequential(nn.Linear(hidden_size, num_classes),
                         nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        
        x_trends = torch.narrow(x, 1, 0, self.seq_len).to(dev)
        x_points = torch.narrow(x, 1, self.seq_len, self.seq_len).to(dev)
        x_points = torch.narrow(x_points, 2, 0, 1).to(dev)
        x_points = torch.swapaxes(x_points, 1, 2).to(dev)
        h_0 = torch.zeros(
            self.num_layers, x_trends.size(0), self.hidden_size).to(dev)
        c_0 = torch.zeros(
            self.num_layers, x_trends.size(0), self.hidden_size).to(dev)
        output, (hidden, _) = self.LSTM(x_trends, (h_0, c_0))
        out = output[:, -1, :]
        #out = hidden[-1]
        out = self.dropout(out)
        
        cnn_out = self.main(x_points)
        blah = torch.add(out, cnn_out)
        blah = self.fusion(blah)

        return blah