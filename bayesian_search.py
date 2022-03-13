import optuna
from utils.dnn_methods import *
from utils.polar_pla import preprocess
from models import *
import math

from utils.dnn_methods import *
from utils.polar_pla import preprocess, preprocess_from_pickle
from models import *

''' This script should be used to test a configuration. Below are the datasets. 
        Fields which can be modified between tests are marked with an x.'''
                        #---------------------#   
                        # "DataSets/Name.csv" #  (This is what each of the fields mean below)
                        #  Median Fiter Size  #   
                        #    PLA Max Error    #   

datasets = [["DataSets/CTtemp.csv",10,6000],["DataSets/snp500.csv",10,5],["DataSets/power_hour.csv",4,0.5],["DataSets/chaotic_functions1.txt",0,0.00000001], ["DataSets/AirPassengers.csv",0,0.1], ["DataSets/JSE.csv",5,0]]

# dataset 1 gets about 100% accuracy with 500 epochs [10, 6000]

models_to_average = 1 # keep this constant across tests

        #--------- your test goes here, modifiable attributes are labelled with an x ---------#

# dataset and model type #
dataset = datasets[1]  # Change the index to test different datasets.                                       # x 
component = 2 # 0 to predict trend, 1 to predict duration, 2 for a dual approach (trend and duration)       # x

# hyperparameters #                                                                                         # x

hidden_size=32
lr=0.001
batch_size=64
seq_length=4
dropout=0.0
training_epochs=300
# TCN only ↓
kernel_size=2
n_layers=2

#inputs, outputs = preprocess_from_pickle("DataSets/jse.pkl", seq_length, component)


# This system uses walk forward validation. Define your k here.
k = 4
train_ratio = 4 # : 1 : 1 <- ratio of train to validate to test. 


# now just simply uncomment the model you'd like to test:

def create_DNN():                                                                                           # x
    #return MLP(seq_length*2, hidden_size, max(1,component), dropout).to(dev)
    #return CNN(seq_length, hidden_size, max(1,component), 2, dropout).to(dev)
    return TCN(seq_length,max(1, component), [hidden_size]*n_layers, kernel_size, dropout).to(dev)
    #return LSTM(seq_length, hidden_size, max(1,component), dropout).to(dev)
    #return RNN(max(1,component), 2, hidden_size, 1, dropout).to(dev)
    #return LSTM(max(1,component), 2, hidden_size, 1, dropout).to(dev)
    #return BiLSTM(max(1,component), 2, hidden_size, 1, dropout).to(dev)

outputfile = "" # if this is empty it will just print instead

print_output = True if outputfile == "" else False
if not print_output:
    outf = open(outputfile, 'a')
import math
import statistics
res_1 = []
res_2 = []
tp = 0 # true positives
tn = 0 # true negatives
fp = 0 # false positives
fn = 0 # false negatives
    
def objective(trial):
    
    
    n_layers = trial.suggest_int('n_layers', 1, 6)
    hidden_size = trial.suggest_int('hidden_size', 4, 256)
    lr=trial.suggest_float('lr', 0.000001, 0.01)
    batch_size=64
    seq_length=trial.suggest_int('seq_len', 2, 12)
    dropout=trial.suggest_float('dropout', 0, 0.8)
    training_epochs=500
    # TCN only ↓
    kernel_size=trial.suggest_int('kernel', 2, 8)
    inputs, outputs = preprocess(dataset[0], dataset[1], dataset[2], seq_length, component)
    def create_DNN():
    
        #return MLP(seq_len*2, hidden, max(1,component), drop).to(dev)
        #return CNN(seq_len, hidden, max(1,component), 2, drop).to(dev)
        return TCN(seq_length,max(1, component), [hidden_size]*n_layers, kernel_size, dropout).to(dev)
        #return LSTM(seq_len, hidden, max(1,component), drop).to(dev)
        #return RNN(max(1,component), 2, hidden, 1, drop).to(dev)
        #return LSTM(max(1,component), 2, hidden, 1, drop).to(dev)
        #return BiLSTM(max(1,component), 2, hidden, 1, drop).to(dev)
    result = train_and_test(create_DNN, inputs, outputs, lr, batch_size, seq_length, training_epochs, component, k, train_ratio) # train and test it

    if component == 2:
        return math.sqrt(result[0])+math.sqrt(result[1])

    else:
        return math.sqrt(result[0])

study = optuna.create_study()
study.optimize(objective, n_trials=200)

print(study.best_params)
os.remove("temp.pt")   