import optuna
from utils.dnn_methods import *
from utils.polar_pla import preprocess
from models import *
import math

''' This script should be used to test a configuration. Below are the datasets. 
        Fields which can be modified between tests are marked with an x.'''
    #----------1----------# #---------2---------#  #--------3--------#
    # "DataSets/CTtemp.csv" "DataSets/snp500.csv" "DataSets/hpc.csv" 
    #          5                      10                   40
    #         6000                    10                  5000

datasets = [["DataSets/CTtemp.csv",5,6000],["DataSets/snp500.csv",10,10],["DataSets/hpc.csv",40,5000]]

train_proportion = 0.7 # keep this constant across tests
models_to_average = 10 # keep this constant across tests

        #--------- your test goes here, modifiable attributes are labelled with an x ---------#

# dataset and model type #
dataset = datasets[1]  # Change the index to test different datasets.                                       # x 
component = 2  # 0 to predict trend, 1 to predict duration, 2 for a dual approach (trend and duration)      # x

# hyperparameters #                                                                                         # x

# now just simply uncomment the model you'd like to test:


inputs, outputs = preprocess(dataset[0], dataset[1], dataset[2], 8, component)
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 6)
    hidden_size = trial.suggest_int('hidden_size', 4, 256)
    lr=trial.suggest_float('lr', 0.0001, 0.1)
    batch_size=64
    #seq_length=8#trial.suggest_int('seq_len', 4, 12)
    
    print("LEN INPUTS AHHHAHAHAHAHHA",len(inputs))
    dropout=trial.suggest_float('dropout', 0, 0.5)
    training_epochs=400
    # TCN only ↓
    kernel_size=trial.suggest_int('kernel', 2, 8)
    #stuff = [n_layers, hidden_size, seq_length, dropout, kernel_size]
    #stuff = [seq_length, hidden_size, dropout, kernel_size, n_layers]

    def create_DNN():
        
        #return MLP(seq_len*2, hidden, max(1,component), drop).to(dev)
        #return CNN(seq_len, hidden, max(1,component), 2, drop).to(dev)
        return TCN(8,max(1, component), [hidden_size]*n_layers, kernel_size, dropout).to(dev)
        #return LSTM(seq_len, hidden, max(1,component), drop).to(dev)
        #return RNN(max(1,component), 2, hidden, 1, drop).to(dev)
        #return LSTM(max(1,component), 2, hidden, 1, drop).to(dev)
        #return BiLSTM(max(1,component), 2, hidden, 1, drop).to(dev)

    #model = create_DNN(seq_length, hidden_size, dropout, kernel_size, n_layers) # create a fresh model
    result = train_and_test(create_DNN, inputs, outputs, lr, batch_size, 8, training_epochs, component) # train and test it

    if component == 2:
        return math.sqrt(result[0])+math.sqrt(result[1])

    else:
        return math.sqrt(result[0])

study = optuna.create_study()
study.optimize(objective, n_trials=200)

print(study.best_params)

os.remove("temp.pt")   