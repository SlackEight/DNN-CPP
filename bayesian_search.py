from utils.dnn_methods import *
from utils.polar_pla import preprocess, preprocess_from_pickle
from models import *
import optuna

''' This script should be used to test a configuration. Below are the datasets. 
        Fields which can be modified between tests are marked with an x.'''
                        #---------------------#   
                        # "DataSets/Name.csv" #  (This is what each of the fields mean below)
                        #  Median Fiter Size  #   
                        #    PLA Max Error    #   

datasets = [["DataSets/CTtemp.csv",10,6000],["DataSets/snp500.csv",10,10],["DataSets/power_hour.csv",4,0.5],["DataSets/chaotic_functions1.txt",0,0.00000005], ["DataSets/AirPassengers.csv",0,0.1], ["DataSets/JSE.csv",5,0]]

# dataset 1 gets about 100% accuracy with 500 epochs [10, 6000]

models_to_average = 1 # keep this constant across tests

        #--------- your test goes here, modifiable attributes are labelled with an x ---------#

# dataset and model type #
dataset = datasets[1]  # Change the index to test different datasets.                                       # x 
component = 2 # 0 to predict trend, 1 to predict duration, 2 for a dual approach (trend and duration)       # x

# hyperparameters #                                                                                         # x

#inputs, outputs = preprocess_from_pickle("DataSets/jse.pkl", seq_length, component)



# now just simply uncomment the model you'd like to test:

# This system uses walk forward validation. Define your k here.
k = 4
train_ratio = 4 # : 1 : 1 <- ratio of train to validate to test. 

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
    
    
    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    lr=trial.suggest_float('lr', 0.0001, 0.05)
    seq_length=trial.suggest_int('seq_len', 6, 10)
    dropout=trial.suggest_float('dropout', 0, 0.5)
    training_epochs=400
    # TCN only ↓
    kernel_size=trial.suggest_int('kernel', 1, 4)
    
    seq_length = max(seq_length, kernel_size*2-1)
    inputs, outputs = preprocess(dataset[0], dataset[1], dataset[2], seq_length, component, trenet=False)
    
    print("KERNEL SIZE", kernel_size)
    print("SEQ LENGTH", seq_length)
    
    def create_DNN():                                                                                           # x
        #return MLP(seq_length*2, hidden_size, max(1,component), dropout).to(dev)
        #return CNN(seq_length, hidden_size, max(1,component), kernel_size, dropout).to(dev)
        return TCN(seq_length,max(1, component), [hidden_size]*n_layers, kernel_size, dropout).to(dev)
        #return RNN(max(1,component), 2, hidden_size, 1, dropout).to(dev)
        #return LSTM(max(1,component), 2, hidden_size, 1, dropout).to(dev)
        #return BiLSTM(max(1,component), 2, hidden_size, 1, dropout).to(dev)
        #return TreNet(max(1,component), 2, hidden_size, 1, dropout, seq_length).to(dev)
    return train_and_test(create_DNN, inputs, outputs, lr, 64, seq_length, training_epochs, component, k, train_ratio, validation=True)[0] # train and test it

import time
start_Time = time.time()
study = optuna.create_study()
study.optimize(objective, n_trials=50)
output = open('optuna_results.txt', 'w')
output.write(str(study.best_params))
os.remove("temp.pt")
print("Time taken: ", time.time() - start_Time)