from utils.dnn_methods import *
from utils.polar_pla import preprocess, preprocess_from_pickle
from models import *

# {'n_layers': 4, 'hidden_size': 33, 'lr': 0.03234797397151763, 'seq_len': 7, 'dropout': 0.17873389879205104, 'kernel': 3}

''' This script should be used to test a configuration. Below are the datasets. 
        Fields which can be modified between tests are marked with an x.'''
                        #---------------------#   
                        # "DataSets/Name.csv" #  (This is what each of the fields mean below)
                        #  Median Fiter Size  #   
                        #    PLA Max Error    #   

datasets = [["DataSets/CTtemp.csv",10,6000],["DataSets/snp500.csv",10,10],["DataSets/power_hour.csv",4,0.5],["DataSets/chaotic_functions1.txt",0,0.00000005], ["DataSets/AirPassengers.csv",0,0.1], ["DataSets/JSE.csv",5,0], ["DataSets/N225P.csv",10,10]]

# dataset 1 gets about 100% accuracy with 500 epochs [10, 6000]

models_to_average = 1 # keep this constant across tests

        #--------- your test goes here, modifiable attributes are labelled with an x ---------#

# dataset and model type #
dataset = datasets[1]  # Change the index to test different datasets.                                       # x 
component = 2 # 0 to predict trend, 1 to predict duration, 2 for a dual approach (trend and duration)       # x

# hyperparameters #                                                                                         # x

hidden_size = 83
lr=0.008181482493866717
batch_size=64
seq_length=6
dropout=0.09106628173616016
training_epochs=200
# TCN only ↓
kernel_size=2
n_layers=4

#inputs, outputs = preprocess_from_pickle("DataSets/jse.pkl", seq_length, component)


{'hidden_size': 83, 'lr': 0.008181482493866717, 'seq_len': 6, 'dropout': 0.09106628173616016, 'kernel': 2}
# now just simply uncomment the model you'd like to test:

def create_DNN():                                                                                           # x
    #return MLP(seq_length*2, hidden_size, max(1,component), dropout).to(dev)
    return CNN(seq_length, hidden_size, max(1,component), kernel_size, dropout).to(dev)
    #return TCN(seq_length,max(1, component), [hidden_size]*n_layers, kernel_size, dropout).to(dev)
    #return RNN(max(1,component), 2, hidden_size, 1, dropout).to(dev)
    #return LSTM(max(1,component), 2, hidden_size, 1, dropout).to(dev)
    #return BiLSTM(max(1,component), 2, hidden_size, 1, dropout).to(dev)
    #return TreNet(max(1,component), 2, hidden_size, 1, dropout, seq_length).to(dev)

inputs, outputs = preprocess(dataset[0], dataset[1], dataset[2], seq_length, component, trenet=False)
print(len(inputs))
print("inputsssss",inputs[:10])
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
res_3 = []
tp = 0 # true positives
tn = 0 # true negatives
fp = 0 # false positives
fn = 0 # false negatives

latex = f'& 26.067 \pm\ 1.361 & \\textbf{4.005} \pm\ \\textbf{0.451} & \\textbf{15.036} \pm\ \\textbf{1.434}  \\ \hline TCN & \\textbf{21.747} \pm\ \\textbf{1.487} & 7.881 \pm\ 3.318 & 14.814 \pm 3.636  \\ \hline'

for x in range(models_to_average):
    model = create_DNN() # create a fresh model
    result = train_and_test(create_DNN, inputs, outputs, lr, batch_size, seq_length, training_epochs, component, k, train_ratio) # train and test it

    if component == 2:
        res_1.append(math.sqrt(result[0]))
        res_2.append(math.sqrt(result[1]))
        res_3.append((res_1[-1]+res_2[-1])/2)
        if print_output: print(f'{math.sqrt(result[0])}, {math.sqrt(result[1])}')
        classification = result[2::]
    else:
        res_1.append(math.sqrt(result[0]))
        if print_output: print(f'{math.sqrt(result[0])}')
        classification = result[1::]
    tp += classification[0]
    tn += classification[1]
    fp += classification[2]
    fn += classification[3]
    try:
        os.remove("temp.pt")
    except:
        pass

if component == 0 or component == 2:
    if print_output: print("slope results:") 
    else: outf.write("slope results:\n")
    
    if print_output: print(f'μ = {round(sum(res_1) / len(res_1 ),3)} | σ = {round(statistics.pstdev(res_1),3)} | tp = {tp} | tn = {tn} | fp = {fp} | fn = {fn}')
    else: outf.write(f'av = {round(sum(res_1) / len(res_1 ),3)} | dev = {round(statistics.pstdev(res_1),3)} | tp = {tp} | tn = {tn} | fp = {fp} | fn = {fn}\n')
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f1 = 2*sensitivity*specificity/(sensitivity+specificity)
    if print_output: print(f'accuracy = {round((tp+tn)/(fp+fn+tp+tn),3)} | sensitivity = {round(sensitivity,3)} | specificity = {round(specificity,3)} | F1 = {round(f1,3)}')
    else: outf.write(f'accuracy = {(tp+tn)/(fp+fn+tp+tn)} | sensitivity = {sensitivity} | specificity = {specificity} | F1 = {f1}\n')

if component == 1:
    if print_output: print(f'μ = {round(sum(res_1) / len(res_1 ),3)} | σ = {round(statistics.pstdev(res_1),3)}')
    else: 
        outf.write("length results:\n")
        outf.write(f'av = {round(sum(res_1) / len(res_1 ),3)} | dev = {round(statistics.pstdev(res_1),3)}\n')

if component == 2:
    if print_output: 
        print(f'μ = {round(sum(res_2) / len(res_2 ),3)} | σ = {round(statistics.pstdev(res_2),3)}')
        print("average:")
        print(f'μ = {round(sum(res_3) / len(res_3 ),3)} | σ = {round(statistics.pstdev(res_3),3)}')
    else:
        outf.write("length results:\n")
        outf.write(f'av = {round(sum(res_2) / len(res_2 ),3)} | dev = {round(statistics.pstdev(res_2),3)}\n')
        
    print(f"Naive & {round(sum(res_2) / len(res_2 ),3)} & {round(sum(res_1) / len(res_1 ),3)} & {round(sum(res_3) / len(res_3 ),3)} \\\\ \hline")
if not print_output:
    outf.write("\n")
    outf.close()



