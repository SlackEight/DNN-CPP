from numpy.core.shape_base import block
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import models
import os
import copy



if torch.cuda.is_available():  
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("using GPU")
else:
    dev = torch.device("cpu")

def unfold_inputs_and_outputs_MLP(inputs, outputs):
    # unfold the inputs and outputs so that slope and duration aren't in a list together (basically get rid of channels)
    inp = []
    for sequence in inputs:
        temp = []
        for p in sequence:
            temp.append(p[0])
            if p[1] != 0:
                temp.append(p[1])
        inp.append(temp)

    return torch.Tensor(np.array(inp)).to(dev), torch.Tensor(np.array(outputs)).to(dev).to(dev)

def tensor_dataset(inputs, outputs):
    return torch.Tensor(np.array(inputs)).to(dev), torch.Tensor(np.array(outputs)).to(dev).to(dev)

def dataload_walkforward(window_func ,batch_size, inputs, outputs, seq_len, component, k, index, train_ratio):
    # k fold walk forward validation with overlapping windows
    # after doing some maths it seems logical to divide the data into blocks. We'll use something like
    # 3 blocks of training data, 1 block of validating and 1 block of testing. The relationship between
    # number of folds and block size, given our ratio, is: block_size = setsize/(k+4)

    # get the inputs and outputs for this fold
    print("----- WF VALIDATION INFO -------")
    print()
    print()
    fold_inputs, fold_outputs = window_func(inputs, outputs)
    print("fold inputs len", len(fold_inputs))
    train_blocks = train_ratio # and 1 for validate and one for test. Just change this to change the ratio
    blocks_in_dataset = (train_blocks + 2) + (k-1)

    block_size = len(fold_inputs)//(blocks_in_dataset)
    print("block size", block_size)
    train_size = train_blocks * block_size
    train_index = index * block_size

    training_input = torch.narrow(fold_inputs, 0, train_index, train_size)
    print("training input len", len(training_input))
    training_output = torch.narrow(fold_outputs, 0, train_index, train_size)

    validation_input = torch.narrow(fold_inputs, 0, train_index+train_size, block_size).to(dev)
    print("validation input len", len(validation_input))
    validation_output = torch.narrow(fold_outputs, 0, train_index+train_size, block_size).to(dev)

    testing_input = torch.narrow(fold_inputs, 0, train_index+train_size+block_size, block_size).to(dev)
    print("testing input len", len(testing_input))
    print()
    testing_output = torch.narrow(fold_outputs, 0, train_index+train_size+block_size, block_size).to(dev)

    train = torch.utils.data.TensorDataset(training_input, training_output)
    validate = torch.utils.data.TensorDataset(validation_input, validation_output)
    test = torch.utils.data.TensorDataset(testing_input, testing_output)

    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    validateset = torch.utils.data.DataLoader(validate, batch_size=batch_size, shuffle=False)
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return trainset, validateset, testset

def test_model(model, trainset, validateset, testset, learning_rate, component, training_epochs, current_fold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    epochs = training_epochs
    import torch.optim as optim
    train_loss = [] # an array which will contain the train loss for each epoch, will be used to plot the loss by epoch
    validation_loss = [] # same thing except on validation set

    min_val_loss_epoch = 0 # the epoch with the lowest validation loss
    min_val_loss = 9999999 # the lowest validation loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    validation_direction_accuracy = []
    
    for epoch in range(epochs+1):
        tot = 0
        model.train() # set model to training mode (dropout is on)
        epoch_trainloss = 0 # the average loss for this epoch
        
        for data in trainset:  # for each batch in the train set
            features, labels = data  # split the batches up into their features and labels
            model.zero_grad()
            output = model(features) # get a prediction from the model
            tot += len(features)
            loss = F.mse_loss(output, labels)  # calculate the loss of our prediction
            loss.backward()  # backpropogate the loss
            optimizer.step()  # optimize weights
            epoch_trainloss += loss.item() # add the loss to the epoch loss - note that loss.item() will return the average loss of the batch.
        
        epoch_trainloss /= len(trainset) # average the loss for each batch to get the epoch's average. len(trainset) is the number of batches in the train set
        
        train_loss.append(epoch_trainloss) # add this epoch's average loss to the list of training losses
        
        # now we'll calculate the direction accuracy for the training and validation sets
        epoch_validationloss = 0 
        correct=0
        total_points = 0
        model.eval()
        for data in validateset:
            
            inputs, labels = data
            output = model(inputs)
            total_points += len(output)
            for i in range(len(output)):
                pred = output[i]
                actual = labels[i]
                if pred[0] < 0 and actual[0] < 0 or pred[0] > 0 and actual[0] > 0: #or (pred-actual)<0.01:
                    correct += 1
                #print(output[0],labels[0])
                #if epoch % 20 == 0 and i == 0:
                #    print("Prediction:",pred,"Actual:",actual)
            loss = F.mse_loss(output, labels)  # calculate the loss of our prediction
            epoch_validationloss += loss.item()/len(validateset)
        if epoch_validationloss < min_val_loss:
            if epoch != 0: torch.save(model, 'temp.pt')
            min_val_loss = epoch_validationloss
            min_val_loss_epoch = epoch
        validation_direction_accuracy.append(correct/(total_points))
        validation_loss.append(epoch_validationloss) # we'll need to plot validation loss too
        
        if epoch % 20 == 0:
            print(epoch,"/",epochs)
            print("Training loss:",epoch_trainloss,"Validation loss:",epoch_validationloss,"\n")
    #plt.figure(current_fold)
    #plt.plot(train_loss, label="Training loss for fold "+str(current_fold+1))
    #plt.plot(validation_loss, label="Validation loss for fold "+str(current_fold+1))
    #plt.legend()
    #plt.figure(current_fold+10)
    #plt.plot(validation_direction_accuracy, label = "Directional Accuracy for fold "+str(current_fold+1))
    #plt.legend()
    print(f"Lowest validation loss: {min_val_loss} at epoch {min_val_loss_epoch}")

    model = torch.load('temp.pt')

    model.eval()
    correct=0
    output_file = open("utils/angles.txt", "w")
    '''for data in trainset:

        inputs, labels = data
        output = model(inputs)
        for i in range(len(output)):
            pred = output[i]
            #output_file.write(str(pred.item()*90)+"\n")
    total_loss = 0
    for data in validateset:
        inputs, labels = data
        output = model(inputs)
        model.zero_grad()
        total_loss += F.mse_loss(output, labels).item()/len(validateset)
    #print(f'Directional Accuracy: {correct*100/len(test)} MSE on validate set: {total_loss}')
    '''
    total_loss = 0
    total_loss_slope = 0
    total_loss_length = 0
    tot = 0
    for data in testset:
        inputs, labels = data
        output = model(inputs)
        
        # test for the dual model
        if component == 2:
            output_slopes = []
            for out in labels:
                output_slopes.append(np.array([out[0].cpu().detach().numpy()]))
                tot += 1
            output_slopes = torch.Tensor(output_slopes).to(dev)

            output_lengths = []
            for out in labels:
                output_lengths.append(np.array([out[1].cpu().detach().numpy()]))
            output_lengths = torch.Tensor(output_lengths).to(dev)

            pred_slopes = []
            for out in output:
                pred_slopes.append(np.array([out[0].cpu().detach().numpy()]))
                #pred_slopes.append(np.array([5]))
            pred_slopes = torch.Tensor(pred_slopes).to(dev)

            pred_lengths = []
            for out in output:
                pred_lengths.append(np.array([out[1].cpu().detach().numpy()]))
                #print(out[1].cpu().detach().numpy(),"vs", 2, "vs", output_lengths[-1])
                #pred_lengths.append(np.array([2]))
            pred_lengths = torch.Tensor(pred_lengths).to(dev)
        
            for i in range(len(output_slopes)): # true and false directional classifications
                pred = pred_slopes[i][0]
                actual = labels[i][0]
                if pred > 0 and actual > 0 or 0<(abs(pred)-abs(actual))<0.022: # true positive with 2 degree lee way
                    tp += 1
                elif pred < 0 and actual < 0: # true negative
                    tn += 1
                elif pred > 0 and actual < 0: # false positive
                    fp += 1
                elif pred < 0 and actual > 0: # false negative
                    fn += 1

            model.zero_grad()
            total_loss_slope += F.mse_loss(output_slopes,pred_slopes).item()/len(testset)
            model.zero_grad()
            total_loss_length += F.mse_loss(output_lengths, pred_lengths).item()/len(testset)
        # test for single model
        else:
            model.zero_grad()
            total_loss += F.mse_loss(output, labels).item()#/len(testset)
            for i in range(len(output)): # directional accuracy check
                pred = output[i][0]
                actual = labels[i][0]
                if pred > 0 and actual > 0: # true positive with 2 degree lee way
                    tp += 1
                elif pred < 0 and actual < 0: # true negative
                    tn += 1
                elif pred > 0 and actual < 0: # false positive
                    fp += 1
                elif pred < 0 and actual > 0: # false negative
                    fn += 1
    #print("TEST SET LENGTH", tot)    
    import math
    #print(f'MSE test: {total_loss}, RMSE test: {math.sqrt(total_loss)}\n')         
    if component == 2:
        return [total_loss_slope, total_loss_length,tp,tn,fp,fn]
    else:
        return [total_loss,tp,tn,fp,fn]
        
        #print(f'{math.sqrt(total_loss_slope)}, {math.sqrt(total_loss_length)}')
        #testing_file.write(f'{math.sqrt(total_loss_slope)},{math.sqrt(total_loss_length)}\n')

    



def train_and_test(create_model, inputs, outputs, lr, batch_size, seq_length, training_epochs, component, k, train_ratio):
    
    model = create_model()
    s_window = unfold_inputs_and_outputs_MLP

    if isinstance(model, models.CNN) or isinstance(model, models.TCN):
        s_window = tensor_dataset
    elif isinstance(model, models.RNN) or isinstance(model, models.LSTM) or isinstance(model, models.BiLSTM):
        s_window = tensor_dataset
    print("------ Fold 1 of",k,"------")
    trainset, validationset, testset = dataload_walkforward(s_window ,batch_size, inputs, outputs, seq_length, component, k, 0, train_ratio)
    print("\nTrainset length (batches):", len(trainset))
    print("Validationset length (batches):", len(validationset))
    print("Testset length (batches):", len(testset),"\n")
    output = test_model(model, trainset, validationset, testset, lr, component, training_epochs//k, 0)

    output[0] /= k
    
    
    #if component == 2: output[1] /= k
    
    for i in range(1,k):
        print("\n------ Fold",i+1,"of",k,"------")
        trainset, validationset, testset = dataload_walkforward(s_window ,batch_size, inputs, outputs, seq_length, component, k, i, train_ratio)
        model = torch.load('temp.pt')
        res1 = test_model(model, trainset, validationset, testset, lr, component, training_epochs//k, i)
        if component == 2:
            output[0] += res1[0]/k
            output[1] += res1[1]/k
            for j in range(2,len(output)):
                output[j] += res1[j]
        else:
            output[0] += res1[0]/k
            for j in range(1,len(output)):
                output[j] += res1[j]
    #plt.show()
    return output
    # return test_model(model, trainset, validationset, testset, lr, component, training_epochs) for hold out