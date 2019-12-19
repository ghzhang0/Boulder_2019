#Code in file autograd/two_layer_net_autograd.py
#python nn.py --it_ind 0 --inputSize 100 --hiddenSize 100 --learningRate $GHZHANG17_TASK_ID --epochs 100000000
#grun python nn_sgd.py --it_ind 0 --inputSize 100 --outputSize 100 --hiddenSize 10 --learningRate 5 --epochs 100000
from comet_ml import Experiment

import torch
import copy
import argparse
import numpy as np
from numpy import genfromtxt
import difflib

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--it_ind",type=int, required=True)
parser.add_argument("--epochs",type=int, required=True)
parser.add_argument("--inputSize", nargs='*', type=int, required=True)
parser.add_argument("--hiddenSize", nargs='*', type=float, required=True)
parser.add_argument("--learningRate", nargs='*', type=float, required=True)
parser.add_argument("--outputSize", nargs='*', type=int, required=False)
args = parser.parse_args()

# Define comet experiment
experiment = Experiment(api_key="n8etP46kIWiChy4sbrEWSuqCG",
                        project_name="boulder", workspace="ghzhang0")

learning_rate = args.learningRate[0]
print(learning_rate)
it_ind = args.it_ind

# Define nonlinear activation function
def f(x): # nonlinear conversion function to binary
    return x.sigmoid()

# Track performance for comet
with experiment.train():
    for index_n1 in range(len(args.inputSize)):
        for index_n3 in range(len(args.outputSize)):
            N = args.inputSize[index_n1] # input dimension and number of training behaviors
            M = args.outputSize[index_n3] # output dimension
            x = torch.tensor(np.genfromtxt("x_{}.csv".format(it_ind), delimiter=','), device='cpu').float()
            y = torch.tensor(np.genfromtxt("y_{}.csv".format(it_ind), delimiter=','), device='cpu').float()
            for index_n2 in range(len(args.hiddenSize)):
                R = int((args.hiddenSize[index_n2])) # hidden dimension
                model = torch.nn.Sequential(
                          torch.nn.Linear(N, R),
                          torch.nn.Sigmoid(),
                          torch.nn.Linear(R,M),
                          torch.nn.Sigmoid())
                model.to(device)
                loss_fn = torch.nn.MSELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

                list_behavior = []
                list_loss = []
                
                x = x.to(device)
                y = y.to(device)
                #y_pred = torch.zeros([M, 1], dtype=torch.float64)
                #y_pred.to(device)
                for t in range(args.epochs):
                    # Run the forward pass
                    y_pred = model(x)

                    # Calculate the loss
                    loss = loss_fn(y_pred, y)
                    list_loss.append(loss.item())

                    # Zero the gradients before running the backward pass.
                    optimizer.zero_grad()

                    # Backward pass
                    loss.backward()

                    # Update weights using SGD
                    optimizer.step()
                        
                    #Criterion for stopping according to number of learnt behaviours
                    if t % 1000 == 0:
                        behavior = 0
                        threshold = 0.5
                        per = 0.99
                        y_pred1 = y_pred.to('cpu')
                        y1 = y.to('cpu')
                        y_pred_binary = np.abs(np.round(y_pred1.data.numpy()+0.5-threshold))
                        for j in range(len(y)):
                            s = difflib.SequenceMatcher(None, y1.data.numpy()[j],y_pred_binary[j])
                            if s.ratio() > per:
                                behavior += 1
                        list_behavior.append(behavior)
                        # Log to Comet.ml
                        experiment.log_metrics({"loss":loss.item(), "behavior":behavior})            
                        #experiment.log_metrics({"loss":loss.item(), "behavior":behavior}, step=t)            
                    if behavior == N:
                        # Extract parameters
                        trained_parameters = []
                        for param in model.named_parameters():
                            trained_parameters.append(param[1].to('cpu').data.numpy())

                        w1 = trained_parameters[0]
                        b1 = trained_parameters[1]
                        w2 = trained_parameters[2]
                        b2 = trained_parameters[3]
                        # Save parameters
                        np.savetxt('weights1_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  + '.dat', w1)
                        np.savetxt('weights2_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', w2)
                        np.savetxt('bias1_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', b1)
                        np.savetxt('bias2_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', b2)
                        break
                
                    # print results when max epoch is reached
                    if t == args.epochs - 1: 
                        # Extract parameters
                        trained_parameters = []
                        for param in model.named_parameters():
                            trained_parameters.append(param[1].to('cpu').data.numpy())

                        w1 = trained_parameters[0]
                        b1 = trained_parameters[1]
                        w2 = trained_parameters[2]
                        b2 = trained_parameters[3]
                        np.savetxt('weights1_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', w1)
                        np.savetxt('weights2_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', w2)
                        np.savetxt('bias1_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', b1)
                        np.savetxt('bias2_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', b2)

                #list_y_pred = np.array(list_y_pred)
                list_loss = np.array(list_loss)
                list_behavior = np.array(list_behavior)
                #np.savetxt('y_pred_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2]) + '.dat', np.array(list_y_pred[-500:]).flatten().reshape(500*N, N))
                np.savetxt('loss_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', list_loss[-500:])
                np.savetxt('behavior_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  +'.dat', list_behavior[-500:])
