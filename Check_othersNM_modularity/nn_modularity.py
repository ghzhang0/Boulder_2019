#python nn_modularity.py --it_ind 3 --inputSize 100 --outputSize 100 --k_ind 10 --noise 0 --hiddenSize $anande_TASK_ID --learningRate 5 --epochs 400000
from comet_ml import Experiment

import torch
import argparse
import numpy as np
import copy
from numpy import genfromtxt
import difflib

device = torch.device('cpu')
#device = torch.device('cuda') # Uncomment this to run on GPU

parser = argparse.ArgumentParser()
parser.add_argument("--it_ind",type=int, required=True)
parser.add_argument("--epochs",type=int, required=True)
parser.add_argument("--inputSize", nargs='*', type=int, required=True)
parser.add_argument("--hiddenSize", nargs='*', type=float, required=True)
parser.add_argument("--learningRate", nargs='*', type=float, required=True)
parser.add_argument("--k_ind",type=int, required=True)
parser.add_argument("--noise",type=int, required=True)
parser.add_argument("--outputSize", nargs='*', type=int, required=False)
args = parser.parse_args()

# Define comet experiment
experiment = Experiment(api_key="NN0Px7Aq1Ms7bX1ozbMSU2R9e",
                        project_name="Bottleneck", workspace="anjalika-nande")

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
            k = args.k_ind
            noise = args.noise
            # Define input
            x_in = [[1 if i==j else 0 for i in range(N)] for j in range(N)]
            x = torch.tensor(x_in, device = device).float()
            # Define output
            def generate_out_matrix(dim, nresponding, nclust, nnoise):
                # dim - dimensions of matrix (assuming it's square for now)
                # nresponding - number of neurons to activate per behavior (sparsity)
                # nclust - number of clusters
                # nnoise - number of "imperfect" neurons outside of cluster
                out=np.zeros((dim,dim))
                length=list(range(dim))
                for i in range(nclust):
                    set1=list(range(int(i*dim/nclust), int((i+1)*dim/nclust)))      ### inside dense cluster
                    set2=length[:int(i*dim/nclust)]+length[int((i+1)*dim/nclust):]  ### outside dense cluster
                    for j in set1:
                        out[j,np.random.choice(set1, nresponding-nnoise, replace=False)]=1
                        out[j,np.random.choice(set2, nnoise, replace=False)]=1
                return out
            y_np=generate_out_matrix(dim=M, nresponding=k, nclust=5, nnoise=noise)
            y = torch.tensor(y_np, device = device).float()

            for index_n2 in range(len(args.hiddenSize)):
                R = int((args.hiddenSize[index_n2])) # hidden dimension
                model = torch.nn.Sequential(
                          torch.nn.Linear(N, R),
                          torch.nn.Sigmoid(),
                          torch.nn.Linear(R,M),
                          torch.nn.Sigmoid())
                loss_fn = torch.nn.MSELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


                list_loss = []

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

                    # Log to Comet.ml
                    experiment.log_metrics({"loss":loss.item()}, step=t)

                    # print results when max epoch is reached
                    if t == args.epochs - 1:
                        # Extract parameters
                        trained_parameters = []
                        for param in model.named_parameters():
                            trained_parameters.append(param[1].data.numpy())

                        w1 = trained_parameters[0]
                        b1 = trained_parameters[1]
                        w2 = trained_parameters[2]
                        b2 = trained_parameters[3]
                        np.savetxt('weights1_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2]) + '_' + str(k) + '_' + str(noise) + '_' + str(it_ind) + '.dat', w1)
                        np.savetxt('weights2_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2]) + '_' + str(k) + '_' + str(noise) + '_' + str(it_ind) + '.dat', w2)
                        np.savetxt('bias1_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2]) + '_' + str(k) + '_' + str(noise) + '_' + str(it_ind) + '.dat', b1)
                        np.savetxt('bias2_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2])  + '_' + str(k) + '_' + str(noise) + '_' + str(it_ind) + '.dat', b2)

                behavior = 0
                threshold = 0.5
                per = 0.99
                y_pred_binary = np.abs(np.round(y_pred.data.numpy()+0.5-threshold))
                for j in range(len(y)):
                    s = difflib.SequenceMatcher(None, y.data.numpy()[j],y_pred_binary[j])
                    if s.ratio() > per:
                        behavior += 1

                #list_y_pred = np.array(list_y_pred)
                list_loss = np.array(list_loss)

                #np.savetxt('y_pred_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2]) + '.dat', np.array(list_y_pred[-500:]).flatten().reshape(500*N, N))
                #np.savetxt('loss_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2]) + str(k) +'.dat', list_loss[-500:])
                np.savetxt('behavior_' + str(learning_rate) + '_' + str(args.hiddenSize[index_n2]) + '_' + str(k) + '_' + str(noise) + '_' + str(it_ind) + '.dat', [behavior])