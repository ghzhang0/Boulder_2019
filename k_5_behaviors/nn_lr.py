# code in file autograd/two_layer_net_autograd.py
# python nn_lr.py --inputSize 100 --hiddenSize 10 --learningRate 1 5 10 20 30 40 50 60 --epochs 40000 --trials 10 -activity 0 1
from comet_ml import Experiment

import torch
import copy
import argparse
import numpy as np
import difflib
from numpy import genfromtxt

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",type=int, required=True)
parser.add_argument("--trials",type=int, required=True)
parser.add_argument("--inputSize", nargs='*', type=int, required=True)
parser.add_argument("--learningRate", nargs='*', type=int, required=True)
parser.add_argument("--hiddenSize", nargs='*', type=int, required=True)
parser.add_argument("--activity", nargs='*', type=int, required=True)
args = parser.parse_args()

# Define comet experiment
experiment = Experiment("S1gz8UBL6fpAmuPjMIP4zGXMt", project_name="boulder")

# Define nonlinear activation function
def f(x): # nonlinear conversion function to binary
    return x.sigmoid()
# Track performance for comet

with experiment.train():
    for index_n3 in range(len(args.learningRate)):
        for index_n1 in range(len(args.inputSize)):
            for index_k in range(len(args.activity)):
                #lrs_range = lrs_all[index_n1]
                N = args.inputSize[index_n1] # input dimension and number of training behaviors
                M = N # output dimension
                K = args.activity[index_k]
                x = torch.tensor(np.genfromtxt("x_{}.csv".format(K), delimiter=','), device=device).float()
                y = torch.tensor(np.genfromtxt("y_{}.csv".format(K), delimiter=','), device=device).float()

                for index_n2 in range(len(args.hiddenSize)):
                    behaviours_learnt = []
                    file1=open('lrbehave_' + str(args.inputSize[index_n1]) + '_' + str(args.hiddenSize[index_n2]) +'.dat','ab')
                    for samples in range(args.trials):
                        #learning_rate = lrs_all[index_n1][index_n2]
                        learning_rate = args.learningRate[index_n3]
                        #R = int(N**(args.hiddenSize[index_n2])) # hidden dimension
                        R = args.hiddenSize[index_n2] # hidden dimension

                        # Create random Tensors for weights; setting requires_grad=True means that we
                        # want to compute gradients for these Tensors during the backward pass.
                        w1 = torch.randn(N, R, device=device, requires_grad=True).float()
                        w2 = torch.randn(R, M, device=device, requires_grad=True).float()

                        ## initialize tensor variables for bias terms
                        b1 = torch.randn(1, R, device=device, requires_grad=True).float() # bias for hidden layer
                        b2 = torch.randn(1, M, device=device, requires_grad=True).float() # bias for output layer


                        list_y_pred = []
                        list_loss = []
                        # add loss counter
                        k = 0

                        for t in range(args.epochs):
                            # Random reshuffle
                            ##sh = torch.randperm(x.size(0))
                            ##x = x.index_select(0,sh)
                            ##y = y.index_select(0,sh)

                            # Forward pass: compute predicted y using operations on Tensors. Since w1 and
                            # w2 have requires_grad=True, operations involving these Tensors will cause
                            # PyTorch to build a computational graph, allowing automatic computation of
                            # gradients. Since we are no longer implementing the backward pass by hand we
                            # don't need to keep references to intermediate values.
                            y_pred = f(f(x.mm(w1).add(b1)).mm(w2).add(b2))
                            y_prednp = y_pred.detach().numpy()
                            y_np = y.detach().numpy()
                            list_y_pred.append(y_prednp)
                            # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
                            # is a Python number giving its value.
                            loss = (y_pred - y).pow(2).mean()
                            list_loss.append(loss.item())


                            # Use autograd to compute the backward pass. This call will compute the
                            # gradient of loss with respect to all Tensors with requires_grad=True.
                            # After this call w1.grad and w2.grad will be Tensors holding the gradient
                            # of the loss with respect to w1 and w2 respectively.
                            loss.backward()

                            # Update weights using gradient descent. For this step we just want to mutate
                            # the values of w1 and w2 in-place; we don't want to build up a computational
                            # graph for the update steps, so we use the torch.no_grad() context manager
                            # to prevent PyTorch from building a computational graph for the updates
                            with torch.no_grad():
                                w1 -= learning_rate * w1.grad
                                w2 -= learning_rate * w2.grad
                                b1 -= learning_rate * b1.grad
                                b2 -= learning_rate * b2.grad

                            	# Manually zero the gradients after running the backward pass
                                w1.grad.zero_()
                                w2.grad.zero_()
                                b1.grad.zero_()
                                b2.grad.zero_()

                            # print results anyway if max epoch is reached
#                            if t == args.epochs - 1:
#                                np.savetxt('weights1_' + str(args.inputSize[index_n1]) + '_' + str(args.hiddenSize[index_n2])+ '_' + str(args.learningRate[index_n3]) +'.dat', w1.detach().numpy())
#                                np.savetxt('weights2_' + str(args.inputSize[index_n1]) + '_' + str(args.hiddenSize[index_n2])+ '_' + str(args.learningRate[index_n3]) +'.dat', w2.detach().numpy())
#                                np.savetxt('bias1_' + str(args.inputSize[index_n1]) + '_' + str(args.hiddenSize[index_n2])+ '_' + str(args.learningRate[index_n3]) +'.dat', b1.detach().numpy())
#                                np.savetxt('bias2_' + str(args.inputSize[index_n1]) + '_' + str(args.hiddenSize[index_n2])+ '_' + str(args.learningRate[index_n3]) +'.dat', b2.detach().numpy())

                        #Computing number of learnt behaviour
                        per = 0.99
                        threshold = 0.5
                        behaviour = 0
                        y_pred_binary = np.abs(np.round(y_prednp+0.5-threshold))
                        for j in range(len(y_prednp)):
                            s = difflib.SequenceMatcher(None, y_np[j],y_pred_binary[j])
                            if s.ratio() > per:
                                behaviour += 1
                        behaviours_learnt.append(behaviour/N)
                    file1_k = [args.activity[index_k]]
                    file1_x = [args.learningRate[index_n3]]
                    file1_y = [np.mean(behaviours_learnt)]
                    file1_z = [np.std(behaviours_learnt)]
                    np.savetxt(file1, np.transpose([file1_k, file1_x, file1_y, file1_z]), fmt='%.2f')
                    #list_loss = np.array(list_loss)
                    #np.savetxt('loss_' + str(args.inputSize[index_n1]) + '_' + str(args.hiddenSize[index_n2])+ '_' + str(args.learningRate[index_n3]) +'.dat', list_loss[-500:])
