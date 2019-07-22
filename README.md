# Boulder_2019

Behaviour: In almost all animals there exists a literal information bottleneck (or, minimally, a funnel) between the number of neurons in the brain that make cognitive decisions and the motor units responsible for the performance of behaviors.  For example, in vertebrates, this is the spinal cord, and in arthropods, this is the ventral nerve cord.  In fruit flies, there are order 10,000 neurons in the fly brain, but only ~300 symmetric pairs of neurons (descending neurons) that transmit information from the brain to the rest of the body.  Moreover, in recent optogenetic experiments, we have found that while some of the descending neurons reliably elicit a single behavior when stimulated (e.g. the fly grooming its head uncontrollably), most display context-dependency, where the behavior elicited depends on the animal’s behavior before stimulation. To theoretically-explore how information bottlenecks may affect the encoding of behavioral commands, my suggestion is the following model.  Imagine that there are N tasks that an animal needs to perform, and to achieve the task, the animal must create a binary word amongst M binary motor units (e.g. task 14 requires units 1, 17, 34, and 57 to turn-on, and all the rest to be zero).  However, the brain can only command these units through a bottleneck of R descending neurons (R < M, N).  Let this be set-up like a three-layer neural network, with task being encoded at the top layer, the descending neurons being the intermediate layer, and the motor units being the bottom layer, and weights are trained to performed this mapping as best as possible. There are a bunch of questions one could ask of this set-up, but a few suggestions for simulation/calculation projects are: 1) What is the capacity of this network (given M, R, and a number of active units per task, what is the maximal N)? 2) As this capacity is neared, do we see evidence of context-dependency?  For example, if one of the descending neurons is arbitrarily stimulated (set to ‘1’ regardless of input), how does the set of activated motor units change depending on the brain-level input?  3) From the near-capacity solutions, are there particular task-behavior-commands that are more robust to changes in weights than others?  It’s possible that these are likely to be more conserved behaviors and less robust ones are more likely to show inter-species differences. Define a measure for robustness and see how many behaviors should be robust and how many are less so.  If you replace one mapping with another (say, task 18 used to need 4, 3, 10, and 40 activated, but now it needs 12, 60, 111, and 120 activated), after training, starting from the previous mapping as an initial condition, which other mappings alter?  Are they more/less robust now? (Gordon)

Data up to 19.07.2019:
python nn.py --inputSize 50 100 200 300 --hiddenSize 0.0 0.25 0.5 0.6 0.7 0.75 0.8 0.85 0.9 1.0 --epochs 50000
for three different outputs .dat, _1.dat, _2.dat

New data with dynamic stopping will be:
python nn.py --inputSize 50 100 200 300 --hiddenSize 0.0 0.1 0.25 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 1.0 --epochs 100500

To do list:
- ~~implement dynamic stopping (Veronika)~~ 
- re run simulations with more sizes of hidden layer (Riccardo)
- in analysis_pred: compute the number of activated behaviors (threshold: look at the distribition of data: is there a valley between the two peaks (0,1), anyway check that the results are robust upon change of the threshold) (Grace)
- ~~try Riccardo's code on one input size + hidden sizes to check for loss (Riccardo)~~
- think about how to implement point 2) about context dependency (Anjalika)
- correlated input output ?

