import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.functional import Tensor
import torch.nn.functional as F
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin, bin_to_gray
from deap import creator, base, tools, algorithms
import copy

# Set up neural net
class Net(torch.nn.Module):
    # initialise two hidden layers and one output layer
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer 1
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden) # hidden layer 2
        self.out = torch.nn.Linear(n_hidden, n_output) # output layer

    # connect up the layers: the input passes through the hidden, then the sigmoid, then the output layer
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x)) # activation function for hidden layer
        x = torch.sigmoid(self.hidden2(x)) # activation function for hidden layer 2
        x = self.out(x)
        return x

lossFunc = torch.nn.MSELoss() # Mean sqaured error loss function
popSize = 40 # size of population
dimension = 67 # number of dimensions
numOfBits = 30 # number of bits per weight
numOfGenerations = 10000 # number of generations 0.233
nElitists = 1
crossProb = 0.5 # cross probability
totalBits = dimension * numOfBits # total number of bits for an individual
flipProb = 1 / totalBits # bit mutate prob
mutateprob = 0.3 # mutation prob
maxnum = 2**numOfBits-1

data = [] # collection of samples
genArr = np.asarray(list(range(1, numOfGenerations+1))) # list of gens 1 to numOfGenerations

net = Net(n_feature=2, n_hidden=6, n_output=1) # intitialize neural network
toolbox = base.Toolbox()

# fitness function
def fitness(x1, x2):
    f = np.sin(3.5*x1 + 1)*np.cos(5.5*x2)
    return f

# 3D surface plot of fitness function
def surfacePlot(filename):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    z = fitness(x1, x2) # evaluate fitness for the meshgrid (x1, x2)

    surf = ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, 
    antialiased=False, zorder=0)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')
    ax.view_init(45, 45)
    plt.title('3D Surface Plot of Fitness')
    fig.colorbar(surf, shrink=0.8, aspect=10)
    plt.savefig(filename) # save the image
    plt.close(fig)

# Generates data for training and testing
def generateData(numSamples):
    # Generate samples
    x1 = np.random.uniform(-1, 1, numSamples)
    x2 = np.random.uniform(-1, 1, numSamples)
    f = fitness(x1, x2) # calculate fitness values (labels)

    # Arrange x1 and x2 in ([[x1,x2],[x1n,x2n]]) format
    samples = []
    for i in range(len(x1)):
        samples.append([x1[i], x2[i]])

    # Create training data and labels
    # slice samples to take first 1000
    x_train = torch.as_tensor(np.asarray(samples)[:1000], dtype=torch.double)
    y_train = torch.as_tensor(np.asarray(f)[:1000], dtype=torch.double)

    # Create testing data and labels
    # slice samples to take last 100
    x_test = torch.as_tensor(np.asarray(samples)[1000:], dtype=torch.double)
    y_test = torch.as_tensor(np.asarray(f)[1000:], dtype=torch.double)

    trainingData = (x_train, y_train) # Combine samples and labels for training
    testingData = (x_test, y_test) # Combine samples and labels for testing

    return trainingData, testingData

# 3D scatter plot
def scatterPlot(filename, data, color, label):
    x, y = data

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(x)):
        ax.scatter3D(x[i][0], x[i][1], y[i], c=color, marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')
    plt.title('3D plot of ' + label + ' dataset')
    plt.savefig(filename)
    plt.close(fig)

# Extracts weights out of the neural net
def weightsOutOfNetwork(nn):
    w = []
    for i in nn.parameters():
        w += ((np.array(Tensor.cpu(i.data)).flatten()).tolist()) # append weight value
    return w

# Input weights into neural net
def weightsIntoNetwork(w, nn):
    weights = np.asarray(w)
    nn.hidden.weight = torch.nn.Parameter(torch.from_numpy(weights[:12].reshape(6, )))
    nn.hidden.bias = torch.nn.Parameter(torch.from_numpy(weights[12:18].reshape(1, )))
    nn.hidden2.weight = torch.nn.Parameter(torch.from_numpy(weights[18:54].reshape(6, 6)))
    nn.hidden2.bias = torch.nn.Parameter(torch.from_numpy(weights[54:60].reshape(1, )))
    nn.out.weight = torch.nn.Parameter(torch.from_numpy(weights[60:66].reshape(1, )))
    nn.out.bias = torch.nn.Parameter(torch.from_numpy(weights[66:67].reshape(1, 1)))
    return nn

# creates 3D surface plot of the fitness function
def neuralNetwork3DSurfacePlot(filename):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    xyValues = []

    # Arrange x and y values into ([[x1,x2],[x1n,x2n]]) format
    for i in range(len(X)):
        for j in range(len(X[i])):
            xyValues.append([X[i][j], Y[i][j]])

    inputArr = torch.as_tensor(np.asarray(xyValues), dtype=torch.double) # convert xyValues to tensor

    pred = Tensor.cpu(net(inputArr)).detach().numpy() # get predictions from network

    Z = []
    # Arrange predictions into required format for meshgrid
    for i in range(0,len(pred), 100):
        tmp = []
        for j in range(i,i+100):
            tmp.append(pred[j][0])
            Z.append(tmp)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, np.asarray(Z), rstride=1, cstride=1, 
    cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=0)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')
    ax.view_init(45, 45)
    plt.title('3D Surface Plot of Fitness using a Neural Network')
    fig.colorbar(surf, shrink=0.8, aspect=10)
    plt.savefig(filename)
    plt.close(fig)

# Converts chromosome to weight values
def chrom2Weights(c):
    c = np.array(c).reshape(67, 30) # convert to numpy array and reshape
    w = [] # create empty weights array
    for ind in c:
        ind = chrom2real(ind)
        ind = checkRange(ind)
        w.append(ind)
    return w

# Converts chromosome to real number
def chrom2real(c):
    indasstring=''.join(map(str, c))
    degray=gray_to_bin(indasstring) # convert from gray value to binary
    numasint=int(degray, 2) # convert to int from base 2
    numinrange=-20+40*numasint/maxnum
    return numinrange

# Checks value is in range of 20 to -20 else sets new value
def checkRange(x):
    if x > 20:
        x = 20
    elif x < -20:
        x = -20
    return x

# Converts weights to chromosome
def real2Chrom(weights):
    c = [] # create output array to hold individual
    for w in weights: # clamp weights between -20 and 20
        w = checkRange(w)
        i = (w + 20)*maxnum/40 # convert weight to integer
        b = bin(int(i))[2:].zfill(30) # convert to binary
        g = bin_to_gray(b) # convert to gray-coded digit
        c.append(g)
    c = list(''.join(c))
    newC = []
    for i in range(0,len(c)):
        newC.append(int(c[i]))
    return newC

# Return the loss value of the current individual
def lossMSE(individual):
    w = np.asarray(chrom2Weights(individual)) # array of weights
    weightsIntoNetwork(w, net) # update net with new weights
    loss = lossFunc(net(data[0]).flatten(), data[1]) # run loss function (MSE) using predicted and actual values
    return loss.item(),

# Plots the fitness across generations
def plotFitnessAcrossGen(filename, arr):
    fig = plt.figure()
    plt.plot(genArr, arr, label="Data", color="blue")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness of Best Individual across Generations")
    plt.savefig(filename)
    plt.close(fig)

# lamarckian learning approach to update the geno and pheno types
def lamarckianLearning(individual):
    w = np.asarray(chrom2Weights(individual)) # array of weights
    weightsIntoNetwork(w, net) # update net with new weights

    loss = lossFunc(net(data[0]).flatten(), data[1]) # run loss function (MSE) using predicted and actual values

    w = weightsOutOfNetwork(net) # assign w to updated weights from neural net
    opt = torch.optim.Rprop(net.parameters(), lr=0.05) # intialize RProp opt

    newLoss = 0
    for i in range(0,30):
        newLoss = lossFunc(net(data[0]).flatten(), data[1]) # run loss function (MSE) using predicted and actual values
        opt.zero_grad() # clear gradients for next train
        newLoss.backward() # backpropagation, compute gradients
        opt.step() # apply gradients
        if loss > newLoss: # update weights if newLoss value is better
            w = weightsOutOfNetwork(net)
            loss = newLoss
    newInd = real2Chrom(w) # convert weights to gray coded ind
    return newInd

# baldwinian learning approach to update the pheno type
def baldwinianLearning(individual):
    w = np.asarray(chrom2Weights(individual)) # array of weights
    weightsIntoNetwork(w, net) # update net with new weights

    loss = lossFunc(net(data[0]).flatten(), data[1]) # run loss function (MSE) using predicted and actual values

    opt = torch.optim.Rprop(net.parameters(), lr=0.05) # intialize RProp opt

    newLoss = 0
    for i in range(0,30): 
        opt.zero_grad() # clear gradients for next train
        newLoss = lossFunc(net(data[0]).flatten(), data[1]) # run loss function (MSE) using predicted and actual values
        newLoss.backward(retain_graph=True) # backpropagation, compute gradients
        opt.step() # apply gradients
        weights = weightsOutOfNetwork(net)

        for w in weights:
            w = checkRange(w) # check weight is in range
        weightsIntoNetwork(weights, net)
        if loss > newLoss: # update weights if newLoss value is better
            loss = newLoss
    return loss.item(),

# Plot 2D comparison for 2 arrays against generations
def plotComparison(filename, lamarckian, baldwinian):
    fig = plt.figure()
    plt.plot(genArr, lamarckian, label="Lamarckian Learning", color="blue")
    plt.plot(genArr, baldwinian, label="Baldwinian Learning", color="green")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness of best individual from across the generations")
    plt.savefig(filename)
    plt.close(fig)

# Creates initial population individuals
def intialPop():
    w = []
    for i in range(0, 67):
        w.append(np.random.uniform(-1, 1)) # limits values to between -1 and 1
    w = np.asarray(w)

    newInd = real2Chrom(w) # convert weights to gray coded ind
    return newInd

################################## GA ##################################
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Attribute generator 
# define 'attr_bool' to be an attribute ('gene')
# which corresponds to integers sampled uniformly
# from the range [0,1] (i.e. 0 or 1 with equal
# probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
# define 'individual' to be an individual
# consisting of numOfBits*dimension 'attr_bool' elements 
('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, 
toolbox.attr_bool, numOfBits*dimension)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# register the goal / fitness function
toolbox.register("evaluate", lossMSE)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)

# operator for selecting individuals for breeding the next
# generation: This uses fitness proportionate selection,
# also known as roulette wheel selection
toolbox.register("select", tools.selBest, fit_attr='fitness')

def main(type, dataType):
    bestArr = [] # array to hold best value per generation
    pop = toolbox.population(n=popSize) # create initial population for use in GA

    # Evaluate the entire population
    tmpPop = []
    # Evaluate the entire population dependent on type
    for i in range(0,len(pop)):
        tmpPop.append(intialPop()) # single indiviual intialised between -1 and 1
        for j in range(0,len(pop[i])):
            pop[i][j] = tmpPop[i][j] # assigns individual to population
    fitnesses = list(map(lossMSE, pop)) 

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop] # Extracting all the fitnesses of 

    g = 0 # Variable keeping track of the number of generations

    # Begin the evolution
    while g < numOfGenerations:
        g = g + 1 # A new generation
        print("-- Generation %i --" % g)

        # Evaluate the entire population dependent on type
        if type == 1:
            tmpPop = []
            for i in range(0,len(pop)):
                tmpPop.append(lamarckianLearning(pop[i])) # single indiviual intialised between -1 and 1
                for j in range(0,len(pop[i])):
                    pop[i][j] = tmpPop[i][j] # assigns individual to population
                    fitnesses = list(map(lossMSE, pop)) 
        elif type == 2:
            fitnesses = list(map(baldwinianLearning, pop))
        else:
            fitnesses = list(map(toolbox.evaluate, pop)) 

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        bestArr.append(tools.selBest(pop, 1)[0].fitness.values[0]) # append best generational fitness value
        print(tools.selBest(pop, 1)[0].fitness.values[0])

        offspring = tools.selBest(pop, nElitists) + toolbox.select(pop,len(pop)-
        nElitists) # Select the next generation individuals
        offspring = list(map(toolbox.clone, offspring)) # Clone the selected 
        individuals

        # Apply crossover and mutation on the offspring
        # make pairs of offspring for crossing over
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
        
            # cross two individuals with probability CXPB
            if random.random() < crossProb:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability mutateprob
            if random.random() < mutateprob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring # The population is entirely replaced by the offspring

        # Save best array for each run dependent on type and data used
        if dataType == 'training':
            if type == 0:
                np.save('none_train', bestArr)
            elif type == 1:
                np.save('lamarckian_train', bestArr)
            elif type == 2:
                np.save('baldwinian_train', bestArr)
        else:
            if type == 0:
                np.save('none_test', bestArr)
            elif type == 1:
                np.save('lamarckian_test', bestArr)
            elif type == 2:
                np.save('baldwinian_test', bestArr)

        print("-- End of (successful) evolution --")

if __name__ == "__main__":
    # 3D surface plot
    surfacePlot('Q1_surfacePlot.png')

    # training and testing data plots
    trainingData, testingData = generateData(1100) # Generate data for training and 
    testing
    scatterPlot('Q2_train_scatterPlot.png', trainingData, "blue", "training") 
    scatterPlot('Q2_test_scatterPlot.png', testingData, "red", "testing") 

    data = trainingData # set data to use as training
    print("=========================================== Generic approach - Training Data ==============================================")
    main(0, 'training') # run GA for training data
    trainArr = np.load('none_train.npy') # load best results array for training data
    plotFitnessAcrossGen('Q5_plotFitnessAcrossGen.png', trainArr) # plot fitness across gens for training data
    #plotFitnessAcrossGen('Q5_2_plotFitnessAcrossGen.png', trainArr[:2000]) # plot fitness across gens for training data

    neuralNetwork3DSurfacePlot('Q6_train_surfacePlot.png') # plot 3D surface plot after training data

    data = testingData # set data to use as testing
    print("=========================================== Generic approach - Testing Data ==============================================")
    main(0, 'testing') # run GA for testing data
    testArr = np.load('none_test.npy') # load best results array for testing data
    plotFitnessAcrossGen('Q5_plotFitnessAcrossGentest.png', testArr) # plot fitness across gens for testing data
    #plotFitnessAcrossGen('Q5_2_plotFitnessAcrossGentest.png', testArr[:2000]) # plot fitness across gens for testing data

    neuralNetwork3DSurfacePlot('Q6_test_surfacePlot.png') # plot 3D surface plot after testing data

    print("================================================Extract Weights out of Network ================================================")
    weights = weightsOutOfNetwork(net) # weights extracted from network
    print(weights[:5])
    print("================================================ Convert weights to individual ==============================================")
    ind = real2Chrom(weights) # weights as chromosome (gray form)
    c = np.array(ind).reshape(67, 30)
    print(c[:5])
    real = chrom2Weights(ind) # weights converted back to real numbers
    print(real[:5])

    numOfGenerations = 500 # set generation to run to 1000
    genArr = np.asarray(list(range(1, numOfGenerations+1))) # list of gens 1 to 
    numOfGenerations

    data = trainingData # set data to use to training
    print("=========================================== Lamarckian approach -Training Data ==============================================")
    main(1, 'training') # run GA for Lamarckian approach
    neuralNetwork3DSurfacePlot('Q8_train_surfacePlot.png') # plot 3D surface plot after training data
    print("=========================================== Baldwinian approach -Training Data ==============================================")
    main(2, 'training') # run GA for Baldwinian approach
    neuralNetwork3DSurfacePlot('Q9_train_surfacePlot.png') # plot 3D surface plot after training data

    lamTrain = np.load('lamarckian_train.npy') # load best array for Lamarckian approach
    baldTrain = np.load('baldwinian_train.npy') # load best array for Baldwinian approach

    plotComparison('Q8_train_genPlot.png', lamTrain, baldTrain) # plot comparison of approaches for training data
    plotFitnessAcrossGen('Q8_plotFitnessAcrossGen_train_L.png', lamTrain) # plot fitness across gens for training data Lamarckian approach
    plotFitnessAcrossGen('Q9_plotFitnessAcrossGen_train_B.png', baldTrain) # plot fitness across gens for training data Baldwinian approach

    data = testingData # set data to use to testing

    print("=========================================== Lamarckian approach - Testing Data ==============================================")
    main(1, 'testing') # run GA for Lamarckian approach
    neuralNetwork3DSurfacePlot('Q8_test_surfacePlot.png') # plot 3D surface plot after testing data
    print("=========================================== Baldwinian approach - Testing Data ==============================================")
    main(2, 'testing') # run GA for Baldwinian approach
    neuralNetwork3DSurfacePlot('Q9_test_surfacePlot.png') # plot 3D surface plot after testing data

    lamTest = np.load('lamarckian_test.npy') # load best array for Lamarckian 
    approach
    baldTest = np.load('baldwinian_test.npy') # load best array for Baldwinian 
    approach

    plotComparison('Q8_test_genPlot.png', lamTest, baldTest) # plot comparison of approaches for testing data
    plotFitnessAcrossGen('Q8_plotFitnessAcrossGen_test_L.png', lamTest) # plot fitness across gens for testing data Lamarckian approach
    plotFitnessAcrossGen('Q9_plotFitnessAcrossGen_test_B.png', baldTest) # plot fitness across gens for testing data Baldwinian approach

    print("====================================================== Neural Net ==========================================")
    print(net)

    nn = Net(n_feature=2, n_hidden=6, n_output=1) # Create net with two hidden layers and 6 neurons in each
    weights = weightsOutOfNetwork(nn) # Extracted weights from network
    weightsCheck = nn.hidden.weight.cpu().detach().numpy().flatten()

    print("================================================ Print Weights ================================================")
    print(weights) # prints all initial weight values
    print(nn.hidden.weight) # prints first layer weight values
    print(np.array_equal(weights[:12],weightsCheck)) # evaluate if first layer weights are equal

    # Set new weight values for index 0 to 2
    weights[0] = 0.1
    weights[1] = 0.2
    weights[2] = 0.3

    nn = weightsIntoNetwork(weights, nn)
    weights = weightsOutOfNetwork(nn)

    print("================================================ Print New Weights ================================================")
    print(weights) # prints all new initial weight values
    print(np.array_equal(weights[:12],weightsCheck)) # evaluate if first layer weights are equal

    print("================================================ Print Loss values ================================================")
    print(trainArr[-1])
    print(testArr[-1])
    print(lamTrain[-1])
    print(baldTrain[-1])
    print(lamTrain[-1])
    print(baldTest[-1])

    """data = trainingData # set data to use as training
    loss = []
    for x in range(1, 4):
    if x == 1:
    popSize = 20
    elif x == 2:
    popSize = 40
    elif x == 3:
    popSize = 60 
    for i in range(1, 4):
    if i == 1:
    crossProb = 0.3
    elif i == 2:
    crossProb = 0.5
    elif i == 3:
    crossProb = 0.9
    for j in range(1, 4):
    if j == 1:
    mutateprob = 0.3
    elif j == 2:
    mutateprob = 0.5
    elif j == 3:
    mutateprob = 0.9
    print(popSize, crossProb, mutateprob)
    main(0, 'training') # run GA for training data
    trainArr = np.load('none_train.npy') # load best results array for 
    training data
    loss.append([trainArr[-1],crossProb,mutateprob,popSize])
    print(trainArr[-1])
    print('========================================== end of run 
    ==========================================')"""