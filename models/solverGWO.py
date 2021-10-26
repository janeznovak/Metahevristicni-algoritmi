import math
import pickle

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random as rand
# from numpy import random
import random
from sklearn import decomposition
from deap import creator, base, tools, algorithms
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import matplotlib.ticker as ticker
import os.path

from bioproc.proc_opt import BioProc
from bioproc.proc_models import *

import time
import multiprocessing

'''
Regions consist of cloud of points and principal component that govern the direction of exploration  
'''


class Region:
    def __init__(self, points, model, label, depth=1):
        self.points = np.array(points)
        self.model = model
        self.pca = PCA(n_components=self.model.nParams)
        self.components = None
        self.prevComponents = None
        self.cluster = False
        self.terminated = False
        self.iter = 0
        self.maxIter = 10
        self.threshold = 0.001
        self.label = label
        self.maxVarScale = 6
        self.minVarScale = 3
        self.varScaleDt = (self.maxVarScale - self.minVarScale) / (float(self.maxIter))
        self.varScale = self.maxVarScale
        self.depth = depth

    def updateVariance(self):
        self.varScale = self.varScale - self.varScaleDt

    def updateIter(self):
        self.iter = self.iter + 1
        self.updateVariance()

    def fitPCA(self):
        self.prevComponents = self.components
        self.pca.fit(self.points)
        self.components = self.pca.components_

    def transform(self, points):
        return self.pca.transform(points)

    def inverse_transform(self, points):
        return self.pca.inverse_transform(points)

    def converged(self):
        if self.components is None or self.prevComponents is None:
            return False
        return np.linalg.norm(self.components - self.prevComponents) < self.threshold

    def explored(self):
        return self.terminated or self.iter > self.maxIter or self.converged()


'''
The main class
'''

# must be global for parallelization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Candidate", list, fitness=creator.FitnessMax)

global cunt


def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value


class wolf:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

        self.fitness = fitness(self.position)  # curr fitness

def create_object(tuple):
    wolf1 = wolf(tuple[0], tuple[1], tuple[2], tuple[3], tuple[4])
    # print("Creating object")
    return wolf1

class SolverGWO:
    def __init__(self, model, populationSize=100, NGEN=10, nsamples=1e5):
        self.model = model
        self.populationSize = populationSize
        self.NGEN = NGEN
        self.nsamples = int(nsamples)
        self.indpb = 0.75


        # GA operators
        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("Candidate", list, fitness=creator.FitnessMax)
        # self.toolbox = base.Toolbox()
        # self.toolbox.register("candidate", self.generateCandidate)
        # self.toolbox.register("population", tools.initRepeat, list, self.toolbox.candidate)
        # self.toolbox.register("mate", tools.cxTwoPoint)
        # self.toolbox.register("mutate", self.mutateCandidate, indpb=self.indpb, mult=0.5)
        # self.toolbox.register("select", tools.selTournament, tournsize=int(self.populationSize / 10))

        # estimate initial values with GA

    def findNominalValues(self):
        # fitness = self.model.modes[0]
        # ff = fitness([33.706, 16.347, 14.181, 34.356, 2.913, 0.4983, 2.0619, 4.886, 37.874, 0.413, 7.451, 4.153])
        # print("This is fitness value of one: " + str(ff))

        tic = time.perf_counter()
        pool = multiprocessing.Pool()
        arguments = [[self.model.modes[0], self.model.nParams, 0.1, 50, 1]] * self.populationSize
        # print(arguments)
        res = pool.map(create_object, arguments)
        print(str(res[0].fitness) + " " + str(res[0].position))
        # print(res)
        # print(len(list(res)))
        # population = [wolf(self.model.modes[0], self.model.nParams, 0.1, 50, i) for i in range(self.populationSize)]
        toc = time.perf_counter()
        print(str(toc - tic))
        exit(43)
        nominalVals = []

        for evalMode in self.model.modes:
            nominalValsMode = []

            # self.toolbox.register("map", pool.map)

            # initialize alpha, beta, and delta_pos
            Alpha_pos = numpy.zeros(self.model.nParams)
            Alpha_score = float("inf")

            Beta_pos = numpy.zeros(self.model.nParams)
            Beta_score = float("inf")

            Delta_pos = numpy.zeros(self.model.nParams)
            Delta_score = float("inf")

            # initialize new random population
            Positions = numpy.zeros((self.populationSize, self.model.nParams))
            print("This is shape: " + str(Positions.shape))
            for i in range(12):
                Positions[:, i] = (
                        numpy.random.uniform(0, 1, 100) * (self.model.parameter_values[self.model.params[i]]["max"] -
                                                           self.model.parameter_values[self.model.params[i]]["min"]) +
                        self.model.parameter_values[self.model.params[i]]["min"]
                )

            for gen in range(self.NGEN):

                print("GENERATION: " + str(gen) + " ****************************************************")

                # When you start a new iteration get the values from Positions to posi, which is a list
                posi = Positions.tolist()


                # Run the simulation on positions, which are in a list order in posi
                # fitness = pool.map(fitness_rastrigin, posi)
                # print(type(fitness))
                # print(fitness)

                for i in range(0, self.populationSize):
                    fitness = fitness_rastrigin(Positions[i])
                    # Update Alpha, Beta, and Delta
                    if fitness < Alpha_score:
                        Delta_score = Beta_score  # Update delte
                        Delta_pos = Beta_pos.copy()
                        Beta_score = Alpha_score  # Update beta
                        Beta_pos = Alpha_pos.copy()
                        Alpha_score = fitness
                        # Update alpha
                        Alpha_pos = Positions[i, :].copy()

                    if fitness > Alpha_score and fitness < Beta_score:
                        Delta_score = Beta_score  # Update delte
                        Delta_pos = Beta_pos.copy()
                        Beta_score = fitness  # Update beta
                        Beta_pos = Positions[i, :].copy()

                    if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                        Delta_score = fitness  # Update delta
                        Delta_pos = Positions[i, :].copy()

                    print(type(fitness))
                    print(fitness)
                    if self.model.isViable(Positions[i], fitness):
                        nominalValsMode.append(Positions[i])

                a = 2 - gen * ((2) / self.NGEN)
                # a decreases linearly from 2 to 0

                # Update the Position of search agents including omegas
                for i in range(0, self.populationSize):
                    for j in range(0, self.model.nParams):
                        r1 = random.random()  # r1 is a random number in [0,1]
                        r2 = random.random()  # r2 is a random number in [0,1]

                        A1 = 2 * a * r1 - a
                        # Equation (3.3)
                        C1 = 2 * r2
                        # Equation (3.4)

                        D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                        # Equation (3.5)-part 1
                        X1 = Alpha_pos[j] - A1 * D_alpha
                        # Equation (3.6)-part 1

                        r1 = random.random()
                        r2 = random.random()

                        A2 = 2 * a * r1 - a
                        # Equation (3.3)
                        C2 = 2 * r2
                        # Equation (3.4)

                        D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                        # Equation (3.5)-part 2
                        X2 = Beta_pos[j] - A2 * D_beta
                        # Equation (3.6)-part 2

                        r1 = random.random()
                        r2 = random.random()

                        A3 = 2 * a * r1 - a
                        # Equation (3.3)
                        C3 = 2 * r2
                        # Equation (3.4)

                        D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                        # Equation (3.5)-part 3
                        X3 = Delta_pos[j] - A3 * D_delta
                        # Equation (3.5)-part 3

                        Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
                    # Return back the search agents that go beyond the boundaries of the search space
                    for j in range(self.model.nParams):
                        # print("This is i: " + str(i) + " and j: " + str(j))
                        Positions[i, j] = numpy.clip(Positions[i, j], self.model.parameter_values[self.model.params[j]]["min"], self.model.parameter_values[self.model.params[j]]["max"])


            nominalVals.extend(nominalValsMode)

            #     print("This is generation: " + str(gen))
            #     print("This is length of NGEN: " + str(self.NGEN))
            #     # generate offspprings with crossover and mutations
            #     offspring = algorithms.varAnd(self.popu, self.toolbox, cxpb=0.5, mutpb=0.75)
            #     print("Offspring is done")
            #     print("This is type of offspring: " + str(type(offspring)))
            #     print("This is length of offspring: " + str(len(offspring)))
            #     print("THIS IS OFFSPRING:")
            #     print(*offspring)
            #     # evaluate individuals
            #     fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            #     print("Fits is done")
            #     counter = 0
            #     for fit, ind in zip(fits, offspring):
            #         print("Counter:" + str(counter))
            #         print("This is ind:")
            #         print(ind)
            #         print("And this is fit:")
            #         print(fit)
            #         if self.model.isViable(ind, fitness=fit) and ind not in nominalValsMode:
            #             nominalValsMode.append(ind)
            #         ind.fitness.values = fit
            #         counter = counter + 1
            #     blop = time.perf_counter()
            #     print("Time so far: " + str(blop - tic) + "s")
            #     # roulete wheel selection
            #     self.popu = self.toolbox.select(offspring, k=len(self.popu))
            #
            pool.close()
            print("Number of viable points: " + str(len(nominalValsMode)))
            # nominalVals.extend(nominalValsMode)
        toc = time.perf_counter()
        print("Elapsed time: " + str(toc - tic) + "s")
        return nominalVals

        # creates an array of random candidates

    def generateCandidate(self):
        candidate = []
        for ind in range(self.model.nParams):
            candidate.append(random.uniform(self.model.parameter_values[self.model.params[ind]]["min"],
                                            self.model.parameter_values[self.model.params[ind]]["max"]))
        return creator.Candidate(candidate)

    def checkOutAllBounds(self, candidate):
        for idx, val in enumerate(candidate):
            if self.checkOutOfBounds(candidate, idx):
                return True
        return False

    def checkOutOfBounds(self, candidate, idx):
        # if out of bounds return True
        if candidate[idx] < self.model.parameter_values[self.model.params[idx]]["min"] or candidate[idx] > \
                self.model.parameter_values[self.model.params[idx]]["max"]:
            return True
        return False

        # returns a tuple of mutated candidate

    def mutateCandidate(self, candidate, indpb, mult):
        for idx, val in enumerate(candidate):
            rnd = random.uniform(0, 1)
            if rnd >= indpb:
                rnd2 = random.uniform(1 - mult, 1 + mult)
                candidate[idx] = val * rnd2
                if candidate[idx] < self.model.parameter_values[self.model.params[idx]]["min"]:
                    candidate[idx] = self.model.parameter_values[self.model.params[idx]]["min"]
                if candidate[idx] > self.model.parameter_values[self.model.params[idx]]["max"]:
                    candidate[idx] = self.model.parameter_values[self.model.params[idx]]["max"]
        return candidate,

    def getViablePoints(self, points):
        pool = multiprocessing.Pool()

        viables = np.array(pool.map(self.model.isViable, points))
        viable = np.array(points)[viables]
        """                
        viable = list() 
        i = 0
        for point in points:  
            i += 1
            if i % 1000 == 0:
                print(i)     

            #check if point is viable 
            if self.model.isViable(point): 
                viable.append(point)        
        """
        pool.close()
        return viable

        # gap statistic method

    # returns the optimal number of clusters
    def gapStatistic(self, region, number_ref=10, max_clusters=2, plot=False):
        # sample size is equal to the number of samples in gaussian sampling
        sample_size = self.nsamples
        subjects = np.array(region.points)
        gaps = []
        deviations = []
        references = []
        clusters_range = range(1, max_clusters + 1)

        transformed = region.transform(subjects)
        # get min and max parameter values in pca space
        minP = np.min(transformed, axis=0)
        maxP = np.max(transformed, axis=0)

        for gap_clusters in clusters_range:
            print(gap_clusters)
            reference_inertia = []
            for index in range(number_ref):
                # OBB ... orientated bounding box
                # random sampling within the PCA bounding box
                reference = minP + random.rand(sample_size, self.model.nParams) * (maxP - minP)
                reference = region.inverse_transform(reference)

                kmeanModel = KMeans(gap_clusters)
                kmeanModel.fit(reference)
                reference_inertia.append(kmeanModel.inertia_)

            kmeanModel = KMeans(gap_clusters)
            kmeanModel.fit(subjects)
            log_ref_inertia = np.log(reference_inertia)
            # calculate gap
            gap = np.mean(log_ref_inertia) - np.log(kmeanModel.inertia_)
            sk = math.sqrt(1 + 1.0 / number_ref) * np.std(log_ref_inertia)
            gaps.append(gap)
            deviations.append(sk)

            # Plot the gaps
        if plot:
            plt.clf()
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            lines = plt.errorbar(clusters_range, gaps, ecolor='dodgerblue', yerr=deviations, fmt='-',
                                 color='dodgerblue')
            plt.setp(lines[0], linewidth=1.5)
            plt.ylabel('Gaps')
            plt.show()

            # return optimal number of clusters
        for k in range(0, max_clusters - 1):
            if gaps[k] >= gaps[k + 1] - deviations[k + 1]:
                print("Optimal number of clusters: " + str(k + 1))
                return k + 1
        print("Optimal number of clusters: " + str(max_clusters))
        return max_clusters

    # returns the viable volume for
    def getViableVolume(self, viableRegions, sample_size=int(1e4)):  # 1e4
        volume = 0

        for region in viableRegions:
            regPoints = region.points
            region.fitPCA()
            transformed = region.transform(regPoints)

            minP = np.min(transformed, axis=0)
            maxP = np.max(transformed, axis=0)

            dP = maxP - minP
            volB = np.prod(dP)

            mcRef = minP + random.rand(sample_size, self.model.nParams) * dP
            mcRef = region.inverse_transform(mcRef)

            viaPoints = self.getViablePoints(mcRef)
            count = np.ma.size(viaPoints, axis=0)

            # volume for region
            ratio = count / sample_size
            volume = volume + ratio * volB

        total = self.model.getTotalVolume()
        ratio = volume / total

        description = "Bounding box volume:" + str(volB) + "\n"
        description += "Volume:" + str(volume) + "\n"
        description += "Total volume:" + str(total) + "\n"
        description += "Volume ratio:" + str(ratio)

        return (volume, total, ratio), description

    def setBoxColors(self, bp, nRegions, ax, colors=["#0E74C8", "#15A357", "r", "k"]):
        colorLen = len(colors)

        for i in range(nRegions):
            col = colors[i % colorLen]
            plt.setp(bp['boxes'][i], color=col, linewidth=1.5)
            plt.setp(bp['caps'][2 * i], color=col, linewidth=1.5)
            plt.setp(bp['caps'][2 * i + 1], color=col, linewidth=1.5)
            plt.setp(bp['whiskers'][2 * i], color=col, linewidth=1.5)
            plt.setp(bp['whiskers'][2 * i + 1], color=col, linewidth=1.5)
            plt.setp(bp['fliers'][i], color=col)
            plt.setp(bp['medians'][i], color=col, linewidth=1.5)

    def plotParameterVariances(self, viableSets, names=None, units=None):
        # go through all parameters
        params = self.model.params
        figure = plt.figure()
        nRows = math.ceil(len(params) / 3)
        for pcount, param in enumerate(params):
            ax1 = plt.subplot(nRows, 3, pcount + 1)
            # if names == None:
            #   ax1.set_title(str(param) + str(pcount))
            # else:
            #   ax1.set_title(names[pcount])
            if units != None:
                plt.ylabel(names[pcount] + " " + units[pcount])
            allRegions = []
            # go through all regions
            numSets = len(viableSets)
            allNames = []
            allBoxes = []
            for count, reg in enumerate(viableSets):
                points = np.array(reg.points)
                data = points[:, pcount]
                allRegions.append(data)
                allNames.append("Region " + str(count + 1))
            bp = ax1.boxplot(allRegions, positions=list(range(1, numSets + 1)), widths=0.4)
            self.setBoxColors(bp, numSets, ax1)
            allBoxes = bp['boxes']

            # draw legend
        figure.legend(allBoxes, allNames, 'lower right')
        plt.show()

        # Main method

    def run(self, filename, maxDepth=0):
        # filename is a file to which viable sets will be serialized

        # estimate the inital viable set
        viablePoints = self.findNominalValues()

        if not viablePoints:
            print("No viable points found!")
            return

        return viablePoints


if __name__ == '__main__':
    param_values = {"transcription": {"min": 0.01, "max": 50},
                    "translation": {"min": 0.01, "max": 50},
                    "protein_production": {"min": 0.1, "max": 50},
                    "rna_degradation": {"min": 0.1, "max": 100},
                    "protein_degradation": {"min": 0.001, "max": 50},
                    "hill": {"min": 1, "max": 5},
                    "Kd": {"min": 0.01, "max": 250},
                    "protease_concentration": {"min": 10, "max": 1000}
                    }

    # flip flop with instructions external clock
    filename = os.path.join(".", "bioproc", "three_bit_model_new_new", "bioproc")
    print(filename)
    model = BioProc(np.array(
        ["protein_production", "protein_production", "protein_production", "protein_production", "protein_degradation",
         "protein_degradation", "Kd", "hill", "protein_production", "protein_degradation", "Kd", "hill"]),
                    model_mode=three_bit_processor_ext, parameter_values=param_values, avg_dev=30)
    solver = SolverGWO(model)
    solver.run(filename, maxDepth=1)  # do not cluster

    """ 
    model modes: 
        - one_bit_processor_ext
        - two_bit_processor_ext 
        - three_bit_processor_ext
    """
