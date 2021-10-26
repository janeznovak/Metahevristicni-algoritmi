import copy
import math
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random as rand
from numpy import random
from sklearn import decomposition
from deap import creator, base, tools, algorithms
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import matplotlib.ticker as ticker
import os.path
import random

from bioproc.proc_opt import BioProc
from bioproc.proc_models import *

import time
import multiprocessing



'''
The main class
'''

# must be global for parallelization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Candidate", list, fitness=creator.FitnessMax)

global cunt


class whale:
    def __init__(self, fitness, model):

        candidate = []
        for ind in range(model.nParams):
            candidate.append(random.uniform(model.parameter_values[model.params[ind]]["min"],
                                            model.parameter_values[model.params[ind]]["max"]))
        self.position = candidate
        self.position1 = candidate
        self.fitness = fitness(self.position)  # curr fitness


def create_object(tuple):
    whale1 = whale(tuple[0], tuple[1])
    # print("Creating object")
    return whale1


class SolverWOAGFG:
    def __init__(self, model, populationSize=100, NGEN=100, nsamples=1e5):
        self.model = model
        self.populationSize = populationSize
        self.NGEN = NGEN
        self.nsamples = int(nsamples)
        self.indpb = 0.75

    def findNominalValues(self):
        bestInIteration = []
        pool = multiprocessing.Pool()
        rnd = random.Random(0)

        arguments = [[self.model.modes[0], self.model]] * self.populationSize
        tic = time.perf_counter()

        whalePopulation = pool.map(create_object, arguments)
        print("WhalePopulation initialization done")
        # whalePopulation = [create_object(arguments[i]) for i in range(self.populationSize)]

        # compute the value of best_position and best_fitness in the whale Population
        Xbest = [0.0 for i in range(self.model.nParams)]
        Fbest = sys.float_info.max

        for i in range(self.populationSize):  # check each whale
            if whalePopulation[i].fitness[0] < Fbest:
                Fbest = whalePopulation[i].fitness[0]
                Xbest = copy.copy(whalePopulation[i].position)

        # main loop
        Iter = 0
        posi = []
        while Iter < self.NGEN:

            # after every 10 iterations
            # print iteration number and best fitness value so far
            if Iter % 1 == 0:
                print("Iter = " + str(Iter) + " best fitness = %.3f" % Fbest)

            # linearly decreased from 2 to 0
            a = 2 * (1 - Iter / self.NGEN / 20)
            a2 = -1 + Iter * ((-1) / self.NGEN / 20)

            posi.clear()

            for i in range(self.populationSize):
                A = 2 * a * rnd.random() - a
                C = 2 * rnd.random()
                b = 1
                l = (a2 - 1) * rnd.random() + 1;
                p = rnd.random()

                D = [0.0 for i in range(self.model.nParams)]
                D1 = [0.0 for i in range(self.model.nParams)]
                Xnew = [0.0 for i in range(self.model.nParams)]
                Xrand = [0.0 for i in range(self.model.nParams)]
                if p < 0.5:
                    if abs(A) > 1:
                        for j in range(self.model.nParams):
                            D[j] = abs(C * Xbest[j] - whalePopulation[i].position[j])
                            Xnew[j] = Xbest[j] - A * D[j]
                    else:
                        p = random.randint(0, self.populationSize - 1)
                        while (p == i):
                            p = random.randint(0, self.populationSize - 1)

                        Xrand = whalePopulation[p].position

                        for j in range(self.model.nParams):
                            D[j] = abs(C * Xrand[j] - whalePopulation[i].position[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                else:
                    for j in range(self.model.nParams):
                        D1[j] = abs(Xbest[j] - whalePopulation[i].position[j])
                        Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]

                for j in range(self.model.nParams):
                    if Xnew[j] > self.model.parameter_values[self.model.params[j]]["max"]:
                        Xnew[j] = self.model.parameter_values[self.model.params[j]]["max"] - random.uniform(0, 1)
                    elif Xnew[j] < self.model.parameter_values[self.model.params[j]]["min"]:
                        Xnew[j] = self.model.parameter_values[self.model.params[j]]["min"] + random.uniform(0, 1)
                    else:
                        whalePopulation[i].position1[j] = Xnew[j]
                posi.append(Xnew)
            toc = time.perf_counter()
            print(" This is time: " + str(toc - tic))

            # for i in range(self.populationSize):
            #     # if Xnew < minx OR Xnew > maxx
            #     # then clip it
            #     for j in range(self.model.nParams):
            #         whalePopulation[i].position[j] = max(whalePopulation[i].position[j], self.model.parameter_values[self.model.params[j]]["min"])
            #         whalePopulation[i].position[j] = min(whalePopulation[i].position[j], self.model.parameter_values[self.model.params[j]]["max"])


            # need to parallel the fitness calculation
            # whalePopulation[i].fitness = fitness(whalePopulation[i].position)
            new = pool.map(self.model.modes[0], posi)

            for i in range(self.populationSize):
                if self.check_parameters(whalePopulation[i]):
                    whalePopulation[i].fitness = new[i]
                    whalePopulation[i].position = posi[i]
                    if (new[i][0] > Fbest):
                        Xbest = copy.copy(whalePopulation[i].position)
                        Fbest = whalePopulation[i].fitness
                        print("This is best fitness: " + str(Fbest))
                        print("At position: " + str(whalePopulation[i].position))
            bestInIteration.append(-Fbest[0])
            Iter += 1

        # iterations = list(range(1, 101))
        # fig, ax = plt.subplots()
        # ax.plot(iterations, bestInIteration)
        # ax.set(xlabel='iteracije', ylabel='vrednost funkcije', title='graf vrednosti funkcije za WOA')
        # plt.show()
        exit(11)





        ############################################################################################################
        fitness = self.model.modes[0]
        ff = fitness([33.706, 16.347, 14.181, 34.356, 2.913, 0.4983, 2.0619, 4.886, 37.874, 0.413, 7.451, 4.153])
        print("This is fitness value of one: " + str(ff))

        tic = time.perf_counter()
        nominalVals = []

        for evalMode in self.model.modes:
            nominalValsMode = []

            # initialize new random population
            self.popu = self.toolbox.population(self.populationSize)
            self.toolbox.register("evaluate", evalMode)

            pool = multiprocessing.Pool()
            self.toolbox.register("map", pool.map)

            for gen in range(self.NGEN):
                print("This is generation: " + str(gen))
                print("This is length of NGEN: " + str(self.NGEN))
                # generate offspprings with crossover and mutations
                offspring = algorithms.varAnd(self.popu, self.toolbox, cxpb=0.5, mutpb=0.75)
                print("Offspring is done")
                print("This is type of offspring: " + str(type(offspring)))
                print("This is length of offspring: " + str(len(offspring)))
                print("THIS IS OFFSPRING:")
                print(*offspring)
                # evaluate individuals
                fits = self.toolbox.map(self.toolbox.evaluate, offspring)
                print("Fits is done")
                counter = 0
                for fit, ind in zip(fits, offspring):
                    print("Counter:" + str(counter))
                    print("This is ind:")
                    print(ind)
                    print("And this is fit:")
                    print(fit)
                    if self.model.isViable(ind, fitness=fit) and ind not in nominalValsMode:
                        nominalValsMode.append(ind)
                    ind.fitness.values = fit
                    counter = counter + 1
                blop = time.perf_counter()
                print("Time so far: " + str(blop - tic) + "s")
                # roulete wheel selection
                self.popu = self.toolbox.select(offspring, k=len(self.popu))

            pool.close()
            print("Number of viable points: " + str(len(nominalValsMode)))
            nominalVals.extend(nominalValsMode)
        toc = time.perf_counter()
        print("Elapsed time: " + str(toc - tic) + "s")
        return nominalVals

    def check_parameters(self, whale):
        for i in range(len(whale.position)):
            if whale.position[i] > self.model.parameter_values[self.model.params[i]]["max"] or whale.position[i] < self.model.parameter_values[self.model.params[i]]["min"]:
                return False

        return True



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
        exit(99)




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
    solver = SolverWOAGFG(model)
    solver.run(filename, maxDepth=1)  # do not cluster

    """ 
    model modes: 
        - one_bit_processor_ext
        - two_bit_processor_ext 
        - three_bit_processor_ext
    """
