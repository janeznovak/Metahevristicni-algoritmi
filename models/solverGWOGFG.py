import copy
import math
import pickle
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
import os

class wolf:
    def __init__(self, fitness, model):

        candidate = []
        for ind in range(model.nParams):
            candidate.append(random.uniform(model.parameter_values[model.params[ind]]["min"],
                                            model.parameter_values[model.params[ind]]["max"]))
        for j in range(model.nParams):
            if candidate[j] > model.parameter_values[model.params[j]]["max"]:
                print("too high")
                exit(77)
            if candidate[j] < model.parameter_values[model.params[j]]["min"]:
                print("too low")
                exit(78)
            


        self.position = candidate
        self.fitness = fitness(self.position)  # curr fitness Check if the fitness is ok, if not change it to edge, like below in the code
        self.xnew = np.zeros(12)


def create_object(tuple):
    wolf1 = wolf(tuple[0], tuple[1])
    # print("Creating object")
    return wolf1


'''
The main class
'''


class SolverGWOGFG:
    def __init__(self, model, populationSize=300, NGEN=100, nsamples=1e5):
        self.model = model
        self.populationSize = populationSize
        self.NGEN = NGEN
        self.nsamples = int(nsamples)
        self.indpb = 0.75


    def findNominalValues(self):
        print(f"Number of cpus: {multiprocessing.cpu_count()}")
        print(f"Number of cpus from os: {os.cpu_count()}")
        pool = multiprocessing.Pool()
        rnd = random.Random(0)
        bestInIteration = []
        goodEnough = []
        times = []

        arguments = [[self.model.modes[0], self.model]] * self.populationSize

        res = pool.map(create_object, arguments)
        #print(res)
        #print("Going to sleep")
        time.sleep(5)
        #print("Out of sleep")

        # start time
        tic = time.perf_counter()

        # We sort it reversed, so it is descending, because wee need maximization
        population = sorted(res, key=lambda temp: temp.fitness, reverse=True)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
        #print("Best first")
        #print(alpha_wolf.fitness)
        #print(alpha_wolf.position)
        #print(beta_wolf.fitness)
        #print(beta_wolf.position)
        #print(gamma_wolf.fitness)
        #print(gamma_wolf.position)

        # main loop of gwo
        Iter = 0
        posi = []
        negative_infinity = float('-inf')
        bestFitInIteration = []
        while Iter < self.NGEN:

            # after every 10 iterations
            # print iteration number and best fitness value so far
            if Iter % 1 == 0 and Iter > 1:
                print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness + " and best position" + str(alpha_wolf.position))

            # linearly decreased from 2 to 0
            a = 2 * (1 - Iter / self.NGEN)

            posi.clear()

            # updating each population member with the help of best three members
            for i in range(self.populationSize):
                # print("This are parameters of wolf")
                # print(population[i].position)
                # print("This is fitness: " + str(population[i].fitness) + " and these are parameters: " + str(population[i].position))
                A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                        2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
                C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

                X1 = [0.0 for i in range(self.model.nParams)]
                X2 = [0.0 for i in range(self.model.nParams)]
                X3 = [0.0 for i in range(self.model.nParams)]
                Xnew = [0.0 for i in range(self.model.nParams)]
                for j in range(self.model.nParams):
                    X1[j] = alpha_wolf.position[j] - A1 * abs(
                        C1 * alpha_wolf.position[j] - population[i].position[j])
                    X2[j] = beta_wolf.position[j] - A2 * abs(
                        C2 * beta_wolf.position[j] - population[i].position[j])
                    X3[j] = gamma_wolf.position[j] - A3 * abs(
                        C3 * gamma_wolf.position[j] - population[i].position[j])
                    Xnew[j] += X1[j] + X2[j] + X3[j]
                for j in range(self.model.nParams):
                    Xnew[j] /= 3.0
                    if Xnew[j] > self.model.parameter_values[self.model.params[j]]["max"]:
                        Xnew[j] = self.model.parameter_values[self.model.params[j]]["max"] - random.uniform(0, 1)
                    elif Xnew[j] < self.model.parameter_values[self.model.params[j]]["min"]:
                        Xnew[j] = self.model.parameter_values[self.model.params[j]]["min"] + random.uniform(0, 1)
                #if Iter % 2 == 0 and i in range(1, 5):
                #    print(Xnew)
                population[i].xnew = Xnew
                # self.check_parameters(Xnew)
                posi.append(Xnew)

            bestBest = negative_infinity
             # fitness calculation of new solution
            fnew = pool.map(self.model.modes[0], posi)
            for m in range(self.populationSize):
                #print(fnew[m] > population[m].fitness)
                if fnew[m][0] > bestBest:
                    bestBest = fnew[m][0]
                # if fnew[m][0] < -1000:
                #     print(posi[m])
                #     exit(89)
                # greedy selection
                if fnew[m] > population[m].fitness:
                    #print(fnew[m])
                    population[m].position = population[m].xnew
                    population[m].fitness = fnew[m]
                    # if population[m].fitness[0] >= -17:
                    #     bestInIteration.append(population[m].fitness[0])
                    #     goodEnough.append(population[m].position)
                if fnew[m] != population[m].fitness and fnew[m][0] >= -30:
                    bestInIteration.append(fnew[m][0])
                    goodEnough.append(population[m].xnew)
                    
            #print(f"this is the best in this iteration: {bestBest}")
            # if bestBest >= -17 and bestBest not in bestInIteration:
            #     bestInIteration.append(bestBest)
            # On the basis of fitness values of wolves
            # sort the population in descending order
            population = sorted(population, key=lambda temp: temp.fitness, reverse=True)

            # best 3 solutions will be called as
            # alpha, beta and gama
            alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
            bestFitInIteration.append(alpha_wolf.fitness)
            tuc = time.perf_counter()
            print(f"this is the best in this iteration: {Iter}: {bestBest}, time: {tuc - tic}")
            times.append(tuc - tic)
            Iter += 1

        pool.close()
        # stop time
        tac = time.perf_counter()
        endTime = tac - tic
        times.append(endTime)


        # print("The best ones", bestInIteration)
        print(f"Number of found: {len(bestInIteration)}")
        # put all the best in iteration in a file
        with open("bestInIterationGWOGFG10-1-300-default-attack-01.09.2022.txt", "a") as resultFile:
            print(f"opened the file")
            count = 0
            for best in bestFitInIteration:
                print(f"writting to file: {best[0]}")
                if count == len(bestFitInIteration) - 1:
                    resultFile.write(str(count) + " " + str(best[0]) + " " + str(times[count]) + " " + str(times[count + 1]) + "\n")
                else:
                    resultFile.write(str(count) + " " + str(best[0]) + " " + str(times[count]) + "\n")
                count += 1
            count = 0
            resultFile.write("good enough:\n")
            for bestIn in bestInIteration:
                resultFile.write(str(goodEnough[count]) + "\n")
                count += 1
            resultFile.write("\n")
            resultFile.write("\n")
        return

        # iterations = list(range(1, 151))
        # fig, ax = plt.subplots()
        # ax.plot(iterations, bestInIteration)
        # ax.set(xlabel='iteracije', ylabel='vrednost funkcije', title='graf vrednosti funkcije za GWO')
        # plt.show()
        # exit(89)





    def check_parameters(self, parameters):
        for i in range(len(parameters)):
            if parameters[i] < self.model.parameter_values[self.model.params[i]]["min"]:
                # print("Checking min")
                parameters[i] = self.model.parameter_values[self.model.params[i]]["min"]
            if parameters[i] > self.model.parameter_values[self.model.params[i]]["max"]:
                # print("Checking max")
                parameters[i] = self.model.parameter_values[self.model.params[i]]["max"]

    def run(self, filename, maxDepth=0):
        # filename is a file to which viable sets will be serialized

        # estimate the inital viable set
        viablePoints = self.findNominalValues()

        if not viablePoints:
            print("No viable points found!")
            return




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
    model_mode=one_bit_processor_ext, parameter_values=param_values, avg_dev=30)
    #for i in range(10):
    #print(f"In range {i}")
    solver = SolverGWOGFG(model)
    solver.run(filename, maxDepth=1)  # do not cluster

    """ 
    model modes: 
        - one_bit_processor_ext
        - two_bit_processor_ext 
        - three_bit_processor_ext
    """
