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




'''
The main class
'''

class Particle:
    def __init__(self, model):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=(-1,)          # best fitness individual
        self.err_i=(-100,)               # fitness individual
        self.mode = model.modes[0]
        self.num_particles = len(model.params)

        
        candidate = []
        for i in range(0, model.nParams):
            candidate.append(random.uniform(model.parameter_values[model.params[i]]["min"],
                                            model.parameter_values[model.params[i]]["max"]))
            self.velocity_i.append(random.uniform(-1,1))
        self.position_i = candidate
        self.position_curr = candidate

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g, w):
        print("the g", pos_best_g)
        w=w       # constant inertia weight
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,self.num_particles - 1):
            r1=random.random()
            r2=random.random()
            # print("errors", self.pos_best_i[i], self.position_i[i], self.num_particles, len(self.pos_best_i), len(self.position_i))

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,model):
        for i in range(0,len(model.params)):
            self.position_curr[i] = self.position_i[i] #for checking if it the last one was the same as now
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > model.parameter_values[model.params[i]]["max"]:
                self.position_i[i] = model.parameter_values[model.params[i]]["max"] - random.uniform(0, 1)

            # adjust minimum position if neseccary
            if self.position_i[i] < model.parameter_values[model.params[i]]["min"]:
                self.position_i[i] = model.parameter_values[model.params[i]]["min"] + random.uniform(0, 1)

def evaluate(parti): # parti is a particle object
    parti.err_i=parti.mode(parti.position_i)
    # print(f"fitness: {parti.err_i}")

    # check to see if the current position is an individual best
    if parti.err_i[0] < parti.err_best_i[0] or parti.err_best_i[0]==-1:
        parti.pos_best_i=parti.position_i
        parti.err_best_i=parti.err_i
    return parti

def create_object(tuple):
    particle = Particle(tuple)
    # print("Creating object")
    return particle


class SolverPSONR:
    def __init__(self, model, populationSize=300, NGEN=100, nsamples=1e5):
        self.model = model
        self.populationSize = populationSize
        self.NGEN = NGEN
        self.nsamples = int(nsamples)
        self.indpb = 0.75
        # self.num_dimensions
        self.err_best_g= -100 # best fitness for group
        self.pos_best_g=[]

    def findNominalValues(self):
        pool = multiprocessing.Pool()
        rnd = random.Random(0)

        tic = time.perf_counter()
        goodEnough = [] # for positions
        itsFitness = [] # for goodEnough fitness
        bestInIteration = [] # for the best in iteration
        times = []

        arguments = [self.model] * self.populationSize
        swarm = pool.map(create_object, arguments) # create the particles
        # print(f"the swarm", swarm[0].position_i)

        # posi = [] # positions of the swarm
        # for part in swarm:
        #     posi.append(part.position_i)
        # print(posi, type(posi[0]))
        # for sw in swarm:
        #     print(sw.position_i)

        # establish the swarm
        # swarm=[]
        # for i in range(0,15): # 15 is the number of particles
        #     swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        Iter = 0
        while Iter < self.NGEN:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitnes and save back to swarm array
            # print(f"model: {self.model.modes[0]}")
            # evaluate(self.model.modes[0], swarm[i])
            # print(f"The swarm:", type(swarm))

            swarm = pool.map(evaluate, swarm) # calculate fitness of swarm
            for j in range(0,self.populationSize):
                # change the 
                # determine if current particle is the best (globally)
                if swarm[j].err_i[0] > self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=list(swarm[j].position_i)
                    self.err_best_g=float(swarm[j].err_i[0])
                if swarm[j].err_i[0] > -17:
                    print(swarm[j].position_i, swarm[j].position_curr)
                    goodEnough.append(swarm[j].position_i)
                    itsFitness.append(swarm[j].err_i[0])

            # cycle through swarm and update velocities and position
            for j in range(0,self.model.nParams):
                swarm[j].update_velocity(self.pos_best_g, 0.9)
                swarm[j].update_position(self.model)
            Iter += 1
            tac = time.perf_counter()
            times.append(tac - tic)
            bestInIteration.append(self.err_best_g)
            print(f"The best in iter: {Iter} {self.err_best_g}")
            print(f"time: {tac -tic}")
        end = time.perf_counter()
        times.append(end - tic)
        print(f"end time: {end - tic}")

        with open("bestInIterationPSO-4-300-27-08-2022.txt", "a") as resultFile:
            count = 0
            for best in bestInIteration:
                print(f"Writin to the file: {best}")
                if count == len(bestInIteration) - 1:
                    resultFile.write(str(count) + " " + str(best) + " " + str(times[count]) + " " + str(times[count + 1]) + "\n")
                else:
                    resultFile.write(str(count) + " " + str(best) + " " + str(times[count]) + "\n")
                count += 1
            count = 0
            resultFile.write("good enough: \n")
            for bestIn in goodEnough:
                resultFile.write(str(goodEnough[count]) + " " + str(itsFitness[count]) + "\n")
                count += 1
            resultFile.write("\n")
            resultFile.write("\n")
        return

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
                    model_mode=four_bit_processor_ext, parameter_values=param_values, avg_dev=30)
    #for i in range(9):
    solver = SolverPSONR(model)
    solver.run(filename, maxDepth=1)  # do not cluster

    """ 
    model modes: 
        - one_bit_processor_ext
        - two_bit_processor_ext 
        - three_bit_processor_ext
    """
