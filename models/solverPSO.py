import math
import operator
import pickle

import numpy
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

from bioproc.proc_opt import BioProc
from bioproc.proc_models import *

import time
import multiprocessing

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

'''
Regions consist of cloud of points and principal component that govern the direction of exploration  
'''

'''
The main class
'''

# must be global for parallelization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Candidate", list, fitness=creator.FitnessMax)
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None)

global cunt
w = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
minus = [0.5 / 100, 0.5 / 100, 0.5 / 100, 0.5 / 100, 0.5 / 100, 0.5 / 100, 0.5 / 100, 0.5 / 100, 0.5 / 100, 0.5 / 100,
         0.5 / 100, 0.5 / 100]


class SolverPSO:
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

        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.generateCandidate)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.updateParticle1, phi1=2.0, phi2=2.0)

        # estimate initial values with GA

    def findNominalValues(self):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        tic = time.perf_counter()
        nominalVals = []

        for evalMode in self.model.modes:
            nominalValsMode = []

            # initialize new random population
            self.pop = self.toolbox.population(self.populationSize)
            # print(self.pop)
            # print(type(self.pop))
            #
            # Positions = numpy.zeros((100, 12))
            # for i in range(12):
            #     Positions[:, i] = (
            #             numpy.random.uniform(0, 1, 100) * (self.model.parameter_values[self.model.params[i]]["max"] - self.model.parameter_values[self.model.params[i]]["min"]) + self.model.parameter_values[self.model.params[i]]["min"]
            #     )
            # print(Positions[0])
            # posi = Positions.tolist()
            # print(type(posi))
            # print(posi)
            # print(type(Positions))
            pool = multiprocessing.Pool()
            # something = pool.map(evalMode, posi)
            # print(something)
            # exit(90)
            # print("This is population")
            # print(self.pop)
            self.toolbox.register("evaluate", evalMode)

            self.toolbox.register("map", pool.map)
            best = None

            for g in range(self.NGEN):
                print("GENERATION: " + str(g) + "***************************************************")
                values = self.toolbox.map(self.toolbox.evaluate, self.pop)
                # print(values)
                i = 0
                for part in self.pop:
                    # print(type(part))
                    # print("This is particle vertices:")
                    # print(part)
                    # print(self.pop)

                    # part.fitness.values = self.toolbox.evaluate(part)
                    # print("Tuple: " + str(values[i][0]))
                    if math.isnan(values[i][0]):
                        print("These are parameters of nan")
                        print(part)
                        part = self.pop[0]
                        print("Parameters after self.pop[0]: ")
                        print(part)
                        if g == 0:
                            part.best = creator.Particle(self.pop[0])
                            part.best.fitness.values = values[0]
                            print("This should be new part.best: ")
                            print(part.best)
                        # i = i + 1
                        print("IS NAN+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        part.fitness.values = values[0]
                        print("This is part.best of pop when nan")
                        print(self.pop[0].best.fitness)
                    else:
                        part.fitness.values = values[i]

                    if part.fitness.values[0] < -200:
                        print("This are parameters for weird fitness: " + str(part))
                    # print("This is evaluated:")
                    # print(part.fitness.values[0])
                    if not part.best or part.best.fitness < part.fitness:
                        part.best = creator.Particle(part)
                        part.best.fitness.values = part.fitness.values
                        # print(part.best)
                        # exit(11)
                    if not best or best.fitness < part.fitness:
                        best = creator.Particle(part)
                        best.fitness.values = part.fitness.values
                        print("This is part that is ind:")
                        print(part)
                        if self.model.isViable(part, fitness=part.fitness.values) and part not in nominalValsMode:
                            nominalValsMode.append(part)
                    i = i + 1
                    if part.best is None:
                        print("part.best is None. This is part and part.best in loop and g value: " + str(
                            part) + " " + str(
                            part.best) + " " + str(g))
                        exit(101)
                    # self.toolbox.update(part, best)
                    # print("This is speed of a particle: " + str(part.speed))
                print("This is best value: " + str(best.fitness.values) + " in generation: " + str(
                    g) + "and its parameters: " + str(best))
                tac = time.perf_counter()
                print("Time: " + str(tac - tic) + "s")
                for part in self.pop:

                    # Check if part.best is null or empty
                    if part.best is None:
                        print("part.best is None. This is part and part.best: " + str(part) + " " + str(
                            part.best))
                        exit(101)
                    self.toolbox.update(part, best)

                # Gather all the fitnesses in one list and print the stats
                logbook.record(gen=g, evals=len(self.pop), **stats.compile(self.pop))
                print(logbook.stream)
                map(operator.sub, w, minus)
            nominalVals.extend(nominalValsMode)
        return nominalVals
        #     for gen in range(self.NGEN):
        #         print("This is generation: " + str(gen))
        #         print("This is length of NGEN: " + str(self.NGEN))
        #         # generate offspprings with crossover and mutations
        #         offspring = algorithms.varAnd(self.popu, self.toolbox, cxpb=0.5, mutpb=0.75)
        #         print("Offspring is done")
        #         print("This is type of offspring: " + str(type(offspring)))
        #         print("This is length of offspring: " + str(len(offspring)))
        #         print("THIS IS OFFSPRING:")
        #         print(*offspring)
        #         # evaluate individuals
        #         fits = self.toolbox.map(self.toolbox.evaluate, offspring)
        #         print("Fits is done")
        #         counter = 0
        #         for fit, ind in zip(fits, offspring):
        #             print("Counter:" + str(counter))
        #             if self.model.isViable(ind, fitness=fit) and ind not in nominalValsMode:
        #                 nominalValsMode.append(ind)
        #             ind.fitness.values = fit
        #             counter = counter + 1
        #         blop = time.perf_counter()
        #         print("Time so far: " + str(blop - tic) + "s")
        #         # roulete wheel selection
        #         self.popu = self.toolbox.select(offspring, k=len(self.popu))
        #
        #     pool.close()
        #     print("Number of viable points: " + str(len(nominalValsMode)))
        #     nominalVals.extend(nominalValsMode)
        # toc = time.perf_counter()
        # print("Elapsed time: " + str(toc - tic) + "s")
        # return nominalVals

        # creates an array of random candidates

    def findNominalValues1(self):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        tic = time.perf_counter()
        nominalVals = []

        # initialize new random population
        self.pop = self.toolbox.population(self.populationSize)

        pool = multiprocessing.Pool()

        self.toolbox.register("evaluate", self.model.modes[0])

        self.toolbox.register("map", pool.map)

        best = None

        # Now we can go in iterations, which represent "evolution". In each iteration we move the particles and then
        # calculate their value
        for g in range(self.NGEN):
            print("GENERATION: " + str(g) + "***************************************************")
            # evaluate all the particles(for their parameters) and store it in a variable
            values = self.toolbox.map(self.toolbox.evaluate, self.pop)

            # When we have the values at particles' parameters, we check if any of them has moved to a better position(
            # in space), if so we change the local best of it, and check if it is maybe even better than the global best
            # in that case, we change the global best to that particle

            # variable i is used to know at which particle in values we are
            i = 0
            for part in self.pop:
                # Give the particle the value that was calculated and stored in values variable(tuple)
                part.fitness.values = values[i]

                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values

                    # Just some prints, to know what is what
                    print(f'This is part: {part}')
                    print(f'This is part.fitness: {part.fitness}')
                    print(f'This is part.fitness.values: {part.fitness.values}')
                    print(f'This is part.best: {part.best}')
                    print(f'This is part.best.fitness: {part.best.fitness}')
                    print(f'This is part.best.fitness.values: {part.best.fitness.values}')
                    print(f'This is best: {best}')
                    print(f'This is best.fitness: {best.fitness}')

                    i = i + 1
            # Now we got the local best position of each particle and global best particle. We need to update their
            # positions accordingly to their speed and current position(in comparison to the global best).
            # We do this with map function which is parallelized.

            # When we go through all the particles, we need to put i at 0 again
            i = 0
            for part in self.pop:
                self.toolbox.update(part, best)

            # Now that we have the global best, we need to update all the particles
            # exit(999)


    def generateCandidate(self):
        candidate = []
        for ind in range(self.model.nParams):
            candidate.append(random.uniform(self.model.parameter_values[self.model.params[ind]]["min"],
                                            self.model.parameter_values[self.model.params[ind]]["max"]))
        part = creator.Particle(candidate)
        part.speed = [random.uniform(-1, 1) for _ in range(self.model.nParams)]
        part.smax = 1
        part.smin = -1
        return part

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
        # viablePoints = self.findNominalValues()

        # Trying out
        viablePoints = self.findNominalValues1()

        if not viablePoints:
            print("No viable points found!")
            return

            # dump viable points to file

    def updateParticle(self, part, best, phi1, phi2):
        print("This is part and part.best:")
        print(part)
        print(part.best)
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = list(map(operator.add, part, part.speed))
        self.check_params(part)

    def updateParticle1(self, part, best, phi1, phi2):
        print("This is part and part.best:")
        print(part)
        print(part.best)
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = list(map(operator.add, part, part.speed))
        for i in range(len(part)):
            if part[i] < self.model.parameter_values[self.model.params[i]]["min"]:
                part[i] = self.model.parameter_values[self.model.params[i]]["min"] + random.uniform(0, 1)
            if part[i] > self.model.parameter_values[self.model.params[i]]["max"]:
                part[i] = self.model.parameter_values[self.model.params[i]]["max"] - random.uniform(0, 1)

    def check_params(self, particle):
        for i in range(len(particle)):
            if particle[i] < self.model.parameter_values[self.model.params[i]]["min"]:
                particle[i] = self.model.parameter_values[self.model.params[i]]["min"]
            if particle[i] > self.model.parameter_values[self.model.params[i]]["max"]:
                particle[i] = self.model.parameter_values[self.model.params[i]]["max"]


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
    # evalMode([26.21148920747431, 19.985924982983867, 42.63863455110639, 15.464715609539361, 14.809546046765158,
    #           2.6835833908713513, 0.6211809975977776, 4.754512814131854, 40.20691486599725, 5.871720376553965,
    #           151.98292098146118, 4.934930939650192])
    # flip flop with instructions external clock
    filename = os.path.join(".", "bioproc", "three_bit_model_new_new", "bioproc")
    print(filename)
    model = BioProc(np.array(
        ["protein_production", "protein_production", "protein_production", "protein_production", "protein_degradation",
         "protein_degradation", "Kd", "hill", "protein_production", "protein_degradation", "Kd", "hill"]),
        model_mode=three_bit_processor_ext, parameter_values=param_values, avg_dev=30)
    solver = SolverPSO(model)
    solver.run(filename, maxDepth=1)  # do not cluster

    """ 
    model modes: 
        - one_bit_processor_ext
        - two_bit_processor_ext 
        - three_bit_processor_ext
    """
