import numpy as np
import random

from deap import algorithms, base, creator, tools
from scipy import spatial

MIN = 0
MAX = 999
POPULATION = 1000
NBGEN = 500
TOURNAMENTSIZE = 5
INDPB = 20 / 100
CXPB = 0.7
MUTPB = 0.3
EVOLALGMU = int(POPULATION / 2)
EVOLALGLAMBDA = EVOLALGMU * 2


def _cxTwoPoint(ind1, ind2):
    '''
    Implementation coming from DEAP example:

    https://github.com/DEAP/deap/blob/f6accf730555c5bbc1c50ac310250ad707353080/examples/ga/onemax_numpy.py
    '''
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


def _evalMatching(individual, target, reference, shape, nbclass):
    new = (reference + individual).reshape(shape)

    # normalization to the number of expected classes
    if np.count_nonzero(new) > 0:
            new = np.ceil(new * (nbclass - 1) /
                          new.max()).astype(int)
    _, _, d = spatial.procrustes(target, new)

    # reduction strategy on multiple objectives:
    # 1. spatial analysis to be the closest possible
    # 2. as less as possible new contributions
    return d, individual.sum()


def _initIndividual(icls, content):
    return icls(content)


def _initPopulation(pcls, ind_init, actual, target, popsize, size):
    contents = []

    actual_mean = int(actual.mean() + 1)
    actual_max = int(actual.max()) + 1
    target_max = target * actual_max
    random_ind = np.random.randint(0, actual_max, size=size)
    for _ in range(popsize):
        newind = None
        rand_mul = random.randint(1, actual_max)
        [choice] = random.choices([0, 1, 2, 3], weights=[2, 6, 1, 1])
        if choice == 0:
            newind = np.where(np.logical_and(actual > 0, target > 0),
                              actual, 0)
            newind = np.where(np.logical_and(actual == 0, target > 0),
                              target_max * rand_mul,
                              newind)
        elif choice == 1:
            newind = np.where(target > 0, random_ind, 0)
        elif choice == 2:
            newind = np.where(target > 0,
                              target * 10 - actual + rand_mul, 0)
        else:
            newind = random_ind
        newind[newind < 0] = 0
        contents.append(newind)

    return pcls(ind_init(c) for c in contents)


def _eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, maxgen,
                    stats=None, halloffame=None,
                    plateau=10, verbose=None):
    '''
    This is the :math:`(\mu + \lambda)` evolutionary algorithm
    from https://github.com/DEAP/deap/blob/04c09bf287256a337bc1be0f87c3eadaefd910ce/deap/algorithms.py  # noqa
    extended with a plateau strategy to stop evolution algorithm.
    '''

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    gen = 1
    plateau_reached_n_times = 0
    plateau_previous = None
    while True and gen < maxgen and plateau_reached_n_times < plateau:
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        best = tools.selBest(population, 1)
        if plateau_previous is not None \
           and plateau_previous.fitness.values == best[0].fitness.values:
            plateau_reached_n_times += 1
        else:
            plateau_reached_n_times = 0
        plateau_previous = best[0]

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        gen += 1

    return population, logbook, gen


def getOptimizedIndividual(expected_calendar,
                           actual_calendar,
                           shape,
                           flatshape,
                           nbclass,
                           verbose=False):
    random.seed(1984)

    creator.create('Fitness', base.Fitness, weights=(-2.0, -1.0))
    creator.create('Individual', np.ndarray, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register('attr_int', random.randint, 0, 1)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_int,
                     n=flatshape)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate',
                     _evalMatching,
                     target=expected_calendar.reshape(shape),
                     reference=actual_calendar,
                     shape=shape,
                     nbclass=nbclass)
    toolbox.register("mate", _cxTwoPoint)
    toolbox.register("mutate",
                     tools.mutUniformInt,
                     low=MIN,
                     up=MAX,
                     indpb=INDPB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENTSIZE)

    toolbox.register('individual_guess', _initIndividual, creator.Individual)
    toolbox.register('population_guess',
                     _initPopulation,
                     list,
                     toolbox.individual_guess,
                     actual=actual_calendar,
                     target=expected_calendar,
                     popsize=POPULATION,
                     size=flatshape)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # pop = toolbox.population(n=POPULATION)
    pop = toolbox.population_guess()
    hof = tools.HallOfFame(1, similar=np.array_equal)

    population, logbook, gen = _eaMuPlusLambda(pop, toolbox,
                                               mu=EVOLALGMU,
                                               lambda_=EVOLALGLAMBDA,
                                               cxpb=CXPB, mutpb=MUTPB,
                                               maxgen=NBGEN,
                                               stats=stats,
                                               halloffame=hof,
                                               plateau=40,
                                               verbose=verbose)

    return hof[0], gen
