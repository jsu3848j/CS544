import copy
from time import time
from random import Random
import inspyred
import math
    
def generator_schwefel(random, args):
    size = args.get('num_inputs', 2)
    return [random.uniform(-500.0, 500.0) for i in range(size)]
        
def evaluator_schwefel(candidates, args):
    fitness = []
    size = 2
    for c in candidates:
        fitness.append(418.9829 * size - sum([-x * math.sin(math.sqrt(abs(x))) for x in c]))
    return fitness

rand = Random()
rand.seed(int(time()))
size = 2

algorithm = inspyred.ec.EvolutionaryComputation(rand)
algorithm.terminator = inspyred.ec.terminators.evaluation_termination
algorithm.observer = inspyred.ec.observers.file_observer
algorithm.selector = inspyred.ec.selectors.tournament_selection
algorithm.replacer = inspyred.ec.replacers.generational_replacement
algorithm.variator = inspyred.ec.variators.gaussian_mutation

final_pop = algorithm.evolve(generator=generator_schwefel,
                             evaluator=evaluator_schwefel,
                             pop_size=100,
                             maximize=False,
                             bounder=inspyred.ec.Bounder([-500.0] * size, [500.0] * size),
                             num_selected=100,
                             tournament_size=2,
                             mutation_rate=0.25,
                             max_evaluations=2000)

final_pop.sort(reverse=True)
best = final_pop[0]
components = best.candidate
print('\nFittest individual:')
print(best)
