from collections import namedtuple
from typing import Any, Callable, Dict, List, Literal, Union

import numpy as np

from blackboxopt import space
from blackboxopt.algorithms import base

Phenome = namedtuple("Phenome", ['param', 'value'])  # A phenome is a single argument


class Gene:  # A gene is a list of phenomes (arguments) for a given fitness function
    def __init__(self, phenomes: List[Phenome]):
        self._phenomes = phenomes
        self._params = [phenome.param for phenome in self._phenomes]
        self._params_set = set(self._params)
        self._values = [phenome.value for phenome in self._phenomes]
        self._param_dict = {phenome.param: phenome.value for phenome in self._phenomes}
        self._length = len(phenomes)

    def get_fitness(self, func: Callable[..., float]) -> float:
        return func(**self._param_dict)

    def k_point_crossover(self, other: "Gene", k: int):
        assert 0 < k <= self._length
        assert self._params_set == other._params_set

        crossover_points = np.sort(np.random.choice(np.arange(self._length + 1), k, replace=False))
        new_gene_values = self._values[:crossover_points[0]]
        for i, (prev_point, curr_point) in enumerate(zip(crossover_points, crossover_points[1:]), 1):
            if i % 2:
                phenomes = other._values[prev_point:curr_point]
            else:
                phenomes = self._values[prev_point:curr_point]
            new_gene_values.extend(phenomes)

        final_crossover = crossover_points[-1]
        if k % 2:
            remaining_phenomes = other._values[final_crossover:]
        else:
            remaining_phenomes = self._values[final_crossover:]
        new_gene_values.extend(remaining_phenomes)

        new_gene = Gene([Phenome(param, value) for param, value in zip(self._params, new_gene_values)])
        return new_gene

    def single_point_crossover(self, other: "Gene"):
        return self.k_point_crossover(other, 1)

    def two_point_crossover(self, other: "Gene"):
        return self.k_point_crossover(other, 2)

    def uniform_crossover(self, other: "Gene", weight: float = 0.5):
        assert self._params_set == other._params_set

        new_gene_values = self._values
        for i in range(self._length):
            if np.random.random() >= weight:
                new_gene_values[i] = other._values[i]

        new_gene = Gene([Phenome(param, value) for param, value in zip(self._params, new_gene_values)])
        return new_gene

    def mutate(self, space: space.SearchSpace, mutation_probability: float = 0.5):
        mutated_phenomes = self._phenomes
        for i, phenome in enumerate(self._phenomes):
            if np.random.random() >= mutation_probability:
                new_phenome = Phenome(phenome.param, space[phenome.param].sample())
                mutated_phenomes[i] = new_phenome
        return Gene(mutated_phenomes)

    @property
    def params_set(self):
        return self._params_set

    @property
    def param_dict(self):
        return self._param_dict


class Population:  # A population is a list of genes
    def __init__(self, genes: List[Gene]):
        self.size = len(genes)
        assert self.size > 0
        
        self.genes = genes
        self.population_params = self.genes[0].params_set
        assert all(gene.params_set == self.population_params for gene in self.genes)

    def update_population(self, genes: List[Gene]):
        assert len(genes) == self.size, f"The new population size does not match the old. Got {len(genes)} genes, but expected {self.size}"
        assert all(gene.params_set == self.population_params for gene in genes)
        self.genes = genes

    def fittest_gene(self, func: Callable[..., float]) -> Gene:
        best_gene = None
        best_fitness = float('-inf')
        for gene in self.genes:
            fitness = gene.get_fitness(func)
            if fitness > best_fitness:
                best_gene = gene
                best_fitness = fitness
        return best_gene

    def rank_select(self, func: Callable[..., float], k: int) -> np.ndarray:
        fitness = np.array([gene.get_fitness(func) for gene in self.genes], dtype=float)
        top_k = np.array(self.genes, dtype=Gene)[fitness.argsort()[-k:]][::-1]
        return top_k

    def roulette_select(self, func: Callable[..., float], k: int) -> np.ndarray:
        fitness = np.array([gene.get_fitness(func) for gene in self.genes], dtype=float)
        fitness = fitness + fitness.min(initial=0)  # adjust to min of 0, so no gene has a negative probability
        total_fitness = fitness.sum()
        fitness_probability = fitness / total_fitness
        select_k = np.random.choice(self.genes, k, p=fitness_probability)
        return select_k


def genetic_algorithm(
        func: Callable[..., float],
        space: space.SearchSpace,
        maximize: bool = True,
        pop_size: int = 100,
        generations: int = 100,
        crossover_rate: float = 1./3,
        mutation_rate: float = 1./3,
        k_crossover: int = 1,
        select_method: Union[Literal["rank"], Literal["roulette"]] = 'rank'
):
    func, space = base.handle_base_params(func, space, maximize)
    assert crossover_rate + mutation_rate <= 1

    select_k = int(pop_size * (1 - crossover_rate))
    crossover_k = int(pop_size * crossover_rate)
    mutate_k = int(pop_size * mutation_rate)
    survive_k = pop_size - (crossover_k + mutate_k)

    ordered_params = list(space.keys())
    initial_pop = [
        Gene([Phenome(param, space[param].sample()) for param in ordered_params])
        for _ in range(pop_size)
    ]
    population = Population(initial_pop)

    for generation in range(1, generations + 1):
        if select_method == 'rank':
            fit_genes = population.rank_select(func, select_k)
        elif select_method == 'roulette':
            fit_genes = population.roulette_select(func, select_k)
        else:
            raise AttributeError

        crossover_pairs = [np.random.choice(fit_genes, 2, replace=False) for _ in range(crossover_k)]
        crossed_genes = [gene_a.k_point_crossover(gene_b, k_crossover) for gene_a, gene_b in crossover_pairs]
        mutating_genes = np.random.choice(fit_genes, mutate_k, replace=False)
        mutated_genes = [gene.mutate(space) for gene in mutating_genes]
        surviving_genes = np.random.choice(fit_genes, survive_k, replace=False)

        new_genes = crossed_genes + mutated_genes + surviving_genes.tolist()
        population.update_population(new_genes)

    return population.fittest_gene(func).param_dict
