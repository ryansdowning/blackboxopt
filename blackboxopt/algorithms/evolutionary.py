from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Callable, Dict, List, Literal, Union, Type, Optional

import numpy as np
from tqdm import tqdm

from blackboxopt import space as sp
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

    @property
    def params_set(self):
        return self._params_set

    @property
    def param_dict(self):
        return self._param_dict

    def get_fitness(self, func: Callable[..., float]) -> float:
        return func(**self._param_dict)

    def k_point_crossover(self, other: "Gene", k: int):
        assert 0 < k <= self._length
        assert self._params_set == other._params_set

        crossover_points = np.sort(sp.rng.choice(np.arange(self._length + 1), k, replace=False))
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
            if sp.rng.random() >= weight:
                new_gene_values[i] = other._values[i]

        new_gene = Gene([Phenome(param, value) for param, value in zip(self._params, new_gene_values)])
        return new_gene

    def mutate(self, space: sp.SearchSpace, mutation_probability: float = 0.5):
        mutated_phenomes = self._phenomes[:]
        for i, phenome in enumerate(self._phenomes):
            if sp.rng.random() >= mutation_probability:
                new_phenome = Phenome(phenome.param, space[phenome.param].sample())
                mutated_phenomes[i] = new_phenome
        return Gene(mutated_phenomes)

    def mutate_one(self, space: sp.SearchSpace):
        mutated_phenomes = self._phenomes[:]
        phenome_to_mutate = sp.rng.integers(0, self._length - 1)
        mutated_param = mutated_phenomes[phenome_to_mutate].param
        mutated_value = space[mutated_param].sample()
        mutated_phenomes[phenome_to_mutate] = Phenome(mutated_param, mutated_value)
        return Gene(mutated_phenomes)

    def __str__(self):
        return str(self._param_dict)


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
        """OUTPUT GENES MUST BE SORTED IN DESCENDING ORDER LIKE RANK SELECT OR THE ELITIST SELECTION WILL FAIL"""
        fitness = np.array([gene.get_fitness(func) for gene in self.genes], dtype=float)
        fitness_dict = dict(zip(self.genes, fitness))  # Need to store dict for sorting at the end
        fitness = fitness + fitness.min(initial=0)  # adjust to min of 0, so no gene has a negative probability
        total_fitness = fitness.sum()
        fitness_probability = fitness / total_fitness
        select_k = sp.rng.choice(self.genes, k, p=fitness_probability)
        select_k = sorted(select_k, key=lambda gene: fitness_dict[gene], reverse=True)
        return np.array(select_k)

    def __str__(self):
        return f"Population of size: {self.size} with genes: {[str(gene) for gene in self.genes]}"


def genetic_algorithm(
        func: Callable[..., float],
        space: sp.SearchSpace,
        maximize: bool = True,
        pop_size: int = 100,
        generations: int = 100,
        purge_rate: float = 1./3,
        crossover_rate: float = 1./3,
        mutation_rate: float = 1./3,
        mutation_probability: float = 0.5,
        elitist_rate: float = 0.,
        k_crossover: int = 1,
        select_method: Union[Literal["rank"], Literal["roulette"]] = 'rank',
        progress: bool = True
) -> Dict[str, Any]:
    func, space = base.handle_base_params(func, space, maximize)
    assert crossover_rate + mutation_rate + elitist_rate <= 1

    select_k = int(pop_size * (1 - purge_rate))
    crossover_k = int(pop_size * crossover_rate)
    mutate_k = int(pop_size * mutation_rate)
    elite_k = int(pop_size * elitist_rate)
    survive_k = pop_size - (crossover_k + mutate_k + elite_k)

    ordered_params = list(space.keys())
    initial_pop = [
        Gene([Phenome(param, space[param].sample()) for param in ordered_params])
        for _ in range(pop_size)
    ]
    population = Population(initial_pop)

    for generation in tqdm(range(1, generations + 1), disable=not progress, total=generations, desc="Generation"):
        if select_method == 'rank':
            fit_genes = population.rank_select(func, select_k)
        elif select_method == 'roulette':
            fit_genes = population.roulette_select(func, select_k)
        else:
            raise AttributeError

        crossover_pairs = [sp.rng.choice(fit_genes, 2, replace=False) for _ in range(crossover_k)]
        crossed_genes = [gene_a.k_point_crossover(gene_b, k_crossover) for gene_a, gene_b in crossover_pairs]
        mutating_genes = sp.rng.choice(fit_genes, mutate_k, replace=False)
        mutated_genes = [gene.mutate(space, mutation_probability) for gene in mutating_genes]
        elite_genes = fit_genes[:elite_k]
        surviving_genes = sp.rng.choice(fit_genes, survive_k, replace=False)

        new_genes = crossed_genes + mutated_genes + elite_genes.tolist() + surviving_genes.tolist()
        population.update_population(new_genes)

    return population.fittest_gene(func).param_dict


class CoolingSchedule(ABC):
    def __init__(self, initial_temperature: float, steps: int, **kwargs):
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.steps = steps
        self.step_count = 0

    @abstractmethod
    def step(self) -> float:
        raise NotImplementedError

    def __next__(self):
        if self.step_count == 0:
            self.step_count += 1
            return self.temperature
        elif self.step_count >= self.steps:
            self.step_count = 0
            self.temperature = self.initial_temperature
            raise StopIteration
        return self.step()

    def __iter__(self):
        return self


class LinearCoolingSchedule(CoolingSchedule):
    def __init__(self, initial_temperature: float, steps: int):
        super().__init__(initial_temperature, steps)
        self.decay_rate = initial_temperature / (self.steps - 1)

    def step(self) -> float:
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature -= self.decay_rate
        self.step_count += 1
        return self.temperature


class MultiplicativeCoolingSchedule(CoolingSchedule, ABC):
    def __init__(self, initial_temperature: float, steps: int, alpha: Optional[float] = None):
        super().__init__(initial_temperature, steps)
        self.alpha = self.handle_alpha(alpha)

    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        raise NotImplementedError


class ExponentialMultiplicativeCoolingSchedule(MultiplicativeCoolingSchedule):
    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        if alpha is None:
            return 0.85
        return alpha

    def step(self) -> float:
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature = self.initial_temperature * (self.alpha ** self.step_count)
        self.step_count += 1
        return self.temperature


class LogMultiplicativeCoolingSchedule(MultiplicativeCoolingSchedule):
    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        if alpha is None:
            return 10
        return alpha

    def step(self) -> float:
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature = self.initial_temperature / (1 + self.alpha * np.log(1 + self.step_count))
        self.step_count += 1
        return self.temperature


class LinearMultiplicativeCoolingSchedule(MultiplicativeCoolingSchedule):
    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        if alpha is None:
            return 1
        return alpha

    def step(self) -> float:
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature = self.initial_temperature / (1 + self.alpha * self.step_count)
        self.step_count += 1
        return self.temperature


class QuadraticMultiplicativeCoolingSchedule(MultiplicativeCoolingSchedule):
    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        if alpha is None:
            return 2
        return alpha

    def step(self) -> float:
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature = self.initial_temperature / (1 + self.alpha * (self.step_count ** 2))
        return self.temperature


COOLING_SCHEDULE_DICT: Dict[str, Type[CoolingSchedule]] = {'linear': LinearCoolingSchedule}


def kirkpatrick_acceptance(old_energy: float, new_energy: float, temperature: float) -> float:
    if new_energy > old_energy:
        return 1.
    return np.exp(-(old_energy - new_energy) / temperature)


def simulated_annealing_algorithm(
        func: Callable[..., float],
        space: sp.SearchSpace,
        maximize: bool = True,
        initial_temperature: float = 100.,
        steps: int = 1000,
        cooling_schedule: Union[Literal["linear"]] = 'linear',
        acceptance_probability_func: Callable[[float, float, float], float] = kirkpatrick_acceptance
) -> Dict[str, Any]:
    func, space = base.handle_base_params(func, space, maximize)
    cooling_schedule = COOLING_SCHEDULE_DICT[cooling_schedule](initial_temperature, steps)

    gene = Gene([Phenome(param, space[param].sample()) for param in space.keys()])
    energy = gene.get_fitness(func)
    for temperature, step in zip(cooling_schedule, range(steps)):
        new_gene = gene.mutate_one(space)
        new_energy = new_gene.get_fitness(func)

        if new_energy > energy:
            gene, energy = new_gene, new_energy
        elif acceptance_probability_func(energy, new_energy, temperature) > sp.rng.random():
            gene, energy = new_gene, new_energy

    return gene.param_dict
