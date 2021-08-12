"""The evolutionary module is responsible for implementing optimization algorithms with relationship to Genetics,
genetic algorithms. Currently includes base genetic algorithm, simmulated annealing, stochastic hill climbing,
and random restart hill climbing"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import numpy as np
from tqdm import tqdm, trange

from blackboxopt import space as sp
from blackboxopt.algorithms import base

# pylint: disable=R0913


@dataclass
class Phenome:
    """Holds the name and value for a given sample of a parameter"""
    param: str
    value: Any


class Gene:
    """A gene is a list of phenomes (arguments) for a given fitness function"""

    def __init__(self, phenomes: List[Phenome]):
        """
        Args:
            phenomes: List of phenomes (parameter values) that form the Gene
        """
        self.phenomes = phenomes
        self.values = [phenome.value for phenome in self.phenomes]
        self.length = len(phenomes)
        self._params = [phenome.param for phenome in self.phenomes]
        self._params_set = set(self._params)
        self._param_dict = {phenome.param: phenome.value for phenome in self.phenomes}

    @property
    def params_set(self):
        """stores the set of parameters to speed up computation in some methods"""
        return self._params_set

    @property
    def param_dict(self):
        """stores the dictionary of parameters to speed up computation in some methods"""
        return self._param_dict

    def get_fitness(self, func: Callable[..., float]) -> float:
        """Calculates the score of the Gene given a fitness function

        Args:
            func: fitness function that takes as parameters the Gene's Phenome values to evaluate Gene's parameter score

        Returns:
            float representing the score evaluated by the fitness function
        """
        return func(**self.param_dict)

    def k_point_crossover(self, other: "Gene", k: int) -> "Gene":
        """Performs k point crossover between the current gene and the provided gene, returning an offspring

        Splits at k sequential points, returning a "zipped" Gene that results from the splits

        Args:
            other: The other Gene to crossover with
            k: The number of splits in the crossover

        Returns:
           Offspring from crossover between Genes
        """
        assert 0 < k <= self.length, f"Cannot have more splits than there are Phenomes" \
                                      f" in Gene, total Phenomes: {self.length}, got {k=}"
        assert self.params_set == other.params_set, f"Phenomes in Genes do not match, cannot perform crossover.\n" \
                                                      f"Got Current Phenomes: {self.params_set}\n" \
                                                      f"And Other Phenomes: {other.params_set}"

        # Choose k sorted points from the range of values
        crossover_points = np.sort(
            sp.rng.choice(np.arange(self.length + 1), k, replace=False)
        )
        # Start with initial split
        new_gene_values = self.values[: crossover_points[0]]
        # For each split in the crossover, switch between Genes to select split from, and extend offspring Gene
        for i, (prev_point, curr_point) in enumerate(
            zip(crossover_points, crossover_points[1:]), 1
        ):
            # Get split
            if i % 2:
                phenomes = other.values[prev_point:curr_point]
            else:
                phenomes = self.values[prev_point:curr_point]
            new_gene_values.extend(phenomes)

        # Get the remaining phenomes and add to offspring
        final_crossover = crossover_points[-1]
        if k % 2:
            remaining_phenomes = other.values[final_crossover:]
        else:
            remaining_phenomes = self.values[final_crossover:]
        new_gene_values.extend(remaining_phenomes)

        # Create and return new Gene
        new_gene = Gene(
            [
                Phenome(param, value)
                for param, value in zip(self._params, new_gene_values)
            ]
        )
        return new_gene

    def single_point_crossover(self, other: "Gene") -> "Gene":
        """Performs single point crossover between the current gene and the provided gene, returning an offspring

        Args:
            other: The other Gene to crossover with

        Returns:
            Offspring from crossover between Genes
        """
        return self.k_point_crossover(other, 1)

    def two_point_crossover(self, other: "Gene") -> "Gene":
        """Performs single point crossover between the current gene and the provided gene, returning an offspring

        Args:
            other: The other Gene to crossover with

        Returns:
            Offspring from crossover between Genes
        """
        return self.k_point_crossover(other, 2)

    def uniform_crossover(self, other: "Gene", weight: float = 0.5) -> "Gene":
        """Performs crossover between Genes were the probability of taking the Phenome from either Gene is constant. By
        default this is 0.5, 50%, probability of either gene, resulting in a uniform crossover

        Random number, n, is generated for each Phenome and if n < weight, the offspring takes the current gene's
        Phenome, and vice versa for n >= weight

        Args:
            other: The other Gene to crossover with
            weight: The probability of taking each Phenome from the current Gene

        Returns:
            Offspring from crossover between Genes
        """
        assert self.params_set == other.params_set

        offspring_phenomes = self.phenomes[:]  # Store offspring phenomes, starting with current Gene's phenomes
        for i in range(self.length):
            if sp.rng.random() >= weight:
                # Whenever the generated number is greater than the weight, replace value with the other Gene's vlaue
                offspring_phenomes[i].value = other.phenomes[i].value

        # Create and return offspring gene
        return Gene(offspring_phenomes)

    def mutate(self, sampler: sp.SearchSpaceSampler, mutation_probability: float = 0.5) -> "Gene":
        """Mutates each Phenome selected for mutation by sampling a new value for the parameter

        Iterates through the Gene's Phenomes and generates n in [0, 1] where if n < mutation_probability, that Phenome's
        parameter is resampled (mutated)

        Args:
            sampler: SearchSpaceSampler for resampling the Phenome's parameters
            mutation_probability: The probability of each Phenome being mutated

        Returns:
            A Gene constructed with the mutated Phenomes
        """
        mutated_phenomes = self.phenomes[:]  # Store mutated phenomes, which starts as current gene's Phenomes
        for i, phenome in enumerate(self.phenomes):
            if sp.rng.random() < mutation_probability:
                # When the generated number is less than the mutation probability, mutate the Phenome at curr iteration
                new_val = sampler.resample_one(phenome.param)
                new_phenome = Phenome(phenome.param, new_val)
                mutated_phenomes[i] = new_phenome
        # Create and return mutated Gene
        return Gene(mutated_phenomes)

    def mutate_one(self, sampler: sp.SearchSpaceSampler) -> "Gene":
        """Mutates a single Phenome in the Gene, chosen at random, and returns a new mutated Gene

        Args:
            sampler: SearchSpaceSampler for resampling the Phenome's parameters

        Returns:
            A Gene constructed with the current Gene's Phenomes, where one Phenome was mutated
        """
        mutated_phenomes = self.phenomes[:]  # Store offspring phenomes
        phenome_to_mutate = sp.rng.integers(0, self.length - 1)  # Generate Phenome index to mutate
        mutated_param = mutated_phenomes[phenome_to_mutate].param
        mutated_value = sampler.resample_one(mutated_param)  # Get new parameter value
        mutated_phenomes[phenome_to_mutate] = Phenome(mutated_param, mutated_value)
        # Return new Gene where a single phenome was mutated
        return Gene(mutated_phenomes)

    def __str__(self):
        """Creates a string from the phenome's parameter, value dictionary"""
        return str(self.param_dict)

    def __repr__(self):
        """Creates string for rebuilding the Gene object"""
        return f"Gene([{', '.join(repr(phenome) for phenome in self.phenomes)}])"

    def __eq__(self, other):
        """Determines if Genes are equivalent by comparing their types, parameter names, and individual Phenomes"""
        return (
            isinstance(other, Gene) and
            self.params_set == other.params_set and
            all(self.param_dict[p] == other.param_dict[p] for p in self.params_set)
        )


class Population:  # A population is a list of genes
    """A group of Genes that represent a population of an evolutionary cycle"""

    def __init__(self, genes: List[Gene]):
        """
        Args:
            genes: List of Genes that form a population
        """
        self.size = len(genes)
        assert self.size > 0

        self.genes = genes
        self.population_params = self.genes[0].params_set
        assert all(gene.params_set == self.population_params for gene in self.genes)

    def update_population(self, genes: List[Gene]):
        """Updates the current population by replace the list of Genes

        Performs validation on the new genes to ensure the Population is being updated rather than altered

        Args:
            genes: List of Genes representing the updated population
        """
        assert (
            len(genes) == self.size
        ), f"The new population size does not match the old. Got {len(genes)} genes, but expected {self.size}"
        assert all(gene.params_set == self.population_params for gene in genes)
        self.genes = genes

    def fittest_gene(self, func: Callable[..., float]) -> Gene:
        """Gets the fittest Gene by evaluating all Genes in the population with the provided reward function

        Args:
            func: The reward function used to evaluate the Genes' fitness, must accept the Gene's Phenomes as inputs

        Returns:
            Gene that obtained the highest fitness
        """
        # Theta(n) search through all Genes to find fittest
        best_gene = None
        best_fitness = float("-inf")
        for gene in self.genes:
            fitness = gene.get_fitness(func)
            if fitness > best_fitness:
                best_gene = gene
                best_fitness = fitness
        return best_gene

    def rank_select(self, func: Callable[..., float], k: int) -> np.ndarray:
        """Selects the k top Genes from the population based on their fitness value obtained from the reward function

        OUTPUT GENES MUST BE SORTED IN DESCENDING ORDER LIKE RANK SELECT OR THE ELITIST SELECTION WILL FAIL

        Args:
            func: The reward function used to evaluate the Genes' fitness, must accept the Gene's Phenomes as inputs
            k: The number of top Genes to select

        Returns:
            An np.ndarray of the top k Genes sorted by their evaluated fitness (descending order)
        """
        assert k <= self.size, f"Cannot rank select {k=} Genes when the population size is {self.size}"
        fitness = np.array([gene.get_fitness(func) for gene in self.genes], dtype=float)
        top_k = np.array(self.genes, dtype=Gene)[fitness.argsort()[-k:]][::-1]
        return top_k

    def roulette_select(self, func: Callable[..., float], k: int) -> np.ndarray:
        """Selects k Genes from population roulette style, in which each Gene has a probability of being chosen
        proportional to its relative fitness score among the Gene population

        OUTPUT GENES MUST BE SORTED IN DESCENDING ORDER LIKE RANK SELECT OR THE ELITIST SELECTION WILL FAIL

        Args:
            func: The reward function used to evaluate the Genes' fitness, must accept the Gene's Phenomes as inputs
            k: The number of Genes to select

        Returns:
            An np.ndarray of the k selected Genes sorted by their evaluated fitness (descending order)
        """
        assert k <= self.size, f"Cannot roulette select {k=} Genes when the population size is {self.size}"
        fitness = np.array([gene.get_fitness(func) for gene in self.genes], dtype=float)
        fitness_dict = dict(
            zip(self.genes, fitness)
        )  # Need to store dict for sorting at the end
        fitness = fitness + fitness.min()  # adjust to min of 0, so no gene has a negative probability
        total_fitness = fitness.sum()
        fitness_probability = fitness / total_fitness
        select_k = sp.rng.choice(self.genes, k, p=fitness_probability)
        # sort output list for elitist selection
        select_k = sorted(select_k, key=lambda gene: fitness_dict[gene], reverse=True)
        return np.array(select_k)

    def __str__(self):
        """Brief description of the Population"""
        return f"Population of size: {self.size} with genes: {[str(gene) for gene in self.genes]}"

    def __repr__(self):
        """Creates string for rebuilding the Population object"""
        return f"Population([{', '.join(repr(gene) for gene in self.genes)}])"


def genetic_algorithm(
    func: Callable[..., float],
    sampler: sp.SearchSpaceSampler,
    maximize: bool = True,
    pop_size: int = 100,
    generations: int = 100,
    purge_rate: float = 1.0 / 3,
    crossover_rate: float = 1.0 / 3,
    mutation_rate: float = 1.0 / 3,
    mutation_probability: float = 0.5,
    elitist_rate: float = 0.0,
    k_crossover: int = 1,
    select_method: Union[Literal["rank"], Literal["roulette"]] = "rank",
    progress: bool = True,
) -> Dict[str, Any]:
    """

    Args:
        func:
        sampler:
        maximize:
        pop_size:
        generations:
        purge_rate:
        crossover_rate:
        mutation_rate:
        mutation_probability:
        elitist_rate:
        k_crossover:
        select_method:
        progress:

    Returns:

    """
    func, _ = base.handle_base_params(func, sampler, maximize)
    assert crossover_rate + mutation_rate + elitist_rate <= 1

    select_k = int(pop_size * (1 - purge_rate))
    crossover_k = int(pop_size * crossover_rate)
    mutate_k = int(pop_size * mutation_rate)
    elite_k = int(pop_size * elitist_rate)
    survive_k = pop_size - (crossover_k + mutate_k + elite_k)

    initial_pop = [
        Gene([Phenome(param, sample) for param, sample in sampler.sample().items()])
        for _ in range(pop_size)
    ]
    population = Population(initial_pop)

    for generation in (
        pbar := tqdm(range(1, generations + 1), disable=not progress, total=generations)
    ):
        pbar.set_description(f"Generation {generation}")
        if select_method == "rank":
            fit_genes = population.rank_select(func, select_k)
        elif select_method == "roulette":
            fit_genes = population.roulette_select(func, select_k)
        else:
            raise AttributeError

        crossover_pairs = [
            sp.rng.choice(fit_genes, 2, replace=False) for _ in range(crossover_k)
        ]
        crossed_genes = [
            gene_a.k_point_crossover(gene_b, k_crossover)
            for gene_a, gene_b in crossover_pairs
        ]
        mutating_genes = sp.rng.choice(fit_genes, mutate_k, replace=False)
        mutated_genes = [
            gene.mutate(sampler, mutation_probability) for gene in mutating_genes
        ]
        elite_genes = fit_genes[:elite_k]
        surviving_genes = sp.rng.choice(fit_genes, survive_k, replace=False)

        new_genes = (
            crossed_genes +
            mutated_genes +
            elite_genes.tolist() +
            surviving_genes.tolist()
        )
        population.update_population(new_genes)

    return population.fittest_gene(func).param_dict


class CoolingSchedule(ABC):
    """"""

    def __init__(self, initial_temperature: float, steps: int, **kwargs):
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.steps = steps
        self.step_count = 0

    @abstractmethod
    def step(self) -> float:
        """

        Returns:

        """
        raise NotImplementedError

    def __next__(self):
        """

        Returns:

        """
        if self.step_count == 0:
            self.step_count += 1
            return self.temperature
        if self.step_count >= self.steps:
            self.step_count = 0
            self.temperature = self.initial_temperature
            raise StopIteration
        return self.step()

    def __iter__(self):
        """"""
        return self


class LinearCoolingSchedule(CoolingSchedule):
    """"""

    def __init__(self, initial_temperature: float, steps: int):
        super().__init__(initial_temperature, steps)
        self.decay_rate = initial_temperature / (self.steps - 1)

    def step(self) -> float:
        """

        Returns:

        """
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature -= self.decay_rate
        self.step_count += 1
        return self.temperature


class MultiplicativeCoolingSchedule(CoolingSchedule, ABC):
    """"""

    def __init__(
        self, initial_temperature: float, steps: int, alpha: Optional[float] = None
    ):
        super().__init__(initial_temperature, steps)
        self.alpha = self.handle_alpha(alpha)

    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        """"""
        raise NotImplementedError


class ExponentialMultiplicativeCoolingSchedule(MultiplicativeCoolingSchedule):
    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        """"""
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
        """"""
        if alpha is None:
            return 10
        return alpha

    def step(self) -> float:
        """

        Returns:

        """
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature = self.initial_temperature / (
            1 + self.alpha * np.log(1 + self.step_count)
        )
        self.step_count += 1
        return self.temperature


class LinearMultiplicativeCoolingSchedule(MultiplicativeCoolingSchedule):
    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        """"""
        if alpha is None:
            return 1
        return alpha

    def step(self) -> float:
        """

        Returns:

        """
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature = self.initial_temperature / (1 + self.alpha * self.step_count)
        self.step_count += 1
        return self.temperature


class QuadraticMultiplicativeCoolingSchedule(MultiplicativeCoolingSchedule):
    """"""

    @staticmethod
    def handle_alpha(alpha: Optional[float]) -> float:
        """"""
        if alpha is None:
            return 2
        return alpha

    def step(self) -> float:
        """

        Returns:

        """
        if self.step_count >= self.steps:
            raise ValueError("Attempted to step past the max steps set by the instance")
        self.temperature = self.initial_temperature / (
            1 + self.alpha * (self.step_count ** 2)
        )
        return self.temperature


COOLING_SCHEDULE_DICT: Dict[str, Type[CoolingSchedule]] = {
    "linear": LinearCoolingSchedule,
    "exponential_multiplicative": ExponentialMultiplicativeCoolingSchedule,
    "log_multiplicative": LogMultiplicativeCoolingSchedule,
    "linear_multiplicative": LinearMultiplicativeCoolingSchedule,
    "quadratic_multiplicative": QuadraticMultiplicativeCoolingSchedule,
}


def kirkpatrick_acceptance(
    old_energy: float, new_energy: float, temperature: float
) -> float:
    """

    Args:
        old_energy:
        new_energy:
        temperature:

    Returns:

    """
    if new_energy > old_energy:
        return 1.0
    return np.exp(-(old_energy - new_energy) / temperature)


def simulated_annealing_algorithm(
    func: Callable[..., float],
    sampler: sp.SearchSpaceSampler,
    maximize: bool = True,
    initial_temperature: float = 100.0,
    steps: int = 1000,
    cooling_schedule: str = "linear",
    acceptance_probability_func: Callable[
        [float, float, float], float
    ] = kirkpatrick_acceptance,
    **cooling_schedule_kwargs,
) -> Dict[str, Any]:
    """

    Args:
        func:
        sampler:
        maximize:
        initial_temperature:
        steps:
        cooling_schedule:
        acceptance_probability_func:
        **cooling_schedule_kwargs:

    Returns:

    """
    func, _ = base.handle_base_params(func, sampler, maximize)
    cooling_schedule = COOLING_SCHEDULE_DICT[cooling_schedule](
        initial_temperature, steps, **cooling_schedule_kwargs
    )

    gene = Gene([Phenome(param, val) for param, val in sampler.sample().items()])
    energy = gene.get_fitness(func)
    for temperature in cooling_schedule:
        new_gene = gene.mutate_one(sampler)
        new_energy = new_gene.get_fitness(func)

        if new_energy > energy:
            gene, energy = new_gene, new_energy
        elif acceptance_probability_func(energy, new_energy, temperature) > sp.rng.random():
            gene, energy = new_gene, new_energy

    return gene.param_dict


def stochastic_hill_climbing(
    func: Callable[..., float],
    sampler: sp.SearchSpaceSampler,
    maximize: bool = True,
    steps: int = 1000,
) -> Dict[str, Any]:
    func, _ = base.handle_base_params(func, sampler, maximize)

    gene = Gene([Phenome(param, val) for param, val in sampler.sample().items()])
    score = gene.get_fitness(func)
    for _ in trange(steps):
        new_gene = gene.mutate(sampler)
        new_score = new_gene.get_fitness(func)
        if new_score > score:
            gene = new_gene

    return gene.param_dict


def random_restart_hill_climbing(
    func: Callable[..., float],
    sampler: sp.SearchSpace,
    maximize: bool = True,
    steps: int = 1000,
    restart_probability: float = 0.05,
) -> Dict[str, Any]:
    """

    Args:
        func:
        sampler:
        maximize:
        steps:
        restart_probability:

    Returns:

    """
    func, _ = base.handle_base_params(func, sampler, maximize)

    curr_gene = Gene([Phenome(param, val) for param, val in sampler.sample().items()])
    best_gene = curr_gene
    curr_score = curr_gene.get_fitness(func)
    best_score = curr_score
    for _ in range(steps):
        if sp.rng.random() <= restart_probability:
            if curr_score > best_score:
                best_gene = curr_gene
                best_score = curr_score
            curr_gene = Gene(
                [Phenome(param, val) for param, val in sampler.sample().items()]
            )
        else:
            new_gene = curr_gene.mutate(sampler)
            new_score = new_gene.get_fitness(func)
            if new_score > curr_score:
                curr_gene = new_gene
                curr_score = new_score

    return best_gene.param_dict
