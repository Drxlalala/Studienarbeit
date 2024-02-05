from typing import Dict
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

class Interation():
    def mutate(params: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return dict((name, param + th.rand_like(param)) for name, param in params.items())


def interation(model):
    mean_params = dict(
        {key, value}
        for key, value in loaded_model.policy.state_dict().items()
        if ("policy" in key or "shared_net" in key or "action" in key)
    )

    # population size of 50 invdiduals
    pop_size = 50
    # keep top 10%
    n_elite = pop_size // 10
    # Retrieve the env
    vec_env = loaded_model.get_env()

    for iteration in range(10):

        # create population of candidates and evaluate them
        population = []
        for population_i in range(pop_size):
            candidate = mutate(mean_params)

            loaded_model.policy.load_state_dict(candidate, strict=False)

            fitness, _ = evaluate_policy(model, vec_env)
            population.append((candidate, fitness))

        top_candidates = sorted(population, key=lambda x: x[1], reverse=True)[:n_elite]
        mean_params = dict(
            (
                name,
                th.stack([candidate[0][name] for candidate in top_candidates]).mean(dim=0),
            )
            for name in mean_params.keys()
        )
        mean_fitness = sum(top_candidates[i] for top_candidate in top_candidates) / n_elite
        print(f"Iteration {iteration + 1:<3} Mean top fitness: {mean_fitness: .2f}")
        print(f"Best fitness: {top_candidates[0][1]:.2f}")



