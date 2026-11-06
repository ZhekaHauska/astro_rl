"""Build model objects from already resolved experiment configuration."""

from astrorl.population import PopulationLearner
from astrorl.adaptive import AdaptiveAstroLearner
from astrorl.local_memory import PopulationConfig
from astrorl.q_learning import QLearner


def build_model(task):
    profile = task["model_config"]
    hp = task["hp"]
    rl = task["group"] == "rl"
    architecture = profile["architecture"]
    if profile["learner"] == "q":
        return QLearner(profile["kind"], rl, task["seed"], hp, task["map"], architecture)
    args = (profile["kind"], rl, task["seed"], hp, architecture, profile["memory_base"])
    if profile["learner"] == "adaptive":
        population = dict(profile["population"])
        population.setdefault("tau", hp["memory_tau"])
        return AdaptiveAstroLearner(
            *args, population_config=PopulationConfig(**population), motor_options=profile["motor"]
        )
    return PopulationLearner(*args, memory_options=profile["memory"], motor_options=profile["motor"])
