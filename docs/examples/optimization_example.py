from dpf2.optimization import random_pareto_search


def evaluate(params):
    x = params[0]
    yield_val = -((x - 1) ** 2)
    spot = (x - 1) ** 2
    return yield_val, spot


pareto = random_pareto_search(evaluate, {"x": (0.0, 2.0)}, n_samples=200, seed=0)
print("Pareto-optimal samples:", pareto)
