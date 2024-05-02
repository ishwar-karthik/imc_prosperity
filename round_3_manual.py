import numpy as np

multipliers = np.array([
    [24, 70, 41, 21, 60],
    [47, 82, 87, 80, 35],
    [73, 89, 100, 90, 17],
    [77, 83, 85, 79, 55],
    [12, 27, 52, 15, 30]
])

hunters = np.array([
    [2, 4, 3, 2, 4],
    [3, 5, 5, 5, 3],
    [4, 5, 8, 7, 2],
    [5, 5, 5, 5, 4],
    [2, 3, 4, 2, 3]
])

second_expedition = (7500 / 25000) * multipliers - hunters
print(f"second_expedition=\n{second_expedition}")

third_expedition = (7500 / 100000) * multipliers - hunters
print(f"third_expedition=\n{third_expedition}")

# Start with uniform distribution
SHAPE = [5, 5]
dist_init = np.ones(SHAPE) / 25

def find_best_location(dist: np.ndarray) -> [int, int]:
    profits = 7500 * multipliers / (hunters + dist)
    # print(f"profits = {profits}")
    amax = np.unravel_index(np.argmax(profits, axis=None), profits.shape)
    return amax  # Change this
# print(find_best_location(dist_init))

def compute_new_dist(dist: np.ndarray, rationality: float) -> np.ndarray:
    best_array = np.zeros(SHAPE)
    best = find_best_location(dist)
    best_array[best[0]][best[1]] = 1
    # print(best_array)
    return dist * (1 - rationality) + best_array * rationality

# new = compute_new_dist(dist_init, 0.1)
# print(compute_new_dist(dist_init, 0.1))
# assert (np.sum(np.sum(new)) == 1)

n_iterations = 100
rationality = 0.15
display = [1, 5, 10, 20, 50, 100]
num = 0
for i in range(n_iterations):
    dist_init = compute_new_dist(dist_init, rationality)
    num += 1
    if num in display:
        print(f"After {num} iterations:\n{dist_init.round(decimals=4)}")
