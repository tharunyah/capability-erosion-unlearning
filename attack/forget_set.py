import numpy as np


def build_influence_forget_set(
    lt_train_indices : np.ndarray,
    influence_scores : np.ndarray,
    budget           : int
) -> np.ndarray:
    assert len(lt_train_indices) == len(influence_scores)
    assert budget <= len(lt_train_indices)
    top_k = np.argsort(influence_scores)[::-1][:budget]
    return lt_train_indices[top_k]


def build_random_forget_set(
    lt_train_indices : np.ndarray,
    budget           : int,
    seed             : int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(lt_train_indices), size=budget, replace=False)
    return lt_train_indices[chosen]


if __name__ == "__main__":
    lt_train_indices = np.load("data/lt_train_indices.npy")
    influence_scores = np.load("data/influence_scores.npy")

    print(f"Loaded {len(lt_train_indices)} long-tail indices, {len(influence_scores)} influence scores")

    for budget in [50, 100, 200]:
        inf_set = build_influence_forget_set(lt_train_indices, influence_scores, budget)
        rand_set = build_random_forget_set(lt_train_indices, budget)

        np.save(f"data/forget_influence_{budget}.npy", inf_set)
        np.save(f"data/forget_random_{budget}.npy", rand_set)

        print(f"Budget {budget}: influence set [{inf_set.min()}, {inf_set.max()}] | random set [{rand_set.min()}, {rand_set.max()}]")

    print("Done. 6 forget sets saved to data/")