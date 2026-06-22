
import numpy as np


def build_influence_forget_set(
    lt_train_indices : np.ndarray,
    influence_scores : np.ndarray,
    budget           : int
) -> np.ndarray:
    """
    Selects the top-k most influential training samples as the forget set.

    Parameters
    ----------
    lt_train_indices : global training indices of long-tail samples, shape (N,)
    influence_scores : influence score per sample, shape (N,)
    budget           : number of samples to forget (k)

    Returns
    -------
    forget_indices : np.ndarray of global training indices, shape (budget,)
    """
    assert len(lt_train_indices) == len(influence_scores)
    assert budget <= len(lt_train_indices)

    # Highest score = most influential = forget these first
    top_k = np.argsort(influence_scores)[::-1][:budget]
    return lt_train_indices[top_k]


def build_random_forget_set(
    lt_train_indices : np.ndarray,
    budget           : int,
    seed             : int = 42
) -> np.ndarray:
    """
    Selects k random samples from lt_train_indices as the forget set.
    This is the baseline to compare against influence-guided selection.

    Parameters
    ----------
    seed : fixed for reproducibility across experiments
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(lt_train_indices), size=budget, replace=False)
    return lt_train_indices[chosen]