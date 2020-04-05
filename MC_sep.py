import typing as t
from functools import partial

import click
import numpy as np

# TODO: what to do with gaps?

# Probability for the binomial distribution
P = 0.5
# Amino acid one-letter mapping to integer classes
AA = {c: i for i, c in enumerate("-ACDEFGHIKLMNPQRSTVWY")}


@click.command()
@click.option('-i', '--inp', required=True, type=click.File(),
              help='path to a file where the first line is a sequence of characters, '
                   'the second line is a true binary labeling, '
                   'and the third line is optional initial (guess) labeling')
@click.option('-I', '--init', is_flag=True, default=False,
              help='a flag whether the third line -- guess labeling -- is to be used')
@click.option('-N', '--steps', type=int, default=10000,
              help='a number of steps to run the algorithm')
@click.option('-T', '--temp', type=float, default=1.0,
              help='unitless temperature factor')
@click.option('-M', '--mut_prop', type=float, default=0.2,
              help='proportion of the labels in the sequence which are allowed to change at each step')
def cli(inp, init, steps, temp, mut_prop):
    """
    The tool runs MC for optimization of the alignment column separation into two subsets with minimal entropy.
    """
    # cli function passes encoded input to `run` and prints the final score of the column
    column, true_labels, init_labels = encode_input(inp, init)
    optimized_labels = run(column, init_labels, steps, temp, mut_prop)
    print(score(true_labels, optimized_labels))


def run(column: np.ndarray, labels: np.ndarray, steps: int, temp: float, mut_prop: float):
    """
    Runs a simple MCMC optimizing distribution of binary labels
    :param column: array of encoded column characters
    :param labels: array of initial binary labels
    :param steps: number of steps
    :param temp: temperature
    :param mut_prop: proportion of characters allowed to mutate
    :return: best solution found during the run
    """
    # number of positions allowed to mutate
    num_mut = int(len(labels) * mut_prop)
    # encapsulate common arguments into step and gain function
    step = partial(flip_labels, num_flip=num_mut)
    gain = partial(entropy_gain, column=column, e_column=entropy(column))
    # setup the simulation
    current = labels.copy()
    best = (current, gain(current))
    # run the simulation
    for s in range(steps):
        proposal = step(current)
        gain_current, gain_proposal = gain(current), gain(proposal)
        p_accept = np.exp(-temp * (gain_proposal - gain_current))
        if np.random.rand() < p_accept:
            current, gain_current = proposal, gain_proposal
            if gain_current > best[1]:
                best = current, gain_current
    return best[0]


def flip_labels(labels: np.ndarray, num_flip: int) -> np.ndarray:
    """
    Flips labels at `num_flip` random positions
    """
    pos = np.random.choice(np.arange(len(labels)), size=num_flip, replace=False)
    lab = labels.copy()
    new = np.logical_not(lab).astype(int)
    lab[pos] = new[pos]
    return lab


def mut_labels(labels: np.ndarray, num_mut: int) -> np.ndarray:
    """
    Randomly changes labels at `num_mut` random positions
    """
    pos = np.random.choice(np.arange(len(labels)), size=num_mut, replace=False)
    labels = labels.copy()
    new = np.random.binomial(1, P, len(labels))
    labels[pos] = new[pos]
    return labels


def entropy(labels: np.ndarray, base: int = 2) -> float:
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def entropy_gain(labels: np.ndarray, column: np.ndarray, e_column: t.Optional[float] = None) -> float:
    p1, p2 = column[labels], column[~labels]
    e_col = e_column or entropy(column)
    return e_col - entropy(p1) - entropy(p2)


def score(labels_original: np.ndarray, labels_optimized: np.ndarray) -> float:
    assert len(labels_optimized) == len(labels_original)
    return round((labels_original == labels_optimized).sum() / len(labels_original), 2)


def encode_input(inp: t.Iterator[str], init: bool = False) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        column = np.array([AA[c] for c in next(inp).rstrip('\n')])
    except KeyError:
        raise ValueError('Some one letter codes in the input column are not allowed')
    try:
        true_labels = np.array(list(map(int, next(inp).rstrip('\n'))))
    except StopIteration:
        raise ValueError('No true labels are found')
    if init:
        try:
            init_labels = np.array(list(map(int, next(inp).rstrip('\n'))))
        except StopIteration:
            raise ValueError('No initial labels are found')
    else:
        init_labels = np.random.binomial(1, P, len(column))

    if not len(column) == len(true_labels) == len(init_labels):
        raise ValueError('Column and labels must be of the same length')
    return column, true_labels, init_labels


if __name__ == '__main__':
    cli()
