import argparse
import random
import time
import sys

import pymc as pm
import numpy as np
import arviz as az

SPACE = ' '
BAR_MARK = '#'
DELAY = 0.01
NUM_SAMPLES = 1000


def write(string: str) -> None:
    """Utility method to write stdout. Make sure rendering that is not buffered."""
    sys.stdout.write(string)
    sys.stdout.flush()


def ascii_histogram(data, bins=20, width=50, delays=[0]):
    """Render a histogram of the data using ascii art."""

    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)
    bin_width = (max(data) - min(data)) / bins

    for i, count in enumerate(hist):
        bar_width = int(width * count / max_count)   # how many markers
        bin_start = min(data) + i * bin_width   # where does this bin start
        write(f'{bin_start:6.2f} | ')
        time.sleep(DELAY)

        for j in range(bar_width):
            write(BAR_MARK)
            time.sleep(random.choice(delays))

        write(f'{SPACE * (width - bar_width)} | {count:3d}\n')
        time.sleep(DELAY)

    write(f"{SPACE * 7}+{'-' * width}+\n")
    write(f'{SPACE * 7}0{SPACE * (width - 2)}{max_count:4d}\n')
    write(
        f'Min: {min(data):.2f}, Max: {max(data):.2f}, Bin size: {bin_width:.3f}\n'
    )


def generate_delays(
    mu: float = DELAY, sigma: float = 0.05, n: int = 100
) -> np.ndarray:
    """Generate values to use for delay when rendering."""
    # Define a truncated normal distribution
    truncated_normal_dist = pm.TruncatedNormal.dist(
        mu=mu, sigma=sigma, lower=0.0
    )

    return pm.draw(truncated_normal_dist, draws=n)


def main():
    print('\n== A Parade of Distributions ==')

    parser = argparse.ArgumentParser(description='Distribution Visualizer')
    parser.add_argument(
        '--delay',
        type=float,
        default=DELAY,
        help='Delay between characters in the visualization',
    )
    args = parser.parse_args()

    delay = args.delay or DELAY
    # generate random delay values centered around delay
    delays = generate_delays(mu=delay, sigma=delay * 0.5, n=50)

    counter = 1

    for params in ((0.0, 1.0), (0.0, 5.0), (0.0, 0.1)):
        mu, sigma = params[0], params[1]
        norm = pm.Normal.dist(mu=mu, sigma=sigma)
        data = pm.draw(norm, draws=NUM_SAMPLES)
        # Print ASCII histogram
        title = f'({counter}) Normal Distribution'
        print(f'\n{"=" * len(title)}')
        print(f'{title}\nmu {mu} and sigma {sigma}\n')
        ascii_histogram(data, delays=delays)
        counter += 1

    for params in ((2, 1),):
        alpha, beta = params[0], params[1]
        gamma = pm.Gamma.dist(alpha=alpha, beta=beta)
        data = pm.draw(gamma, draws=NUM_SAMPLES)
        # Print ASCII histogram
        title = f'({counter}) Gamma Distribution'
        print(f'\n{"=" * len(title)}')
        print(f'{title}\nalpha {alpha} and beta {beta}\n')
        ascii_histogram(data, delays=delays)
        counter += 1

    mixture = pm.Mixture.dist(
        w=[0.3, 0.7],
        comp_dists=[
            pm.Normal.dist(-1, 1),
            pm.Normal.dist(1, 0.5),
        ],
    )
    data = pm.draw(mixture, draws=NUM_SAMPLES)
    # Print ASCII histogram
    title = f'({counter}) Mixture Distribution'
    print(f'\n{"=" * len(title)}')
    print(title)
    ascii_histogram(data, delays=delays)
    counter += 1

    print('\n')


if __name__ == '__main__':
    main()
