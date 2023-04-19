import logging
from scipy.stats import lognorm, norm, binom, bernoulli, normaltest, chisquare
import copy
import numpy as np
import torch


class Distribution:
    code = 0
    p_value = 0.01  # configured fitness p_value

    def __init__(self) -> None:
        pass

    def pdf(self):
        return None

    def test_fitness(self) -> bool:
        """
        Test fitness
        """
        return False

    def aic():
        return None

    def bic():
        return None

    def select(codes, means, stds, direction):
        result = copy.deepcopy(means)

        # Gaussian
        result[codes == Gaussian.code] = Gaussian.select(
            means=means[codes == Gaussian.code],
            stds=stds[codes == Gaussian.code],
            shape=result[codes == Gaussian.code].shape,
            direction=direction)

        # Log Gaussian
        result[codes == LogGaussian.code] = Gaussian.select(
            means=means[codes == LogGaussian.code],
            stds=stds[codes == LogGaussian.code],
            shape=result[codes == LogGaussian.code].shape,
            direction=direction)
        return result

    def aic(D, data):
        """
        -2log(L) + 2k
        """
        k = 2  # means, stds
        L = D.pdf(data, np.mean(data), np.std(data))
        return -2 * sum(np.log(L)) + 2 * k


class Gaussian(Distribution):
    code = 1

    def pdf(x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    def test_fitness(data):
        # _, p_value = chisquare(np.histogram(data, bins='auto', density=False)[
        #                        0], f_exp=Gaussian.pdf(
        #     np.histogram(data, bins='auto')[1], mu, sigma))
        # Gaussian by default
        return True

    def aic(data):
        return Distribution.aic(Gaussian, data)

    def select(means, stds, shape, direction):
        if direction == -1:
            # [μj +3σj,μj +4σj]
            return ((means + 3 * stds) + torch.rand(shape) *
                    ((means + 4 * stds) - (means + 3 * stds)))
        elif direction == 1:
            # [μj − 4σj,μj − 3σj]
            return ((means - 4 * stds) + torch.rand(shape) * (
                    (means - 3 * stds) - (means - 4 * stds)))


class LogGaussian(Distribution):
    code = 2

    def pdf(x, mu, sigma):
        return norm.logpdf(x, mu, sigma)

    def test_fitness(data, mu, sigma) -> bool:
        _, p_value = chisquare(data, f_exp=LogGaussian.pdf(data, mu, sigma))
        return bool(p_value <= Distribution.p_value)

    def aic(data):
        return Distribution.aic(LogGaussian, data)

    def select(means, stds, shape, direction):
        """
        max = exp(μ + kσ)
        min = exp(μ - kσ)
        k = 2.05 (98%)
        k = 2.33 (99%)
        """
        if direction == -1:
            # [exp(μj + 2.05σj),exp(μj + 2.33σj)]
            return (np.exp(means + 3 * stds) + torch.rand(shape) *
                    (np.exp(means + 4 * stds) - np.exp(means + 3 * stds)))
        elif direction == 1:
            # [exp(μj − 2.33σj),exp(μj − 2.05σj)]
            return (np.exp(means - 4 * stds) + torch.rand(shape) * (
                    np.exp(means - 3 * stds) - np.exp(means - 4 * stds)))


class Binomial(Distribution):
    code = 3

    def pdf(x, n=10, p=0.5):
        return binom.pmf(x, n, p)

    def test_fitness(data, n=10, p=0.5):
        _, p_value = chisquare(np.histogram(data, bins='auto', density=False)[
                               0], f_exp=Binomial.pdf(
            np.histogram(data, bins='auto')[1], n, p))
        return bool(p_value <= Distribution.p_value)


class Bernoulli(Distribution):
    code = 4

    def pdf(x, p=0.5):
        return bernoulli.pmf(x, p)

    def test_fitness(data, p=0.5):
        _, p_value = chisquare(np.histogram(data, bins='auto', density=False)[
                               0], f_exp=Bernoulli.pdf(
            np.histogram(data, bins='auto')[1], p))
        return bool(p_value <= Distribution.p_value)


def get_distribution_model(data):
    mu = np.mean(data)
    sigma = np.std(data)

    gaussianAIC = Gaussian.aic(data)

    try:
        logGaussian = LogGaussian.test_fitness(data, mu, sigma)
    except:
        logGaussian = False

    if logGaussian:
        logGaussianAIC = LogGaussian.aic(data)
        if logGaussianAIC < gaussianAIC:
            return LogGaussian.code
    return Gaussian.code


def get_distribution_statistics(distributions):
    gaussian = sum([(tensor == Gaussian.code).sum().item()
                   for tensor in distributions.values()])
    logGaussian = sum([(tensor == LogGaussian.code).sum().item()
                      for tensor in distributions.values()])

    return {
        "gaussian": gaussian,
        "log_gaussian": logGaussian
    }
