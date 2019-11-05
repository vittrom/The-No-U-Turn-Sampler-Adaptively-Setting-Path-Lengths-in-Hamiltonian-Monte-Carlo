"""
This package implements the No-U-Turn Sampler (NUTS) algorithm 6 from the NUTS
paper (Hoffman & Gelman, 2011).

Content
-------

The package mainly contains:
  nuts6                     return samples using the NUTS
  test_nuts6                example usage of this package

and subroutines of nuts6:
  build_tree                the main recursion in NUTS
  find_reasonable_epsilon   Heuristic for choosing an initial value of epsilon
  leapfrog                  Perfom a leapfrog jump in the Hamiltonian space
  stop_criterion            Compute the stop condition in the main loop


A few words about NUTS
----------------------

Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC) is a Markov chain Monte
Carlo (MCMC) algorithm that avoids the random walk behavior and sensitivity to
correlated parameters, biggest weakness of many MCMC methods. Instead, it takes
a series of steps informed by first-order gradient information.

This feature allows it to converge much more quickly to high-dimensional target
distributions compared to simpler methods such as Metropolis, Gibbs sampling
(and derivatives).

However, HMC's performance is highly sensitive to two user-specified
parameters: a step size, and a desired number of steps.  In particular, if the
number of steps is too small then the algorithm will just exhibit random walk
behavior, whereas if it is too large it will waste computations.

Hoffman & Gelman introduced NUTS or the No-U-Turn Sampler, an extension to HMC
that eliminates the need to set a number of steps.  NUTS uses a recursive
algorithm to find likely candidate points that automatically stops when it
starts to double back and retrace its steps.  Empirically, NUTS perform at
least as effciently as and sometimes more effciently than a well tuned standard
HMC method, without requiring user intervention or costly tuning runs.

Moreover, Hoffman & Gelman derived a method for adapting the step size
parameter on the fly based on primal-dual averaging.  NUTS can thus be used
with no hand-tuning at all.

In practice, the implementation still requires a number of steps, a burning
period and a stepsize. However, the stepsize will be optimized during the
burning period, and the final values of all the user-defined values will be
revised by the algorithm.

reference: arXiv:1111.4246
"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
Carlo", Matthew D. Hoffman & Andrew Gelman
"""
import numpy as np
from numpy import log, exp, sqrt
from utils import *
from time import process_time

class NUTS():
    def __init__(self, U, grad_U, start_position, extra_pars, iter):
        self.U = U
        self.grad_U = grad_U
        self.start_position = start_position
        self.iter = iter
        self.dim = len(start_position)
        self.epsilon = None
        self.L = list()
        self.grad_evals = 0
        self.h_bar = 0
        self.mu = self.epsilon_bar = self.gamma = self.t0 = self.kappa = self.delta = self.adapt_epsilon = None
        self.set_dual_averaging_pars(extra_pars)
        self.elapsed_time = 0

    def set_dual_averaging_pars(self, extra_pars):
        self.epsilon_bar = extra_pars["epsilon_bar"]
        self.gamma = extra_pars["gamma"]
        self.t0 = extra_pars["t0"]
        self.kappa = extra_pars["kappa"]
        self.delta = extra_pars["delta"]
        self.adapt_epsilon = extra_pars["adapt_epsilon"]
        if 0 <= extra_pars["adapt_epsilon"] <= 1:
            self.adapt_epsilon = np.int(np.round(self.adapt_epsilon * self.iter))

    def leapfrog(self, theta, r, grad, epsilon, f):
        """ Perfom a leapfrog jump in the Hamiltonian space
        INPUTS
        ------
        theta: ndarray[float, ndim=1]
            initial parameter position

        r: ndarray[float, ndim=1]
            initial momentum

        grad: float
            initial gradient value

        epsilon: float
            step size

        f: callable
            it should return the log probability and gradient evaluated at theta
            logp, grad = f(theta)

        OUTPUTS
        -------
        thetaprime: ndarray[float, ndim=1]
            new parameter position
        rprime: ndarray[float, ndim=1]
            new momentum
        gradprime: float
            new gradient
        logpprime: float
            new lnp
        """
        # make half step in r
        rprime = r + 0.5 * epsilon * grad
        # make new step in theta
        thetaprime = theta + epsilon * rprime
        #compute new gradient
        logpprime, gradprime = f(thetaprime)
        # make half step in r again
        rprime = rprime + 0.5 * epsilon * gradprime
        return thetaprime, rprime, gradprime, logpprime

    def f(self, theta):
        return self.U(theta), self.grad_U(theta)

    def build_tree(self, theta, r, grad, logu, v, j, epsilon, f, joint0):
        if (j == 0):
            # Base case: Take a single leapfrog step in the direction v.
            thetaprime, rprime, gradprime, logpprime = self.leapfrog(theta, r, grad, v * epsilon, f)
            self.grad_evals += 1
            joint = logpprime - 0.5 * np.dot(rprime, rprime.T)
            # Is the new point in the slice?
            nprime = int(logu < joint)
            # Is the simulation wildly inaccurate?
            sprime = int((logu - 1000.) < joint)
            # Set the return values---minus=plus for all things here, since the
            # "tree" is of depth 0.
            thetaminus = thetaprime[:]
            thetaplus = thetaprime[:]
            rminus = rprime[:]
            rplus = rprime[:]
            gradminus = gradprime[:]
            gradplus = gradprime[:]
            # Compute the acceptance probability.
            alphaprime = min(1., np.exp(joint - joint0))
            #alphaprime = min(1., np.exp(logpprime - 0.5 * np.dot(rprime, rprime.T) - joint0))
            nalphaprime = 1
        else:
            # Recursion: Implicitly build the height j-1 left and right subtrees.
            thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, \
                nprime, sprime, alphaprime, nalphaprime = self.build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0)
            # No need to keep going if the stopping criteria were met in the first subtree.
            if (sprime == 1):
                if (v == -1):
                    thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, \
                        nprime2, sprime2, alphaprime2, nalphaprime2 = self.build_tree(thetaminus, rminus, gradminus, logu, v, j - 1, epsilon, f, joint0)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2,\
                        nprime2, sprime2, alphaprime2, nalphaprime2 = self.build_tree(thetaplus, rplus, gradplus, logu, v, j - 1, epsilon, f, joint0)
                # Choose which subtree to propagate a sample up from.
                if np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.)):
                    thetaprime = thetaprime2[:]
                    gradprime = gradprime2[:]
                    logpprime = logpprime2
                # Update the number of valid points.
                nprime = int(nprime) + int(nprime2)
                # Update the stopping criterion.
                sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
                # Update the acceptance probability statistics.
                alphaprime = alphaprime + alphaprime2
                nalphaprime = nalphaprime + nalphaprime2

        return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, \
                    nprime, sprime, alphaprime, nalphaprime

    def find_reasonable_epsilon(self, theta0, grad0, logp0, f):
        """ Heuristic for choosing an initial value of epsilon """
        epsilon = 1.
        r0 = np.random.normal(0., 1., len(theta0))

        # Figure out what direction we should be moving epsilon.
        _, rprime, gradprime, logpprime = self.leapfrog(theta0, r0, grad0, epsilon, f)
        # brutal! This trick make sure the step is not huge leading to infinite
        # values of the likelihood. This could also help to make sure theta stays
        # within the prior domain (if any)
        k = 1.
        while np.isinf(logpprime) or np.isinf(gradprime).any():
            k *= 0.5
            _, rprime, _, logpprime = self.leapfrog(theta0, r0, grad0, epsilon * k, f)

        epsilon = 0.5 * k * epsilon

        # acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
        # a = 2. * float((acceptprob > 0.5)) - 1.
        logacceptprob = logpprime - logp0 - 0.5 * (np.dot(rprime, rprime) - np.dot(r0, r0))
        a = 1. if logacceptprob > np.log(0.5) else -1.
        # Keep moving epsilon in that direction until acceptprob crosses 0.5.
        # while ( (acceptprob ** a) > (2. ** (-a))):
        while a * logacceptprob > -a * np.log(2):
            epsilon = epsilon * (2. ** a)
            _, rprime, _, logpprime = self.leapfrog(theta0, r0, grad0, epsilon, f)
            # acceptprob = np.exp(logpprime - logp0 - 0.5 * ( np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
            logacceptprob = logpprime - logp0 - 0.5 * (np.dot(rprime, rprime) - np.dot(r0, r0))

        # print("find_reasonable_epsilon=", epsilon)

        return epsilon

    def simulate(self):
        f = self.f
        M = self.iter
        Madapt = self.adapt_epsilon
        theta0 = self.start_position
        delta = self.delta

        if len(np.shape(theta0)) > 1:
            raise ValueError('theta0 is expected to be a 1-D array')

        D = len(theta0)
        samples = np.empty((M + Madapt, D), dtype=float)
        lnprob = np.empty(M + Madapt, dtype=float)

        logp, grad = f(theta0)
        samples[0, :] = theta0
        lnprob[0] = logp

        # Choose a reasonable first epsilon by a simple heuristic.
        epsilon = self.find_reasonable_epsilon(theta0, grad, logp, f)

        # Parameters to the dual averaging algorithm.
        gamma = self.gamma
        t0 = self.t0
        kappa = self.kappa
        mu = log(10. * epsilon)

        # Initialize dual averaging algorithm.
        epsilonbar = self.epsilon_bar
        Hbar = self.h_bar

        start_time = process_time()
        for m in range(1, M + Madapt):
            # Resample momenta.
            r0 = np.random.normal(0, 1, D)

            #joint lnp of theta and momentum r
            joint = logp - 0.5 * np.dot(r0, r0.T)

            # Resample u ~ uniform([0, exp(joint)]).
            # Equivalent to (log(u) - joint) ~ exponential(1).
            logu = float(joint - np.random.exponential(1, size=1))

            # if all fails, the next sample will be the previous one
            samples[m, :] = samples[m - 1, :]
            lnprob[m] = lnprob[m - 1]

            # initialize the tree
            thetaminus = samples[m - 1, :]
            thetaplus = samples[m - 1, :]
            rminus = r0[:]
            rplus = r0[:]
            gradminus = grad[:]
            gradplus = grad[:]

            j = 0  # initial heigth j = 0
            n = 1  # Initially the only valid point is the initial point.
            s = 1  # Main loop: will keep going until s == 0.

            while (s == 1):
                # Choose a direction. -1 = backwards, 1 = forwards.
                v = int(2 * (np.random.uniform() < 0.5) - 1)

                # Double the size of the tree.
                if (v == -1):
                    thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime,\
                        nprime, sprime, alpha, nalpha = self.build_tree(thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, \
                        nprime, sprime, alpha, nalpha = self.build_tree(thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint)

                # Use Metropolis-Hastings to decide whether or not to move to a
                # point from the half-tree we just generated.
                _tmp = min(1, float(nprime) / float(n))
                if (sprime == 1) and (np.random.uniform() < _tmp):
                    samples[m, :] = thetaprime[:]
                    lnprob[m] = logpprime
                    logp = logpprime
                    grad = gradprime[:]
                # Update number of valid points we've seen.
                n += nprime
                # Decide if it's time to stop.
                s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
                # Increment depth.
                j += 1

            # Do adaptation of epsilon if we're still doing burn-in.
            eta = 1. / float(m + t0)
            Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
            if (m <= Madapt):
                epsilon = exp(mu - sqrt(m) / gamma * Hbar)
                eta = m ** -kappa
                epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
            else:
                epsilon = epsilonbar

        end_time = process_time()
        self.elapsed_time = end_time - start_time

        samples = samples[Madapt:, :]
        lnprob = lnprob[Madapt:]
        return samples #, lnprob, epsilon