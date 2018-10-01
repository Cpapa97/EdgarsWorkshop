import numpy as np
import random
import matplotlib.pyplot as mpl

def uniform(a, b):
    return random.uniform(a, b)

def gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma * sigma))) * np.exp(-1 * (((x - mu)**2) / (2 * sigma * sigma)))

def target_density_function(x): # this doesn't need to integrate to less than one because rejection sampling can ignore the normalization constant, but doesn't it need to converge still?
    
    y = np.log(x) + 1/x**2

    return y

def sampler_function(n_samples_needed, n_iterations, distribution=uniform): # I need to figure out how this manual documentation syntax works
    """
    n_samples_needed (int): number of samples to be generated, if the number isn't reached it'll add to the amount of iterations, and if it reaches that amount early it'll stop early.

    n_iterations (int): number of iterations currently planned (wait is this even necessary?)

    returns: vector of samples ( and maybe sample density), number of actual iterations (int)
    """
    valid_samples = np.array([0, 1, 2])
    
    return valid_samples
    

def rejection_sampler(density_fnc, sampler_fnc, k=None, n_iterations=10000, sample_distribution=uniform): # I could just pass in the distribution functions themselves
    """
    if k not defined, calculate it automatically,
    k also has to be at least greater than the largest value you can get from your target density function
    ??? Is it possible to continually change the k value as you find larger values from the target density function? or would that have to be done for the final time?
    how will changing the k in the middle of the function run affect the outcome?
    the goal is to find random samples and reject all the samples that aren't within our target density, but if the k is too low then moving it up just above the new highest
    point would help towards that goal without messing up the outcome. the outcome I want is a vector of valid samples, so ensuring I get that on both sides should be fine.
    it only matters if a sample is rejected or not rejected, not by how much the fitting random distribution was off from the target density function, k only needs to keep track
    of the largest possible value and be just above that.
    """

    # need to implement distribution I know, uniform or gaussian for example

    valid_samples = sampler_fnc(1000, n_iterations, sample_distribution)

    # generate the sample density from the valid samples vs ...?

    # return valid_samples, sample_density

    return valid_samples, [0, 0.3, 0.7]

for i in range(1, 100):
    output = target_density_function(i)
    print(output)

# valid_samples, sample_density = rejection_sampler(target_density_function, sampler_function)

# plot histogram of the generated samples, compare that against target density function

# plot proportion of rejection as a bar graph (valid samples / n_iterations)