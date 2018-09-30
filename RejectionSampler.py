import numpy as np

def target_density_function(x):

    y = np.log(x) + 1/x**2

    return y

def sampler_function(n_samples_needed, n_iterations): # I need to figure out how this manual documentation syntax works
    """
    n_samples_needed (int): number of samples to be generated, if the number isn't reached it'll add to the amount of iterations, and if it reaches that amount early it'll stop early.

    n_iterations (int): number of iterations currently planned (wait is this even necessary?)

    returns: vector of samples ( and maybe sample density), number of actual iterations (int)
    """
    raise NotImplementedError

    

def rejection_sampler(density_fnc, sampler_fnc, k=None, n_iterations=10000):
    """
    if k not defined, calculate it automatically,
    k also has to be at least greater than the largest value you can get from your target density function
    """
    # return valid_samples, sample_density

    raise NotImplementedError

for i in range(1, 100):
    output = target_density_function(i)
    print(output)

# valid_samples, sample_density = rejection_sampler(target_density_function, sampler_function)

# plot histogram of the generated samples, compare that against target density function

# plot proportion of rejection as a bar graph (valid samples / n_iterations)