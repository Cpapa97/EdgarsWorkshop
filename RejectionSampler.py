import numpy as np

def target_density_function():
    raise NotImplementedError

def sampler_function():
    raise NotImplementedError

def rejection_sampler(density_fnc, sampler_fnc, n_iterations=10000):
    raise NotImplementedError

    return valid_samples, sample_density

valid_samples, sample_density = rejection_sampler(target_density_function, sampler_function)

# plot histogram of the generated samples, compare that against target density function

# plot proportion of rejection as a bar graph (valid samples / n_iterations)