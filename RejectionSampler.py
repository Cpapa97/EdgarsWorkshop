import numpy as np
# import matplotlib.pyplot as plt

def target_density_function(x): # this doesn't need to integrate to less than one because rejection sampling can ignore the normalization constant, but doesn't it need to converge still?
    
    return np.sin(x)**2

# Could use a decorator pattern to send in the proper distribution and still do the proper work beforehand that's relevant to it
def sampler_function(n_samples, sample_distribution='uniform', a=0, b=1):
    """
    n_samples (int): number of samples to be generated, if the number isn't reached it'll add to the amount of iterations, and if it reaches that amount early it'll stop early.

    sample_distribution (str): either uniform or gaussian at this point

    a (int): start point for uniform or mean for gaussian

    b (int): end point for uniform or standard deviation for gaussian

    returns: vector of samples ( and maybe sample density), number of actual iterations (int)
    """
    if sample_distribution == 'uniform':
        bins = n_samples / 10
        samples = np.random.uniform(a, b, n_samples)
        probability = 1 / bins * np.ones(n_samples)
        
    elif sample_distribution == 'gaussian':
        samples = np.random.normal(a, b, n_samples)
        probability = (1 / (np.sqrt(2 * np.pi * b * b))) * np.exp(-1 * (((samples - a)**2) / (2 * b * b)))
        
    else:
        raise NotImplementedError

    return samples, probability
    
def rejection_sampler(density_fnc, sampler_fnc, k=None, n_iterations=10000):
    samp, prob = sampler_fnc(n_iterations, sample_distribution='gaussian', a=0, b=1)

    k = 1.1 # will eventually compute based on largest value automatically
    
    target_probabilities = target_density_function(samp)

    samples_u = np.random.rand(n_iterations) # get samples from uniform

    check_acceptance = target_probabilities / (k * prob) # need a better name

    valid_sample_indices = np.where(samples_u < check_acceptance)[0]
    valid_samples = samp[valid_sample_indices]

    print(valid_samples)

    print(len(samp), len(valid_samples))

    sample_density = None # Need to still figure this out

    return valid_samples, sample_density

valid_samples, _ = rejection_sampler(target_density_function, sampler_function, n_iterations=100)#0000)
'''
x = np.linspace(0, 1, 100)

plt.plot(x, np.sin(x)**2, label='sine')

plt.xlabel('x label')
plt.ylabel('y label')
plt.hist(valid_samples, bins=50)
plt.title("Simple Plot")

plt.legend()

plt.show()
'''
# plot histogram of the generated samples, compare that against target density function

# plot proportion of rejection as a bar graph (valid samples / n_iterations)