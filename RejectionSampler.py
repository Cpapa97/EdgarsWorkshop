import numpy as np
import random
import matplotlib.pyplot as plt

# use np.ra

# def uniform(a, b):
#     return random.uniform(a, b)

# def gaussian(x, mu, sigma): # I NEED TO PASS THIS IN ANOTHER WAY, so that the implementation of any distribution is the same
#     return (1 / (np.sqrt(2 * np.pi * sigma * sigma))) * np.exp(-1 * (((x - mu)**2) / (2 * sigma * sigma)))

def target_density_function(x): # this doesn't need to integrate to less than one because rejection sampling can ignore the normalization constant, but doesn't it need to converge still?
    
    # y = np.log(x) + np.sqrt(5 * x) # does this even work as a density function?

    y = np.sin(x)

    return y

# Could use a decorator pattern to send in the proper distribution and still do the proper work beforehand that's relevant to it
def sampler_function(n_samples_needed, n_iterations, distribution='uniform'): # I need to figure out how this manual documentation syntax works
    """
    n_samples_needed (int): number of samples to be generated, if the number isn't reached it'll add to the amount of iterations, and if it reaches that amount early it'll stop early.

    n_iterations (int): number of iterations currently planned (wait is this even necessary?)

    returns: vector of samples ( and maybe sample density), number of actual iterations (int)
    """
    valid_samples = np.array([0, 1, 2])
    
    return valid_samples
    

def rejection_sampler(density_fnc, sampler_fnc, k=None, n_iterations=10000, sample_distribution='uniform'): # I could just pass in the distribution functions themselves
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
    # np.random.normal()
    distribution = np.random.uniform
    k = 1.1

    all_samples = []
    valid_samples = []
    for i, sample in enumerate(distribution(1, 15, size=100)):
        print("sample", sample)
        scaled_target = target_density_function(sample)# / k
        print(scaled_target)
        print(i)
        all_samples.append(sample)
        # if sample > scaled_target:
            # valid_samples.append(sample)
        valid_samples.append(scaled_target)

    # need to implement distribution I know, uniform or gaussian for example

    # valid_samples = sampler_fnc(1000, n_iterations, sample_distribution)

    # generate the sample density from the valid samples vs ...?

    # return valid_samples, sample_density

    return valid_samples, [0, 0.3, 0.7], all_samples

valid_samples, _, all_samples = rejection_sampler(None, None)
print(len(valid_samples))

x = np.linspace(0, 15, 100)

# plt.plot(x, np.log(x) + np.sqrt(5 * x), label='idk')
plt.plot(x, np.sin(x), label='sine')
# plt.plot(x, x**2, label='quadratic')
# plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')
plt.scatter(all_samples, valid_samples)
plt.title("Simple Plot")

plt.legend()

plt.show()
# for i in range(1, 100):
#     output = target_density_function(i)
#     print(output)

# valid_samples, sample_density = rejection_sampler(target_density_function, sampler_function)

# plot histogram of the generated samples, compare that against target density function

# plot proportion of rejection as a bar graph (valid samples / n_iterations)