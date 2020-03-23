# Base libraries
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tqdm import tqdm
from tqdm import tnrange
from scipy.special import erf
import pickle
import itertools

from SALib.sample import saltelli
from SALib.analyze import sobol

# Personal libraries
import henon_map as hm

# Matplotlib Settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matplotlib.rcParams['figure.dpi']= 100
matplotlib.rcParams['savefig.dpi'] = 300


# ## Setup

# In[6]:


epsilons = [16.0]#, 16.0, 32.0]

min_turns = 10 ** 3
max_turns = 10 ** 6
n_turn_samples = 100

turn_sampling = np.linspace(min_turns, max_turns, n_turn_samples, dtype=np.int_)[::-1]

d_r = 0.01

baseline_samples = 129
baseline_total_samples = baseline_samples ** 3

other_samples = np.array([9, 17, 33])


# ### Save setup

# In[3]:


with open("setup.pkl", 'wb') as f:
    pickle.dump((epsilons, min_turns, max_turns, n_turn_samples, turn_sampling, d_r, baseline_samples, baseline_total_samples, other_samples), f)

# ## Stratified Monte Carlo

# In[20]:


from stratified_mc import stratified_mc

DA_4 = {}
variance_4 = {}

mc_max_samples = 33 ** 3
mc_samples = np.linspace(0, mc_max_samples, 21, dtype=np.int)[1:]
d_samples = mc_samples[1] - mc_samples[0]
n_sectors = 5

for epsilon in tqdm(epsilons, desc="Stratified Monte Carlo"):
    engine = stratified_mc(n_sectors, mc_max_samples, turn_sampling, d_r, epsilon)
    for iters in mc_samples:
        engine.compute(1, d_samples)
        average, variance = engine.get_result()
        
        DA_4[(epsilon, iters)] = np.power(average, 1/4) 
        variance_4[(epsilon, iters)] = 0.25 * np.power(average, -3/4) * variance 


# ### Saving Data

# In[21]:


with open("data/DA_4.pkl", 'wb') as f:
    pickle.dump(DA_4, f)

with open("data/variance_4.pkl", 'wb') as f:
    pickle.dump(variance_4, f)

# ## Plotting

# ### Comparisons

# In[26]:


import tikzplotlib as tk

cmap = matplotlib.cm.get_cmap('plasma')
epsilon = epsilons[0]

# Stratified Monte Carlo

# In[32]:


plt.figure(figsize=(4, 3))
plt.errorbar(turn_sampling, DA_b[epsilon], yerr=error_b[epsilon], c="black", linewidth=0.5, elinewidth=0.5, label="Baseline")

label = sorted(filter(lambda x: x[0] == epsilon, DA_4), key=lambda a: a[1])[-1]

plt.errorbar(turn_sampling, DA_4[label], yerr=variance_4[label], label="Stratified Monte Carlo")

#plt.title("Comparison with Stratified Monte Carlo")
plt.xlabel("N turns")
plt.ylabel("DA")

plt.legend(fontsize="small")
plt.tight_layout()

plt.savefig("img/stratified_monte_carlo.png", dpi=300)
plt.savefig("img/stratified_monte_carlo.pgf")
tk.save("img/stratified_monte_carlo.tex")


# In[40]:


plt.figure(figsize=(4, 3))
elements = np.linspace(0,1,len(DA_4))
for i, label in enumerate(sorted(DA_4, key=lambda a: a[1])):
    if label[0] == epsilon:
        plt.errorbar(turn_sampling, DA_4[label], variance_4[label], c=cmap(elements[i]), label=str(label[1]), linewidth=0.6, elinewidth=0.6)

plt.xlabel("N turns")
plt.ylabel("DA")

plt.legend(title="N samples", ncol=2, fontsize="small")
plt.tight_layout()

plt.savefig("img/stratified_monte_carlo_evolution.png", dpi=300)
plt.savefig("img/stratified_monte_carlo_evolution.pgf")
tk.save("img/stratified_monte_carlo_evolution.tex")
