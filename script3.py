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

# ## Monte Carlo

# In[17]:


mc_max_samples = 33 ** 3
mc_min_samples = 10 ** 2
mc_samples = np.linspace(mc_min_samples, mc_max_samples, 10, dtype=np.int)

problem = {
    'num_vars': 3,
    'names': ['alpha', 'theta1', 'theta2'],
    'bounds': [[-1, 1],
               [0, np.pi * 2],
               [0, np.pi * 2]]
    }
param_values = saltelli.sample(problem, mc_max_samples // 8 + 8)

alpha = np.array([np.arccos(p[0])/2 for p in param_values])
theta1 = np.array([p[1] for p in param_values])
theta2 = np.array([p[2] for p in param_values])

DA_3 = {}
variance_3 = {}
data_3 = {}
for epsilon in tqdm(epsilons, desc="Monte Carlo"):
    # Data generation
    henon_engine = hm.radial_scan.generate_instance(d_r, alpha, theta1, theta2, epsilon)
    radiuses = henon_engine.compute(turn_sampling)
    
    data_3[epsilon] = radiuses

    # Computing DA
    
    for sample in mc_samples:
        average = np.average(np.power(radiuses[:sample], 4), axis=0)
        error = np.std(np.power(radiuses[:sample], 4), axis=0) / np.sqrt(sample)
        DA_3[(epsilon, sample)] = np.power(average, 1/4)
        variance_3[(epsilon, sample)] = 0.25 * np.power(average, -3/4) * error
    


# ### Saving Data

# In[18]:


with open("data/DA_3.pkl", 'wb') as f:
    pickle.dump(DA_3, f)
    
with open("data/variance_3.pkl", 'wb') as f:
    pickle.dump(variance_3, f)


# ### Loading Data

# In[19]:


with open("data/DA_3.pkl", 'rb') as f:
    DA_3 = pickle.load(f)
    
with open("data/variance_3.pkl", 'rb') as f:
    variance_3 = pickle.load(f)

# Monte Carlo
plt.figure(figsize=(4, 3))
plt.errorbar(turn_sampling, DA_b[epsilon], yerr=error_b[epsilon], c="black", linewidth=0.5, elinewidth=0.5, label="Baseline")

label = sorted(filter(lambda x: x[0] == epsilon, DA_3), key=lambda a: a[1])[-1]
plt.errorbar(turn_sampling, DA_3[label], yerr=variance_3[label], label="Monte Carlo", linewidth=0.6, elinewidth=0.6)

plt.xlabel("N turns")
plt.ylabel("DA")

plt.legend(title="N samples", ncol=2, fontsize="small")
plt.tight_layout()

plt.savefig("img/monte_carlo.png", dpi=300)
plt.savefig("img/monte_carlo.pgf")
tk.save("img/monte_carlo.tex")


# ## Plotting

# ### Comparisons

# In[26]:


import tikzplotlib as tk

cmap = matplotlib.cm.get_cmap('plasma')
epsilon = epsilons[0]


# In[38]:


plt.figure(figsize=(4, 3))
elements = np.linspace(0,1,len(DA_3))
for i, label in enumerate(sorted(DA_3, key=lambda a: a[1])):
    if label[0] == epsilon:
        plt.errorbar(turn_sampling, DA_3[label], yerr=variance_3[label], c=cmap(elements[i]), label=str(label[1]),linewidth=0.6, elinewidth=0.6)

plt.xlabel("N turns")
plt.ylabel("DA")

plt.legend(title="N samples", ncol=2, fontsize="small")
plt.tight_layout()

plt.savefig("img/monte_carlo_evolution.png", dpi=300)
plt.savefig("img/monte_carlo_evolution.pgf")
tk.save("img/monte_carlo_evolution.tex")
