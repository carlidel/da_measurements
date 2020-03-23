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

# ## 2D Scan

# In[23]:


data_5 = {}
DA_5 = {}
uncertainty_5 = {}

alpha_max_samples = 33 ** 3

alpha_values = np.linspace(0, np.pi * 0.5, alpha_max_samples)

for epsilon in tqdm(epsilons, desc="2D scan"):
    henon_engine = hm.radial_scan.generate_instance(d_r, alpha_values, np.zeros(alpha_values.shape), np.zeros(alpha_values.shape), epsilon)
    radiuses = henon_engine.compute(turn_sampling)
    data_5[(epsilon)] = radiuses

    for i in [1, 2, 4, 8, 16, 32]:
        alpha = alpha_values[::i]
        d_alpha = alpha[1] - alpha[0]
        cutted_radiuses = radiuses[::i]
        value = integrate.simps(cutted_radiuses ** 2, alpha, axis=0)
        less_value = integrate.simps(cutted_radiuses[::2] ** 2, alpha[::2], axis=0)
        uncertainty = np.abs((value - less_value))
        
        DA = np.sqrt(value * 2 / np.pi)
        uncertainty = 0.5 * np.power(value * 2 / np.pi, -0.5) * uncertainty
        DA_5[(epsilon, cutted_radiuses.shape)] = np.asarray(DA)
        uncertainty_5[(epsilon, cutted_radiuses.shape)] = uncertainty 


# ### Saving Data

# In[24]:


with open("data/DA_5.pkl", 'wb') as f:
    pickle.dump(DA_5, f)
    
with open("data/uncertainty_5.pkl", 'wb') as f:
    pickle.dump(uncertainty_5, f)

# ## Plotting

# ### Comparisons

# In[26]:


import tikzplotlib as tk

cmap = matplotlib.cm.get_cmap('plasma')
epsilon = epsilons[0]

# 2D Integral

# In[36]:


plt.figure(figsize=(4, 3))
plt.errorbar(turn_sampling, DA_b[epsilon], yerr=error_b[epsilon], c="black", linewidth=0.5, elinewidth=0.5, label="Baseline")
elements = np.linspace(0,1,len(DA_5))

label = sorted(filter(lambda x: x[0] == epsilon, DA_5), key=lambda a: a[1])[-1]
plt.errorbar(turn_sampling, DA_5[label], yerr=uncertainty_5[label], label="2D integral", linewidth=0.6, elinewidth=0.6)

#plt.title("Comparison with 2D integral")
plt.xlabel("N turns")
plt.ylabel("DA")

plt.legend()
plt.tight_layout()

plt.savefig("img/2d_integral.png", dpi=300)
plt.savefig("img/2d_integral.pgf")
tk.save("img/2d_integral.png")


# In[37]:


plt.figure(figsize=(4, 3))
elements = np.linspace(0,1,len(DA_5))
for i, label in enumerate(sorted(DA_5, key=lambda a: a[1])):
    if label[0] == epsilon:
        plt.plot(turn_sampling, DA_5[label], c=cmap(elements[i]), label=str(label[1][0]))

plt.xlabel("N turns")
plt.ylabel("DA")

plt.legend(title="N samples", ncol=2, fontsize="small")
plt.tight_layout()

plt.savefig("img/2d_integral_evolution.png", dpi=300)
plt.savefig("img/2d_integral_evolution.pgf")
tk.save("img/2d_integral_evolution.png")


# In[ ]:

