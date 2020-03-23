# ## Import Libraries

# In[4]:


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


# ### Load setup

# In[4]:


with open("setup.pkl", "rb") as f:
    epsilons, min_turns, max_turns, n_turn_samples, turn_sampling, d_r, baseline_samples, baseline_total_samples, other_samples = pickle.load(f)

# ## Radial average

# In[20]:


DA_2 = {}

DA = []

n_subdivisions = 5

samples = 4097
alpha_preliminary_values = np.linspace(-1.0, 1.0, samples)
alpha_values = np.arccos(alpha_preliminary_values) / 2
d_preliminar_alpha = alpha_preliminary_values[1] - alpha_preliminary_values[0]

for epsilon in tqdm(epsilons, desc="Radial average"):
    # Extracting the radiuses with theta1 = theta2 = 0.0
    
    preliminar_engine = hm.radial_scan.generate_instance(
        d_r, 
        alpha_values, 
        np.zeros(alpha_values.shape),
        np.zeros(alpha_values.shape),
        epsilon
    )
    all_radiuses = preliminar_engine.compute(turn_sampling)
    
    values = []
    for i in tqdm(range(len(turn_sampling))):
        temp_values = np.array([[]])
        for index, j in enumerate(range(0, samples, 128)):
            stopping = (j + 128 if j != samples - 1 else samples)
            radiuses = all_radiuses[j : stopping, i]

            engine = hm.full_track.generate_instance(
                radiuses,
                alpha_values[j : stopping],
                np.zeros(alpha_values.shape)[j : stopping],
                np.zeros(alpha_values.shape)[j : stopping],
                np.ones(alpha_values.shape, dtype=np.int)[j : stopping] * turn_sampling[i],
                epsilon)

            x, y, px, py = engine.compute()
            temp = engine.accumulate_and_return(n_subdivisions)
            if index == 0:
                temp_values = temp
            else:
                temp_values = np.concatenate((temp_values, temp))
        
        values.append(temp_values)
    
    for jump in [1, 2, 4, 8, 16, 32]:
        DA = []
        for i in range(len(turn_sampling)):
            DA.append(np.power(integrate.romb(values[i][::jump], d_preliminar_alpha * jump) * 0.5, 1/4))
        DA_2[(epsilon, len(values[i][::jump]))] = DA


# ### Saving Data

# In[15]:


with open("data/DA_2.pkl", 'wb') as f:
    pickle.dump(DA_2, f)


# ### Loading Data

# In[16]:


with open("data/DA_2.pkl", 'rb') as f:
    DA_2 = pickle.load(f)

# ## Plotting

# ### Comparisons

# In[26]:


import tikzplotlib as tk

cmap = matplotlib.cm.get_cmap('plasma')
epsilon = epsilons[0]


# Radial Average

# In[29]:


# Radial Average
plt.figure(figsize=(4, 3))
plt.errorbar(turn_sampling, DA_b[epsilon], yerr=error_b[epsilon], c="black", linewidth=0.5, elinewidth=0.5, label="Baseline")
elements = np.linspace(0,1,len(DA_2))
for i, label in enumerate(sorted(DA_2, key=lambda a: a[1])):
    if label[0] == epsilon:
        plt.plot(turn_sampling, np.array(DA_2[label]), c=cmap(elements[i]), label=str(label[1]))
#plt.title("Comparison with Angular Average")
plt.xlabel("N turns")
plt.ylabel("DA")
plt.legend(title="N samples", fontsize="small", ncol=2)
plt.tight_layout()

plt.savefig("img/radial_average.png", dpi=300)
plt.savefig("img/radial_average.pgf")
tk.save("img/radial_average.tex")