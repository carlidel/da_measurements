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


# ## Baseline

# In[5]:


data_b = {}
DA_b = {}
error_b = {}

alpha_preliminary_values = np.linspace(-1.0, 1.0, baseline_samples)
alpha_values = np.arccos(alpha_preliminary_values) / 2
theta1_values = np.linspace(0.0, np.pi * 2.0, baseline_samples, endpoint=False)
theta2_values = np.linspace(0.0, np.pi * 2.0, baseline_samples, endpoint=False)

d_preliminar_alpha = alpha_preliminary_values[1] - alpha_preliminary_values[0]
d_theta1 = theta1_values[1] - theta1_values[0]
d_theta2 = theta2_values[1] - theta2_values[0]

alpha_mesh, theta1_mesh, theta2_mesh = np.meshgrid(alpha_values, theta1_values, theta2_values, indexing='ij')

alpha_flat = alpha_mesh.flatten()
theta1_flat = theta1_mesh.flatten()
theta2_flat = theta2_mesh.flatten()


# In[6]:


for epsilon in tqdm(epsilons, desc="Baseline"):
    
    # Data generation
    henon_engine = hm.radial_scan.generate_instance(d_r, alpha_flat, theta1_flat, theta2_flat, epsilon)
    radiuses = henon_engine.compute(turn_sampling)
    radiuses = radiuses.reshape((baseline_samples, baseline_samples, baseline_samples, len(turn_sampling)))
    data_b[epsilon] = radiuses

    # Computing DA
    DA = []
    error_list = []
    mod_radiuses = radiuses.copy()
    mod_radiuses = np.power(radiuses, 4)
    
    mod_radiuses1 = integrate.romb(mod_radiuses, dx=d_theta1, axis=1)
    error_radiuses1 = np.absolute(
        (mod_radiuses1 - integrate.romb(mod_radiuses[:,::2,:], dx=d_theta1 * 2, axis=1)) / mod_radiuses1
    )
    error_radiuses1 = np.average(error_radiuses1, axis=1)
        
    mod_radiuses2 = integrate.romb(mod_radiuses1, dx=d_theta2, axis=1)
    error_radiuses2 = np.absolute(
        (mod_radiuses2 - integrate.romb(mod_radiuses1[:,::2], dx=d_theta2 * 2, axis=1)) / mod_radiuses2
    )
    error_radiuses2 += error_radiuses1
    error_radiuses2 = np.average(error_radiuses2, axis=0)
        
    mod_radiuses3 = integrate.romb(mod_radiuses2, dx=d_preliminar_alpha, axis=0)
    error_radiuses3 = np.absolute(
        (mod_radiuses3 - integrate.romb(mod_radiuses2[::2], dx=d_preliminar_alpha * 2, axis=0)) / mod_radiuses3
    )
    error_radiuses3 += error_radiuses2

    error_raw = mod_radiuses3/ (2 * 2 * 2 * np.pi * np.pi) * error_radiuses3
    error = 0.25 * np.power(mod_radiuses3 / (2 * 2 * 2 * np.pi * np.pi), -3/4) * error_raw

    for i in range(len(turn_sampling)):
        DA.append(
            np.power(
                mod_radiuses3[i] / (2 * 2 * 2 * np.pi * np.pi),
                1/4
            )
        )
        error_list.append(error[i])
    DA_b[epsilon] = np.asarray(DA)
    error_b[epsilon] = np.asarray(error_list)


# ### Saving Data

# In[7]:


with open("data/raw_data_b.pkl", 'wb') as f:
    pickle.dump(data_b, f)
    
with open("data/DA_b.pkl", 'wb') as f:
    pickle.dump(DA_b, f)
    
with open("data/error_b.pkl", 'wb') as f:
    pickle.dump(error_b, f)


# ### Loading Data

# In[8]:


with open("data/raw_data_b.pkl", 'rb') as f:
    data_b = pickle.load(f)
    
with open("data/DA_b.pkl", 'rb') as f:
    DA_b = pickle.load(f)
    
with open("data/error_b.pkl", 'rb') as f:
    error_b = pickle.load(f)


# ## Standard Integral

# In[9]:


DA_1 = {}
error_1 = {}
for epsilon in tqdm(epsilons, desc="Standard Integral"):
    base_radiuses = data_b[epsilon]
    
    for i in [2, 4, 8, 16, 32]:
        radiuses = base_radiuses[::i, ::i, ::i]
        DA = []
        error_list = []
        mod_radiuses = radiuses.copy()
        mod_radiuses = np.power(radiuses, 4)
        
        mod_radiuses1 = integrate.romb(mod_radiuses, dx=d_theta1 * i, axis=1)
        error_radiuses1 = np.absolute(
            (mod_radiuses1 - integrate.romb(mod_radiuses[:,::2,:], dx=d_theta1 * i * 2, axis=1)) / mod_radiuses1
        )
        error_radiuses1 = np.average(error_radiuses1, axis=1)
        
        mod_radiuses2 = integrate.romb(mod_radiuses1, dx=d_theta2 * i, axis=1)
        error_radiuses2 = np.absolute(
            (mod_radiuses2 - integrate.romb(mod_radiuses1[:,::2], dx=d_theta2 * i * 2, axis=1)) / mod_radiuses2
        )
        error_radiuses2 += error_radiuses1
        error_radiuses2 = np.average(error_radiuses2, axis=0)
        
        mod_radiuses3 = integrate.romb(mod_radiuses2, dx=d_preliminar_alpha * i, axis=0)
        error_radiuses3 = np.absolute(
            (mod_radiuses3 - integrate.romb(mod_radiuses2[::2], dx=d_preliminar_alpha * i * 2, axis=0)) / mod_radiuses3
        )
        error_radiuses3 += error_radiuses2
        
        error_raw = mod_radiuses3/ (2 * 2 * 2 * np.pi * np.pi) * error_radiuses3
        error = 0.25 * np.power(mod_radiuses3 / (2 * 2 * 2 * np.pi * np.pi), -3/4) * error_raw

        for j in range(len(turn_sampling)):
            DA.append(
                np.power(
                    mod_radiuses3[j] / (2 * 2 * 2 * np.pi * np.pi),
                    1/4
                )
            )
            error_list.append(error[j])
        DA_1[(epsilon, radiuses.size)] = np.asarray(DA)
        error_1[(epsilon, radiuses.size)] = np.asarray(error_list)


# ### Saving Data

# In[10]:


with open("data/DA_1.pkl", 'wb') as f:
    pickle.dump(DA_1, f)
    
with open("data/error_1.pkl", 'wb') as f:
    pickle.dump(error_1, f)

# ## Plotting

# ### Comparisons

# In[26]:


import tikzplotlib as tk

cmap = matplotlib.cm.get_cmap('plasma')
epsilon = epsilons[0]


# Standard Integral 

# In[27]:


plt.figure(figsize=(4, 3))
plt.errorbar(turn_sampling, DA_b[epsilon], yerr=error_b[epsilon], c="black", linewidth=0.5, elinewidth=0.5, label="Baseline")
elements = np.linspace(0,1,len(DA_1))
for i, label in enumerate(sorted(DA_1, key=lambda a: a[1])):
    if label[0] == epsilon:
        plt.plot(turn_sampling, DA_1[label], c=cmap(elements[i]), label=str(label[1]))
#plt.title("Comparison with Standard Integral")
plt.xlabel("N turns")
plt.ylabel("DA")

plt.legend(title="N samples", ncol=2, fontsize="small")
plt.tight_layout()

plt.savefig("img/standard_integral.png", dpi=300)
plt.savefig("img/standard_integral.pgf")
tk.save("img/standard_integral.tex")


# In[28]:


plt.figure(figsize=(4, 3))
plt.errorbar(turn_sampling, DA_b[epsilon], yerr=error_b[epsilon], c="black", linewidth=0.5, elinewidth=0.5, label="Baseline")
elements = np.linspace(0,1,len(DA_1))
for i, label in enumerate(sorted(DA_1, key=lambda a: a[1])):
    if label[0] == epsilon:
        plt.errorbar(turn_sampling, DA_1[label], yerr=error_1[label], c=cmap(elements[i]), linewidth=0.5, elinewidth=0.5, label=str(label[1]))
#plt.title("Comparison with Standard Integral")
plt.xlabel("N turns")
plt.ylabel("DA")

plt.legend(title="N samples", ncol=2, fontsize="small")
plt.tight_layout()

plt.savefig("img/standard_integral_error.png", dpi=300)
plt.savefig("img/standard_integral_error.pgf")
tk.save("img/standard_integral_error.tex")
