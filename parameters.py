import numpy as np

savepath = "./"

epsilons = [16.0]#, 16.0, 32.0]

min_turns = 10 ** 2
max_turns = 10 ** 5
n_turn_samples = 500

turn_sampling = np.linspace(min_turns, max_turns, n_turn_samples, dtype=np.int_)[::-1]

d_r = 0.01

starting_position = 0.3

# BASELINE COMPUTING
baseline_samples = 129
baseline_total_samples = baseline_samples ** 3

# RADIAL AVERAGE COMPUTING
n_subdivisions = 128
samples = 8193

# MONTE CARLO
mc_max_samples = 5 * 10 ** 4
mc_min_samples = 10 ** 1
mc_samples = np.linspace(mc_min_samples, mc_max_samples, 1000, dtype=np.int)

# STRATIFIED MONTE CARLOif os.path.exists("demofile.txt"):
  os.remove("demofile.txt")
mcs_max_samples = 5 * 10 ** 4
mcs_samples = np.linspace(0, mcs_max_samples, 101, dtype=np.int)[1:]
mcs_n_sectors = 5