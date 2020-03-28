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

"""Implementation of a stratified MC process for my DA problem.
"""


class sector(object):
    def __init__(self, minimums, maximums, max_iters, scanning_list, dr, epsilon):
        self.scanning_list = scanning_list
        self.dr = dr
        self.epsilon = epsilon
        self.problem = {
            'num_vars': 3,
            'names': ['alpha', 'theta1', 'theta2'],
            'bounds': [[maximums[0], minimums[0]],
                       [minimums[1], maximums[1]],
                       [minimums[2], maximums[2]]]
        }
        self.noise_list = saltelli.sample(
            self.problem, max_iters // 8 + 8)
        self.index = 0

        self.data = np.array([])
        self.average = np.nan
        self.variance = np.nan

    def extract(self, n_extractions):
        extraction = self.noise_list[self.index : self.index + n_extractions, :]
        self.index += n_extractions

        alpha = (np.arccos(extraction[:, 0]) / 2)
        
        engine = hm.radial_scan.generate_instance(
            self.dr, alpha, extraction[:, 1], extraction[:, 2], self.epsilon, cuda_device=False)
        radiuses = engine.compute(self.scanning_list)

        if self.data.size == 0:
            self.data = np.power(radiuses, 4)
        else:
            self.data = np.concatenate((self.data, np.power(radiuses, 4)))
        self.average = np.average(self.data, axis=0)
        self.variance = np.std(self.data, axis=0) 

    def reset(self):
        self.index = 0
        self.data = np.array([[]])
        self.average = np.nan
        self.variance = np.nan

    def get_index(self):
        return self.index

    def get_average(self):
        return self.average

    def get_variance(self):
        return self.variance
    
    def get_first_variance(self):
        return self.variance[-1]


class stratified_mc(object):
    def __init__(self, n_sectors, max_iters, scanning_list, dr, epsilon):
        print("Initialization...")
        self.max_iters = max_iters
        self.n_sectors = n_sectors
        self.total_sectors = n_sectors ** 3
        
        self.alpha_sectors =  np.linspace(-1, 1, n_sectors + 1)[::-1]
        self.theta1_sectors = np.linspace(0, np.pi * 2, n_sectors + 1)
        self.theta2_sectors = np.linspace(0, np.pi * 2, n_sectors + 1)
        
        self.sectors = [[[
            sector(
                [self.alpha_sectors[a], self.theta1_sectors[b],
                    self.theta2_sectors[c]],
                [self.alpha_sectors[a+1], self.theta1_sectors[b+1],
                    self.theta2_sectors[c+1]],
                self.max_iters, scanning_list, dr, epsilon
            )
        for c in range(n_sectors)]
        for b in range(n_sectors)]
        for a in range(n_sectors)]

        self.p = np.ones((n_sectors, n_sectors, n_sectors))
        self.averages = np.zeros((n_sectors, n_sectors, n_sectors, len(scanning_list)))
        self.variances = np.zeros((n_sectors, n_sectors, n_sectors, len(scanning_list)))
        self.indices = np.zeros((n_sectors, n_sectors, n_sectors))
        

        print("...first computation...")
        for a in tqdm(range(self.n_sectors)):
            for b in range(self.n_sectors):
                for c in range(self.n_sectors):
                    self.sectors[a][b][c].extract(10)
                    self.p[a,b,c] = np.sqrt(self.sectors[a][b][c].get_first_variance())
                    self.variances[a,b,c] = self.sectors[a][b][c].get_variance()
                    self.averages[a,b,c] = self.sectors[a][b][c].get_average()
                    self.indices[a,b,c] = self.sectors[a][b][c].get_index()
        
        print("...done initializing.")
                        
        

    def compute(self, samples):
        self.p /= np.sum(self.p)
        items = np.asarray(np.rint(self.p * samples), dtype=np.int)
        if np.sum(items) < samples:
            difference = samples - np.sum(items)
            items = items.flatten()
            while difference != 0:
                items[np.random.choice(range(len(items)))] += 1
                difference -= 1
            items = items.reshape(self.n_sectors, self.n_sectors, self.n_sectors)
        for a in range(self.n_sectors):
            for b in range(self.n_sectors):
                for c in range(self.n_sectors):
                    if items[a,b,c] > 0:
                        self.sectors[a][b][c].extract(items[a,b,c])
                        self.p[a,b,c] = np.sqrt(self.sectors[a][b][c].get_first_variance())
                        self.variances[a,b,c] = self.sectors[a][b][c].get_variance()
                        self.averages[a,b,c] = self.sectors[a][b][c].get_average()
                        self.indices[a,b,c] = self.sectors[a][b][c].get_index()

    def get_result(self):
        average = np.average(self.averages, axis=(0,1,2))
        variance = np.sum(np.array([self.variances[:,:,:,i] / (4 * self.indices) for i in range(self.variances.shape[3])]), axis=(1,2,3))
        return average, variance
