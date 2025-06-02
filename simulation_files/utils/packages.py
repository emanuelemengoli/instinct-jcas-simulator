# import subprocess
# import sys

# # Function to install packages
# def install_packages():
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# # Install dependencies before importing other modules
# install_packages()
# from radio_env import *
import random
from tqdm import tqdm
import numpy as np
from math import prod, exp, dist, ceil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy.spatial import distance_matrix, Voronoi, voronoi_plot_2d#, cKDTree
from numpy.random import poisson,uniform,rand, multivariate_normal,randint,normal, exponential
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
# import copy
from tabulate import tabulate # type: ignore
from matplotlib.cm import viridis
import matplotlib.cm as cm
from collections import Counter
from numpy.linalg import norm, eigvals #matrix_norm
from numpy import trace
import seaborn as sns
np.random.seed(142)
rng_sensing = np.random.default_rng(seed=142)         # For sensing procedures
rng_communication = np.random.default_rng(seed=31415)

import multiprocessing
from math import floor
# plt.rcParams['text.usetex'] = True
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import warnings
warnings.filterwarnings("ignore")

from IPython.display import display, Markdown

from matplotlib.ticker import MaxNLocator
from scipy.special import kl_div

import pylab as plot
params = {'legend.fontsize': 9,
          'legend.handlelength': 2}
plot.rcParams.update(params)