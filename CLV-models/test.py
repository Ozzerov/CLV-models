__author__ = 'Alex'

import numpy as np
import pandas as pd
from numpy.random import gamma, rand, exponential, seed
from models import pareto_ndb_ll
from scipy import special
from scipy.optimize import differential_evolution

# seed(7)

time_start, time_end = -52, 52
n = 300000

r, alpha, s, beta = 3, 3, 5, 5
users = pd.DataFrame()
users['lam'] = gamma(r, 1 / alpha, n)
users['mu'] = gamma(s, 1 / beta, n)
users['tau'] = exponential(1 / users.mu)
users['freq'] = 0
users['rec'] = 0

users['age'] = -rand(n) * time_start
users['alive'] = True
users['past'] = users[['tau', 'age']].min(axis=1)

j = 0

from datetime import datetime

a = [datetime.now()]
while users.alive.any():
    users.loc[users.alive, 'dt'] = exponential(1 / users[users.alive].lam)
    users.loc[users.alive, 'alive'] = users.rec + users.dt < users.past
    users.loc[users.alive, 'freq'] += 1
    users.loc[users.alive, 'rec'] += users.dt



# print(rec < T)

#print(freq, rec, T)
x = users.freq
tx = users.rec
T = users.age
penalty = .1

a = pareto_ndb_ll([0.2, 0.3, 0.7, 0.3], x, tx, T, penalty)
b = differential_evolution(pareto_ndb_ll, [(0, 20), (0, 20), (0, 20), (0, 20)], args=(x, tx, T, penalty))
print(b)


