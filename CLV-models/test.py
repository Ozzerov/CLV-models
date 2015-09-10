__author__ = 'Alex'

import pandas as pd
from Pareto_NBD import *

df = pd.DataFrame.from_csv('cdnow_data.csv')
freq = df.p1x
rec = df.t_x
age = df['T']

my_fit = ParetoNBD()
my_fit.fit(freq, rec, age)
df['p_alive'] = my_fit.p_alive(freq, rec, age)
print(df.to_string())



