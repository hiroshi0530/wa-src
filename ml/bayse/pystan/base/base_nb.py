#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pystan
import matplotlib.pyplot as plt

schools_dat = {
 'J': 8,
 'y': [28,  8, -3,  7, -1,  1, 18, 12],
 'sigma': [15, 10, 16, 11,  9, 11, 10, 18]
}

fit = pystan.stan(file='8schools.stan', data=schools_dat, iter=100, chains=4)
print(fit)
fit.plot()
fit.plot()
plt.show()


# In[ ]:




