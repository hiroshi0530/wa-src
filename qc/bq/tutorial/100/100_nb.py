#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install blueqat')


# In[2]:


from blueqat import Circuit


# In[3]:


Circuit().h[0].m[:].run(shots=200)


# In[4]:


Circuit().h[0].m[:].run(shots=100)


# In[ ]:


Circuit().h[0].m[:].run(shots=200)
Circuit().h[0].m[:].run(shots=200)


# In[ ]:




