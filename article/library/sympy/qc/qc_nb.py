#!/usr/bin/env python
# coding: utf-8

# ## sympy で量子演算のシミュレーション
# 
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sympy/base/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sympy/base/base_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)


# ## 基本的な確率分布
# 
# sympyは3つの数値型
# 

# In[4]:


from sympy import *
from sympy.physics.quantum import *
from sympy.physics.quantum.qubit import Qubit,QubitBra
from sympy.physics.quantum.gate import X,Y,Z,H,S,T,CNOT,SWAP, CPHASE
init_printing()


# In[5]:


import sympy
print('sympy version : ', sympy.__version__)


# In[59]:


_0 = Qubit('0')
_1 = qapply(X(0)*_0)
print(represent(_0))


# In[61]:


represent(_1)


# In[ ]:


Qubit('0')


# In[62]:


qapply(H(0)*Qubit('00'))


# In[63]:


get_ipython().run_line_magic('pinfo', 'represent')


# In[36]:


Qubit('1')


# In[38]:


Qubit('00')


# In[39]:


QubitBra('00')


# In[46]:


a1 = QubitBra(0) * Qubit('0')
print(a1)


# In[8]:


represent(psi)


# In[9]:


from sympy import *
from sympy.physics.quantum import *
from sympy.physics.quantum.qubit import Qubit,QubitBra
init_printing() # ベクトルや行列を綺麗に表示するため
psi = Qubit('0')
psi
represent(psi)


# In[56]:


psi = Qubit('0')
psi


# In[57]:


represent(psi)


# In[12]:


a, b = symbols('alpha, beta')  #a, bをシンボルとして、alpha, betaとして表示
ket0 = Qubit('0')
ket1 = Qubit('1')
psi = a * ket0 + b* ket1
psi # 状態をそのまま書くとケットで表示してくれる


# In[13]:





# In[14]:


X(0)


# <div>

# In[15]:


represent(X(0), nqubits=1)


# </div>

# In[16]:


represent(X(0), nqubits=2)


# In[34]:


represent(H(0),nqubits=1)
simplify(represent(H(0),nqubits=1))


# In[18]:


represent(S(0),nqubits=1)


# In[19]:


represent(T(0),nqubits=1)


# In[20]:


ket0 = Qubit('0')
S(0)*Y(0)*X(0)*H(0)*ket0


# In[21]:


qapply(S(0)*Y(0)*X(0)*H(0)*ket0)


# In[23]:


a,b,c,d = symbols('alpha,beta,gamma,delta')
psi = a*Qubit('0')+b*Qubit('1')
phi = c*Qubit('0')+d*Qubit('1')


# In[24]:


TensorProduct(psi, phi) #テンソル積


# In[25]:


represent(TensorProduct(psi, phi))


# In[64]:


from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.gate import H, CNOT
from sympy.physics.quantum.qapply import qapply
bell = {}
# to Bell
for yx in ['00', '10', '01', '11']:
  result = qapply(CNOT(0,1)*H(0)*Qubit(yx))
  bell[yx] = result
  print (f'{yx} -> ', result)
# from Bell
for i, state in bell.items():
  result = qapply(H(0)*CNOT(0,1)*state)
  print(f'beta{i} -> ', result)


# In[ ]:




