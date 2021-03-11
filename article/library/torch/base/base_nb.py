#!/usr/bin/env python
# coding: utf-8

# ## pytorch の基礎
# 
# 最近NLP関連でpytorchからBERTを触ることが増えたので、pytorchでMNISTをやってみようと思います。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/torch/base/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/torch/base/base_nb.ipynb)
# 
# ### 筆者の環境
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[7]:


get_ipython().system('sw_vers')


# In[8]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import torch

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('torch version :', torch.__version__)


# ## sympyの3つの数値型
# 
# sympyは
# 
# - Real
# - Rational
# - Integer
# 
# の三つの型を持つようです。
# 

# In[10]:


from torchvision.datasets import MNIST

from torchvision import transforms
from torch.utils.data import DataLoader

mnist_train = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

print("訓練データの数:", len(mnist_train), "テストデータの数:", len(mnist_test))

img_size = 28
batch_size = 256
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


# In[19]:


mnist_train


# In[15]:


type(mnist_train)


# In[23]:


next(iter(mnist_train))

for i in mnist_train:
  print(len(i))


# In[ ]:





# ## 参考ページ
# 
# こちらのページを参考にしました。
# 
# - http://www.turbare.net/transl/scipy-lecture-notes/packages/sympy.html
