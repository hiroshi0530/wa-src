
# coding: utf-8

# ## blueqat tutorial 100番代
# 
# 今まで自己流でやってきましたが、一度他の方々がどのような考え方やコーディングをしているのか勉強してみようと思ってやってみました。本書は実際の業務に活用する上でとても参考になる内容だと思っています。データ分析に関わる仕事をしたい方にお勧めしたいです。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[3]:


from blueqat import __version__
print('blueqat version : ', __version__)


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[4]:


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('pandas version :', pd.__version__)


# In[5]:


from blueqat import Circuit


# ### 古典ベートと量子ゲートの比較
# 
# #### NOTゲート VS Xゲート
# 
# #### Xゲート
# 
# #### Yゲート
# 
# #### Zゲート
# 
# #### アダマールゲート
# 
# #### 位相ゲート
# 
# #### CNOTゲート

# ### 量子もつれ

# ### 1量子ビットの計算

# In[6]:


for i in range(5):
  print(Circuit().h[0].m[:].run(shots=100))


# ### 2量子ビットの計算

# In[7]:


Circuit().cx[0,1].m[:].run(shots=100)


# In[8]:


Circuit().x[0].cx[0,1].m[:].run(shots=100)


# In[17]:


Circuit(1).x[0].m[:].run(shots=100)


# In[18]:


Circuit(1).m[:].run(shots=100)


# In[19]:


Circuit().m[:].run(shots=100)


# In[20]:


Circuit().x[0].m[:].run(shots=100)


# ### 2量子ビットの計算

# 0に初期化されています。

# In[24]:


Circuit(2).m[:].run(shots=100)


# In[35]:


Circuit(2).m[0:2].run()


# In[36]:


a = [i for i in range(5)]
a[0:2]


# In[22]:


Circuit(2).cx[0,1].m[:].run(shots=100)


# In[23]:


Circuit(3).cx[0,1].m[:].run(shots=100)


# ### 2量子ビット

# 二つの量子ビットを用意して、初期状態を確認します。

# In[37]:


Circuit(2).m[:].run(shots=100)


# 00の状態に、CXゲートをかけて、結果が変わらないことを確認します。

# In[38]:


Circuit(2).cx[0,1].m[:].run(shots=100)


# 0番目のビットにXゲートを作用させてからCXゲートをかけてみます。

# In[39]:


Circuit(2).x[0].cx[0,1].m[:].run(shots=100)


# ### 重ね合わせ
# アダマールゲートを用いて重ね合わせの状態を作り出します。

# In[41]:


Circuit(1).m[:].run(shots=100)


# In[42]:


Circuit(1).h[0].m[:].run(shots=100)


# ### 波動関数の取得

# In[47]:


Circuit().h[0].run()


# In[48]:


Circuit().x[0].h[0].run()


# m : 状態ベクトルから、実際に測定を行う

# ### 量子もつれ
# 
# 因数分解できない⇒量子もつれ

# In[50]:


Circuit().h[0].cx[0,1].m[:].run(shots=100)


# In[49]:


Circuit().h[0,1].m[:].run(shots=100)


# $$
# \left\langle\varphi | \frac{\psi}{a}\right\rangle
# $$

# $$
# \left| x \right> \otimes \left| y \right>
# $$

# 量子もつれはアダマールゲートとCXゲートで作成できます。
# 
# <div>
# $$
# \begin{aligned}
# & \text{CX}\left(H \left| 0 \right> \otimes \left| 1 \right>   \right) \\
# &= 
# \end{aligned}
# $$
# </div>

# ### Bell状態
# 
# <div>
# $$
# \left| \Phi^{+} \right> = \frac{1}{\sqrt{2}}\left(\left| 00 \right> + \left| 11 \right>  \right)
# $$
# $$
# \left| \Phi^{-} \right> = \frac{1}{\sqrt{2}}\left(\left| 00 \right> - \left| 11 \right>  \right)
# $$
# $$
# \left| \Psi^{+} \right> = \frac{1}{\sqrt{2}}\left(\left| 01 \right> + \left| 10 \right>  \right)
# $$
# $$
# \left| \Psi^{-} \right> = \frac{1}{\sqrt{2}}\left(\left| 01 \right> - \left| 10 \right>  \right)
# $$
# </div>
