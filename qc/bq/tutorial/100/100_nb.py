#!/usr/bin/env python
# coding: utf-8

# ## blueqat tutorial 100番代
# 
# すべては[blueqat tutorial](https://github.com/Blueqat/Blueqat-tutorials)を勉強しながらやってみた程度ですので、詳細はそちらを参照してください。
# 
# これはまではIBM社のQiskitを利用していましたが、日本製のblueqatも使ってみることにしました。
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


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

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


# 

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

# In[ ]:





# In[ ]:





# In[ ]:





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


# In[ ]:





# In[22]:


Circuit(2).cx[0,1].m[:].run(shots=100)


# In[23]:


Circuit(3).cx[0,1].m[:].run(shots=100)


# In[ ]:





# In[ ]:





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

# In[1]:


1 + 1


# ### 003 量子重ね合わせ

# In[2]:


from blueqat import Circuit


# In[3]:


# アダマールゲートによる重ね合わせの作成
Circuit().h[0].m[:].run(100)


# アダマールゲートを設定し場合、+状態と言う状態になる。-状態はZゲートをかける。

# In[4]:


Circuit().h[0].z[0].m[:].run(100)


# +状態

# In[5]:


Circuit().h[0].run()


# -状態

# In[6]:


Circuit().h[0].z[0].run()


# アダマールゲート
# 
# $$
# \frac{1}{\sqrt{2}}
# \begin{pmatrix}
# 1 & 1 \\
# 1 & -1 \\
# \end{pmatrix}
# $$
# 

# ### 量子もつれ
# 
# アダマールゲート＋ＣＸゲートで作成します。

# In[7]:


Circuit().h[0].cx[0,1].m[:].run(100)


# 1番目の量子ビットが$\left| 0 \right> $が測定された場合、2番目のビットは$\left| 0 \right> $となり、$\left| 1 \right> $が測定された場合、2番目のビットは$\left| 1 \right> $となります。つまり、同期しているという事です。

# ### 重ね合わせとの違い
# 
# コントロールゲートを作用させないと、以下の様に4通りのビットの状態が測定され、同期がされていません。

# In[9]:


Circuit().h[0,1].m[:].run(100)


# ### Bell状態（2量子ビットのもつれ）

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

# ## Step2. Advanced Operation of Quantum Logic Gate

# ### ブロッホ球とパウリ行列
# 
# 1量子ビットはブロッホ球という3次元の点のどこでも取ることが可能で
# 
# 二つの状態の重ね合わせ状態を表現するために利用される3次元
# 
# ブロッホさんに由来するらしいです。今回初めて知りました･･･
# 
# 

# 3次元上に広がっているので、$X,Y,Z$の三つの軸に対しての回転の作用を考える事が出来ます。チュートリアルでは$X,Y,Z$で書かれていますが、$\sigma_x, \sigma_y, \sigma_z$となどとパウリ行列として表現されることも多いかと思います。

# $$
# X=\sigma_x = 
# \begin{pmatrix}
# 0 & 1 \\
# 1 & 0 \\
# \end{pmatrix}
# $$
# 
# $$
# Y=\sigma_y = 
# \begin{pmatrix}
# 0 & -i \\
# i & 0 \\
# \end{pmatrix}
# $$
# 
# $$
# Z=\sigma_z = 
# \begin{pmatrix}
# 1 & 0 \\
# 0 & -1 \\
# \end{pmatrix}
# $$
# 
# 

# ### 1量子ゲート
# 
# #### 固定回転ゲート
# 
# $x,y,z$のそれぞれの軸に対して180°の固定角度の回転を行います。

# In[ ]:





# #### 任意回転ゲート
# 任意の角度の回転をそれぞれの軸に対して実行することが出来ます。

# In[ ]:





# In[ ]:





# In[10]:


Circuit().cz[0,1].m[:].run(shots=100)


# In[ ]:





# In[ ]:





# ## swap ゲート
# 二つの量子ビットを入れ替えるゲートです。コントロールゲートを3つ使うと実装可能のようです。

# In[12]:


Circuit(2).x[0].m[:].run(shots=100)


# コントールゲートを三回作用させます。

# In[13]:


Circuit().x[0].cx[0,1].cx[1,0].cx[0,1].m[:].run(shots=100)


# blueqatではswapゲートはswapメソッドが用意されているようです。

# In[14]:


Circuit().x[0].swap[0,1].m[:].run(100)


# スワップゲート
# 
# 
# $$
# \text{SWAP}=
# \begin{pmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & 0 & 1 & 0 \\
# 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 1 \\
# \end{pmatrix}
# $$

# ### トフォリゲート
# 三つの量子ビットを利用するゲートで、二つのコントールビットの一つのターゲットビットを持つ。

# ### イジングゲート
# 二つの量子ビットを同時に回転させるゲート

# In[ ]:





# In[ ]:


https://qiita.com/YuichiroMinato/items/531cb67492783a1b19b9

