#!/usr/bin/env python
# coding: utf-8

# ## 特異値分解と低ランク近似
# 
# 主に推薦システムの理解に必要な線型代数の知識をまとめていこうと思います。
# 推薦システムで利用される、user-itemの行列（嗜好行列）に対して、しばしば低ランク近似が成立する事を前提に議論が進められることがあります。
# 
# それは、ユーザーはしばしばある有限のクラスタに分類可能であるという暗黙の仮定に基づいています。例えば、本のECサイト利用者の全ユーザー100万にいたとしても、プログラミングの本をよく買うユーザー、数学の本を買うユーザー、医学書を買うユーザー、週刊誌を買うユーザーというように、ある程度カテゴライズする事が可能です。
# 
# 低ランク近似を利用する際に必要なのが、特異値分解になります。値の大きい$k$個の特異値に属する特異値ベクトルを抽出し、データを圧縮しつつ、見観測のアイテムに対するユーザーの好みを予測します。
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/rec/linalg/base/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/rec/linalg/base/base_nb.ipynb)
# 

# ## 特異値分解
# 
# 特異値分解は正方行列ではない一般の$m \times n$の行列$A$に対して、以下の様に定義されます。
# 
# $$
# A=U\Sigma V^{\mathrm{T}}
# $$
# 
# $$
# \mathbf{A}=\mathbf{U} \cdot \operatorname{diag}\left(\sigma_{1}, \ldots, \sigma_{r}, \mathbf{0}\right) \cdot \mathbf{V}^{*}
# $$
# 
# $\sigma_1, \sigma_2 \cdots$が特異値で、通常大きい方から順番にナンバリングされて定義されます
# 
# $$
# \sigma_{1} \geqq \cdots \geqq \sigma_{r}>0
# $$
# 
# $r$は行列$A$のランクです。
# 
# $$
# r=\operatorname{rank}(\mathbf{A})
# $$
# 
# $U$と$V$は$m \times m$、$n \times n$のユニタリ行列です。
# 
# $$
# \mathbf{U}^{-1}=\mathbf{U}^{*}
# $$
# 
# $$
# \mathbf{V}^{-1}=\mathbf{V}^{*}
# $$
# 
# $A$が対称行列の場合、$A$の特異値と固有値は同じになります。
# 
# ### 行列のイメージ
# 
# 分解後の行列を視覚的に表現すると以下の様になります。

# $$
# A=U\Sigma V^{T}=U\left(\begin{array}{ccc|c}
# \sigma_{1} & & 0 & \\
# & \ddots & & 0 \\
# 0 & & \sigma_{r} & \\
# \hline & 0 & & 0
# \end{array}\right) V^{\mathrm{T}} \\
# = \left(
# \mathbf{u}\_{1} \cdots \mathbf{u}\_{r}\right) \left(\begin{array}{llll}
# \sigma_{1} & & & \\
# & & \ddots & \\
# & & & \sigma_{r}
# \end{array}\right)
# \left(\begin{array}{c}
# \mathbf{v}_{1}^{T} \\
# \vdots \\
# \mathbf{v}_{r}^{T}
# \end{array}\right)
# $$

# ## 固有値分解と特異値分解

# $A=U\Sigma V^{T}$の$U$と$V$はユニタリ行列なので、ベクトル$u$と$v$を用いて、以下の様にかけます。
# 
# $$
# \begin{aligned}
# \mathbf{A} \mathbf{v} &=\sigma \mathbf{u} \\
# \mathbf{A}^{T} \mathbf{u} &=\sigma \mathbf{v}
# \end{aligned}
# $$

# さらに変形します。
# 
# $$
# \begin{aligned}
# &\mathbf{A}^{T} \mathbf{A} \mathbf{v}=\sigma \mathbf{A}^{T} \mathbf{u}=\sigma^{2} \mathbf{v} \\
# &\mathbf{A} \mathbf{A}^{T} \mathbf{u}=\sigma \mathbf{A} \mathbf{v}=\sigma^{2} \mathbf{u}
# \end{aligned}
# $$

# $\mathbf{u}$と$\mathbf{v}$はそれぞれ$\mathbf{A}^{T} \mathbf{A}$と$\mathbf{A} \mathbf{A}^{T} $の固有ベクトルになっています。$\mathbf{u}$と$\mathbf{v}$は左特異ベクトル、右特異ベクトルと呼ばれ、$\mathbf{u}$と$\mathbf{v}$は$\mathbf{AA^{T}}, \mathbf{A^{T}A}$の固有ベクトルになります。また、$\mathbf{A}$の特異値の2乗が$\mathbf{AA^T}$や$\mathbf{A^TA}$の固有値になります。

# この性質から、特異値分解と固有値分解を結びつける事ができます。
# 最初に$\mathbf{A^TA}$の固有値分解を行い、固有ベクトルを求めます。その固有ベクトルは$\mathbf{v_i}$、固有値を$\lambda_i$とします。
# $\lambda_i = \sigma^2$なので、
# 
# $$
# \mathbf{u}_i = \frac{1}{\sqrt{\lambda_i}}\mathbf{A} \mathbf{v_i} 
# $$
# 
# となり、$\mathbf{u}_i$を求める事が出来ます。$\mathbf{v}_i$についても同様に求める事が可能で、固有値分解のアルゴリズムを用いて、特異値分解を計算する事ができます。

# ## 作用素ノルム（演算子ノルム）
# 行列$A$に対して定義されるノルム、
# 
# $$
# \|\|\mathbf{A}\|\|=\max _{x \in \mathbb{C}^{n},\|x\|=1}\|\|\mathbf{A} \mathbf{x}\|\|
# $$
# 
# を作用素ノルムといいます。特異値分解を利用すると作用素ノルムを簡単に求める事ができます。
# 
# $$
# \begin{aligned}
# \|\|A x\|\|^{2} &=\|\|U \Sigma V^{*} x\|\|^{2} \\
# &=x^{\*} V \Sigma^{\*} U^{\*} U \Sigma V^{\*} x=\|\|\Sigma y\|\|^{2} \\
# &=\sigma_{1}^{2}\left|y\_{1}\right|^{2}+\cdots+\sigma\_{r}^{2}\left|y\_{r}\right|^{2} \quad (y = x^{*} V)
# \end{aligned}
# $$
# 
# よって、
# 
# $$
# \begin{aligned}
# \|\|\mathbf{A}\|\|&=\max_{x \in \mathbb{C}^{n},\|\|x\|\|=1}\|\|A x\|\|  \\
# &=\max_{\|\|y\|\|=1} \sqrt{\sigma_{1}^{2}\left|y_{1}\right|^{2}+\sigma_{2}^{2}\left|y_{2}\right|^{2}+\cdots+\sigma_{r}^{2}\left|y_{r}\right|^{2}} \\
# &=\sigma_{1} \quad\left(\sigma_{1} \geq \sigma_{2} \geq \cdots \geq \sigma_{r} \right)
# \end{aligned}
# $$
# 
# となり、作用素ノルムは特異値の最も大きな値となります。
# 
# ### ユニタリ行列の作用素ノルム
# 
# ユニタリ行列の作用素ノルムは1になります。これは定義から明白で、ユニタリ行列を$U (U^{T}U=I)$とすると、
# 
# $$
# \|\|U x\|\|^{2}=x^{T} U^{T} U x=\|\|x\|\|^{2} = 1
# $$
# 
# となります。
# 
# ### 三角不等式
# 
# $$
# \|\|A+B\|\| \leq \|\|A\|\|+\|\|B\|\|
# $$
# 
# ### 積に関する不等式
# 
# $$
# \|\|A B\|\| \leq\|\|A\|\| \cdot\|\|B\|\|
# $$

# ## Eckart-Young（エッカート-ヤング）の定理
# 
# ### 低ランク近似
# 
# 行列$A$を特異値の大きな値から$k$個取り出し、$A$を近似することを低ランク近似と言います。
# 
# $$
# A_{k}=U \Sigma_{k} V^{\mathrm{T}} \equiv U \operatorname{diag}\left(\sigma_{1}, \ldots, \sigma_{k}, 0\right) V^{\mathrm{T}}=\sum_{i=1}^{k} \sigma_{i} u_{i} v_{i}^{\mathrm{T}}(0 \leq k<r)
# $$
# 
# つまり、$\sigma_{k+1}, \cdots, \sigma_{r}$を0とすると言うことになります。
# 
# 
# ### 定理の詳細
# 
# 推薦システムの論文でも、低ランク近似を元に議論がされる場合、エッカートヤングの定理がよく引用されますのでメモっておきます。
# 参考文献[1]に詳細が載っています。
# 
# $$
# \min\_{\operatorname{rank}||X|| \leq k}\|\|X-A\|\|=\left\|\|A_{k}-A\right\|\|=\sigma_{k+1}(A)=\min _{\operatorname{rank}(X)=k}\|\|X-A\|\|
# $$
# 
# この定理は、$A$を中心とする半径$\sigma_{k+1}$の球上には低ランク近似行列$A_k$が存在することになります。それはすべての$k$で成立します。
# 作用素ノルムでノルムを定義すると、非常にわかりやすい直感的な結果を示していると思います。証明は[1]に乗っていますので参考にしてください。
# 
# ### メモ
# 
# $$
# \begin{aligned}
# \left\|A_{k}-A\right\| &=\left\|U\left(\Sigma_{k}-\Sigma\right) V^{T}\right\|=\left\|\left(\Sigma_{k}-\Sigma\right)\right\| \\
# &=\left\|\operatorname{diag}\left(0, \sigma_{k+1}, \ldots, \sigma_{r}, 0\right)\right\|=\sigma_{k+1}
# \end{aligned}
# $$

# ## フロベニウスノルム
# 
# フロベニウスノルムは、行列の大きさを表す指標の一つで、全成分の二乗和で定義されます。$\|\|A\|\|_{\mathrm{F}}$と書かれます。
# 
# $$
# \|\|A\|\|\_{\mathrm{F}}=\sqrt{\sum\_{i, j} a\_{i j}^{2}}
# $$
# 
# ### 特異値との関係
# 
# フロベニウスノルムは特異値の二乗和と等しくなります。
# 
# $$
# \|\|A\|\|\_{\mathrm{F}}^{2}=\sum_{i=1}^{r} \sigma_{i}^{2}
# $$
# 
# ### トレースとの関係
# 
# こちらも実際に計算してみれば明らかですが、転置行列との積のトレースはフロベニウスノルムとなります。
# 
# $$
# \|\|A\|\|\_{\mathrm{F}}^{2}=\operatorname{tr}\left(A A^{\top}\right)=\operatorname{tr}\left(A^{\top} A\right)
# $$

# ## pythonによる実装
# 
# 実際にPythonを用いて特異値分解を実行してみます。
# 
# ### 筆者の環境
# 
# 筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy  version :', scipy.__version__)
print('numpy  version :', np.__version__)


# ### 適当にAを準備
# 
# 整数の4x5の行列を作ります。

# In[70]:


A = np.array([])
for i in range(4): 
  for j in range(5):
    A = np.append(A, np.random.randint(1, 10))
A = A.reshape((4,5))
A


# $\text{rank}(A)$を確認します。

# In[91]:


np.linalg.matrix_rank(A)


# ### 特異値分解
# 
# 実際に特異値分解を実行します。

# In[ ]:


u, S, v_T = np.linalg.svd(A)


# $U$、$\Sigma$、$V$を確認します。

# In[84]:


u.round(2)


# In[85]:


S.round(2)


# In[93]:


v_T.T.round(2)


# ### 再現確認
# 
# 三つの行列をかけることで元の行列が再現できることを確認します。

# In[87]:


(u @ np.append(np.diag(S), [[0], [0], [0], [0]], axis=1) @ v_T).round(2)


# 元の$A$と引き算してみます。

# In[95]:


A.round(2) - (u @ np.append(np.diag(S), [[0], [0], [0], [0]], axis=1) @ v_T).round(2)


# となり、元の$A$となることが確認出来ました。

# また、$U$と$V$がユニタリ行列である事も確認しておきます。

# In[100]:


(u @ u.T).round(2)


# In[102]:


(v_T @ v_T.T).round(2)


# となり、単位行列になっており無事に確認出来ました。
# 
# ### 作用素ノルムの計算
# 
# pythonで作用素ノルムを計算するには、最大の特異値を計算するだけです。
# $A$は先ほど定義した行列とします。

# In[104]:


S[0].round(2)


# ### フロベニウスノルムの計算
# 
# フロベニウスノルムはnumpyのlinalgで簡単に計算できます。

# In[106]:


np.linalg.norm(A, 'fro').round(2)


# また、以下の性質を利用して計算することも出来ます。
# 二つの結果が一致していることが分かります。
# 
# $$
# \|A\|_{\mathrm{F}}^{2}=\operatorname{tr}\left(A A^{\top}\right)=\operatorname{tr}\left(A^{\top} A\right)
# $$

# In[109]:


np.sqrt(np.trace(A.T @ A)).round(2)


# ### スペクトル半径
# 
# 正方行列の最大固有値の絶対値をスペクトル半径といい、ページランクの収束性や連立方程式の収束性にも影響を及ぼします。

# In[114]:


np.abs(np.max(np.linalg.eig(np.array([[1,2,1], [2,1,3], [3,1,2]]))[0])).round(2)


# ## まとめ
# 
# 線型代数は本当に奥が深い。学生時代、サラリーマン時代と今まで2回ほどまとまって勉強する機会を設けたが、まだまだ理解していない事が多い。
# 推薦システムでも論文を読んでいると様々な性質を利用して理論が進んでいくので、一つ一つしっかりと理解していきたい。

# ## 参考文献
# 
# - [1][現代線形代数](https://www.amazon.co.jp/%E7%8F%BE%E4%BB%A3%E7%B7%9A%E5%BD%A2%E4%BB%A3%E6%95%B0-%E2%80%95%E5%88%86%E8%A7%A3%E5%AE%9A%E7%90%86%E3%82%92%E4%B8%AD%E5%BF%83%E3%81%A8%E3%81%97%E3%81%A6%E2%80%95-%E6%B1%A0%E8%BE%BA-%E5%85%AB%E6%B4%B2%E5%BD%A6/dp/4320018818)
