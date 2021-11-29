#!/usr/bin/env python
# coding: utf-8

# ## Graph Convolutional Networks
# 
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/gcn/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/gcn/base_nb.ipynb)
# 
# ### 筆者の環境

# In[2]:


get_ipython().system('sw_vers')


# In[3]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import json
import math
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

print(nx.__version__)


# In[2]:


# torchの読み込み
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print(torch.__version__)


# ## Graph Convolutional Networks 
# 
# ## 参照論文
#   - [1][SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)
#   - [2][Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf)
#   - [3][LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/pdf/2002.02126.pdf)
#   - [4][UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation](https://arxiv.org/abs/2110.15114)
# 

# ![](./1.png)

# $$
# R_{u, i}=\left\lbrace\begin{array}{lr}
# 1, & \text { if }(u, i) \text { interaction is observed } \\
# 0, & \text { otherwise }
# \end{array}\right.
# $$
# 
# $$
# \begin{array}{|l|r|r|r|r|}
# \hline & \text { item_1 } & \text { item_2 } & \text { item_3 } & \text { item_4 } \\
# \hline \text { user_1 } & 0 & 1 & 0 & 0 \\
# \hline \text { user_2 } & 0 & 0 & 1 & 1 \\
# \hline \text { user_3 } & 1 & 0 & 0 & 0 \\
# \hline \text { user_4 } & 0 & 1 & 0 & 0 \\
# \hline \text { user_5 } & 1 & 0 & 1 & 0 \\
# \hline
# \end{array}
# $$

# In[133]:


m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()

print(loss)

# 入力のサイズはN x C = 3 x 5
input = torch.randn(3, 5, requires_grad=True)

print(input)
# ターゲットの各要素は0 <=値<Cである必要があります
target = torch.tensor([1, 0, 4])
print(target)
output = loss(m(input), target)
print(output)
output.backward()


# In[ ]:





# In[ ]:





# In[5]:


G = nx.karate_club_graph()

pos = nx.spring_layout(G)
color = []
for node in G.nodes:
  print(G.node)
  if G.node[node]['club'] == 'Mr. Hi':
    color.append('r')
  elif G.node[node]['club'] == 'Officer':
    color.append('b')

plt.figure(figsize=(5, 5))
#nx.draw(G, pos=pos, node_size=200, node_color=color, with_labels=True)
nx.draw_networkx(G, pos=pos, node_size=200, node_color=color, with_labels=True)


# In[6]:


# networkx, matplotlib の import
import networkx as nx
import matplotlib.pyplot as plt

# グラフの構築
G = nx.karate_club_graph()

# レイアウトの取得
pos = nx.spring_layout(G)

# 可視化
plt.figure(figsize=(6, 6))
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_nodes(G, pos)
plt.axis('off')
plt.show()


# レコメンドにおけるグラフは、通常2部グラフとして表現されます。

# In[24]:


B = nx.Graph()
B.add_nodes_from([1,2,3,4,5], bipartite=0)
B.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'g'], bipartite=1)
B.add_edges_from([(1,'a'), (1,'b'), (2, 'a'), (2, 'c'), (3, 'g'), (4, 'd'), (5, 'e'), (5, 'f')])

node_color = []
for i in range(5):
  node_color.append('red')
for i in range(7):
  node_color.append('blue')
print(node_color)
print(nx.bipartite.sets(B))
top = nx.bipartite.sets(B)[0]
pos = nx.bipartite_layout(B, top)
nx.draw(B, pos=pos, with_labels=True, node_color=node_color)
plt.show()


# In[34]:


B = nx.Graph()
B.add_nodes_from([1,2,3,4], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(['abc','bcd','cef'], bipartite=1)
B.add_edges_from([(1,'abc'), (1,'bcd'), (2,'bcd'), (2,'cef'), (3,'cef'), (4,'cef')])

top = nx.bipartite.sets(B)[0]
pos = nx.bipartite_layout(B, top)
nx.draw(B, pos=pos, with_labels=True, node_color=['green','green','green','green','blue','blue','blue'])
plt.show()


# In[66]:


B = None
B = nx.Graph()
B.add_nodes_from([1,2,3,4], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(['a','b','c', 'd', 'e','f'], bipartite=1)
B.add_edges_from([(1,'a'), (1,'b'), (2,'b'), (2,'c'), (3,'c'), (4,'a'), (4, 'd'),(4, 'e'), (4,'f')])

node_color = []
for i in range(4):
  node_color.append('red')
for i in range(6):
  node_color.append('green')


top = nx.bipartite.sets(B)[0]
pos = nx.bipartite_layout(B, top)
nx.draw(B, pos=pos, with_labels=True, node_color=node_color)
# nx.draw(B, pos=pos, with_labels=True, node_color=['green','green','green','green','blue','blue'])
plt.show()


# レコメンドにおいて、嗜好行列
# 
# 
# |  | item_1 | item_2 | item_3 | item_4 | item_5 |
# | :---: | :---: | :---: | :---: | :---: | :---: |
# | user_1 | 0 | 1 | 1 | 0 | 0 |
# | user_2 | 0 | 0 | 0 | 1 | 0 |
# | user_3 | 1 | 0 | 0 | 0 | 1 |
# | user_4 | 1 | 1 | 1 | 1 | 0 |
# | user_5 | 0 | 0 | 0 | 0 | 1 |
# | user_6 | 0 | 0 | 1 | 0 | 0 |
# 
# 

# $$
# \begin{equation}
# \mathbf{R}_{u, i}=\left\{\begin{array}{l}
# 1 \\
# 0
# \end{array}\right.
# \end{equation}
# $$

# $$
# R_{u, i}=\left\{\begin{array}{lr}
# 1, & \text { if }(u, i) \text { interaction is observed } \\
# 0, & \text { otherwise }
# \end{array}\right.
# $$

# $$
# \boldsymbol{R} \in\{0,1\}^{|\mathcal{U}| \times|\mathcal{I}|}
# $$

# $$
# \mathbf{R}_{u, i}=\left\{\begin{array}{lr}
# 1, & \text { if }(u, i) \text { interaction is observed } \\
# 0, & \text { otherwise }
# \end{array}\right.
# $$

# レコメンドにおいて、嗜好行列
# 
# 
# |  | item_1 | item_2 | item_3 | item_4 | item_5 |
# | :---: | :---: | :---: | :---: | :---: | :---: |
# | user_1 | 0 | 1 | 1 | 0 | 0 |
# | user_2 | 0 | 0 | 0 | 1 | 0 |
# | user_3 | 1 | 0 | 0 | 0 | 1 |
# | user_4 | 1 | 1 | 1 | 1 | 0 |
# | user_5 | 0 | 0 | 0 | 0 | 1 |
# | user_6 | 0 | 0 | 1 | 0 | 0 |
# 
# 
# 
# 
# $$
# A=\left(\begin{array}{cc}
# 0 & R \\
# R^{T} & 0
# \end{array}\right)
# $$

# $$
# \tilde{R}=D_{U}^{-\frac{1}{2}} R D_{I}^{-\frac{1}{2}}
# $$

# In[ ]:





# In[100]:


A = nx.adjacency_matrix(B).todense()
A


# networkxにはグラフラプラシアン$L$を求める関数が酔いされています。
# $$
# L=D-A
# $$

# In[101]:


L = nx.laplacian_matrix(B).todense()
L


# 次数行列は以下の様に求める事が出来ます。
# 
# $$
# D = L + A
# $$

# In[102]:


D = L + A
D


# In[88]:


import numpy as np
import numpy.linalg as LA

Lambda = np.linalg.eig(L)[0]


# In[89]:


U = np.linalg.eig(L)[1]


# In[90]:


U @ np.diagflat(Lambda) @ U.T


# In[ ]:





# ## グラフラプラシアンの正規化
# 
# $$
# \tilde{L}=I-\tilde{A}
# $$

# $$
# \Delta f=\frac{\partial^{2} f}{\partial x^{2}}+\frac{\partial^{2} f}{\partial y^{2}}+\frac{\partial^{2} f}{\partial z^{2}}
# $$

# In[121]:


nD = np.diagflat(np.power(np.array(np.sum(D, axis=0))[0] + 1e-15, -1/2))
nA = nD @ A @ nD
nL = np.eye(len(nA)) - nA
nLambda = np.linalg.eig(nL)[0]
nU = np.linalg.eig(nL)[1]
nU @ np.diagflat(nLambda) @ nU.T


# In[125]:


embedding = torch.nn.Embedding(num_embeddings=len(nA), embedding_dim=len(nA))
dir(embedding)


# $$
# \Delta h_{\omega}(x)=\lambda_{\omega} h_{\omega}(x)
# $$

# $$
# h_{\omega}(x)=e^{i \omega x}
# $$

# $$
# g(t) * f(t)=\sum_{\tau} g(\tau) f(t-\tau)
# $$

# $$
# F[g(t) * f(t)]=F[g(t)] F[f(t)]
# $$

# $$
# \begin{gathered}
# \hat{\mathbf{x}}=\mathbf{U}^{T} \mathbf{x} \\
# \mathbf{x}=\mathbf{U} \widehat{\mathbf{x}}
# \end{gathered}
# $$

# $$
# \mathcal{H}(\boldsymbol{L})=\operatorname{UDiag}\left(h\left(\lambda_{1}\right), \cdots, h\left(\lambda_{n}\right)\right) U^{T}
# $$

# In[ ]:





# In[135]:


import numpy as np
import random
from matplotlib import pyplot as plt

N_steps = 1000
prob = 0.5

def SimpleRandomWalk(N, p, line):

    position = np.empty(N)
    position[0] = 0
    pos_counter = 0

    steps = np.arange(N)

    #ランダムウォークスタート 
    for i in range(1,N):

        test = random.random()

        if test >= p:
            pos_counter += 1
        else:
            pos_counter -= 1

        position[i] = pos_counter

    plt.plot(steps, position, line)
    plt.xlabel('Steps taken')
    plt.ylabel('Distance from Starting Position')


    return position

position_distribution = []
for i in range(1000):
    p = SimpleRandomWalk(N_steps, prob, line="-")
    position_distribution.append(p[-1])

plt.figure()
plt.hist(position_distribution)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


from numpy import *
from matplotlib.pyplot import *
from scipy import stats

def quantum_walk(N):
    P = 2*N+1    # number of positions

    coin0 = array([1, 0])  # |0>
    coin1 = array([0, 1])  # |1>

    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    
    C00 = outer(coin0, coin0)  # |0><0| 
    C01 = outer(coin0, coin1)  # |0><1| 
    C10 = outer(coin1, coin0)  # |1><0| 
    C11 = outer(coin1, coin1)  # |1><1| 

    C_hat = (C00 + C01 + C10 - C11)/sqrt(2.)

    ShiftPlus = roll(eye(P), 1, axis=0)
    ShiftMinus = roll(eye(P), -1, axis=0)
    S_hat = kron(ShiftPlus, C00) + kron(ShiftMinus, C11)

    U = S_hat.dot(kron(eye(P), C_hat))

    posn0 = zeros(P)
    posn0[N] = 1     # array indexing starts from 0, so index N is the central posn
    psi0 = kron(posn0,(coin0+coin1*1j)/sqrt(2.))

    psiN = linalg.matrix_power(U, N).dot(psi0)

    prob = empty(P)
    for k in range(P):
        posn = zeros(P)
        posn[k] = 1     
        M_hat_k = kron( outer(posn,posn), eye(2))
        proj = M_hat_k.dot(psiN)
        prob[k] = proj.dot(proj.conjugate()).real
    return prob, P

N = 100

prob ,P = quantum_walk(N)
fig = figure()
ax = fig.add_subplot(111)

plot(arange(P), prob)
show()


# In[138]:


1 + 1


# In[140]:


np.e


# In[144]:


a = np.array([[1,2], [3,4]])
e = np.array([[1,0], [0,1]])
np.e ** e

a @ e


# In[146]:


np.exp(e)


# In[148]:


A


# In[149]:


# Aを固有値分解して、密度行列を得る

l, p = np.linalg.eig(A)


# In[167]:


p[:, 0]


# In[169]:


p[:, 0].T


# In[171]:


p[:, 0] @ p[:, 0].T


# In[172]:


for i in range(len(p)):
  _ += (p[:, i] @ p[:, i].T) * (p[:, i] @ p[:, i].T)

# _ = _ / np.sum(_, axis=1)
# np.sum(_, axis=1)
_


# In[156]:


len(p)


# In[174]:


np.array([[1,2], [3,4]]) / np.array([10,20])


# In[160]:


A


# In[161]:


plt.show()


# In[ ]:


from scipy.linalg import expm

expm(a)

