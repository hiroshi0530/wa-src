## Graph Convolutional Networks



### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/gcn/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/gcn/base_nb.ipynb)

### 筆者の環境


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G103



```python
!python -V
```

    Python 3.8.5


基本的なライブラリをインポートしそのバージョンを確認しておきます。


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import json
import math
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from tabulate import tabulate

print(nx.__version__)
```

    2.5



```python
# torchの読み込み
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print(torch.__version__)
```

    1.4.0


## Graph Convolutional Networks 



$$
R_{u, i}=\left\lbrace\begin{array}{lr}
1, & \text { if }(u, i) \text { interaction is observed } \\
0, & \text { otherwise }
\end{array}\right.
$$

$$
\begin{array}{|l|r|r|r|r|}
\hline & \text { item_1 } & \text { item_2 } & \text { item_3 } & \text { item_4 } \\
\hline \text { user_1 } & 0 & 1 & 0 & 0 \\
\hline \text { user_2 } & 0 & 0 & 1 & 1 \\
\hline \text { user_3 } & 1 & 0 & 0 & 0 \\
\hline \text { user_4 } & 0 & 1 & 0 & 0 \\
\hline \text { user_5 } & 1 & 0 & 1 & 0 \\
\hline
\end{array}
$$


```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.set_printoptions(threshold=10000000)

B = nx.Graph()

user_num = 7
item_num = 10

user_nodes = [i for i in range(user_num)]
item_nodes = [i for i in range(100, 100 + item_num)]

B.add_nodes_from(user_nodes, bipartite=0)
B.add_nodes_from(item_nodes, bipartite=1)

node_color = []

for u in user_nodes:
  node_color.append('red')
for i in item_nodes:
  node_color.append('blue')

edge_nodes = []
pos = {}
for _i, u in enumerate(user_nodes):
  pos[u] = np.array([-1, - 1.75 * _i])
  for _j, i in enumerate(item_nodes):
    pos[i] = np.array([1, - 1.75 * _j])
    if np.random.random() < 0.15:
      edge_nodes.append((u,i))

B.add_edges_from(edge_nodes)

nx.draw(B, pos=pos, with_labels=True, node_color=node_color)

plt.show()
```


    
![svg](base_nb_files/base_nb_8_0.svg)
    



```python
A = np.array(nx.adjacency_matrix(B).todense())
L = np.array(nx.laplacian_matrix(B).todense())
D = L + A
A
```




    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int64)




```python
A.shape
```




    (17, 17)




```python
# print(tabulate(A, headers=headers, tablefmt='github'))
print(tabulate(A, tablefmt='github'))
```

    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


```python

```

![](./1.png)


```python
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
```

    NLLLoss()
    tensor([[ 0.1064,  0.4910,  0.7079,  1.4534,  0.4364],
            [ 0.1860,  0.4660,  0.4329, -1.0039,  0.3334],
            [-0.1303,  0.4550,  0.3146,  0.0974,  0.7234]], requires_grad=True)
    tensor([1, 0, 4])
    tensor(1.5711, grad_fn=<NllLossBackward>)



```python

```


```python

```


```python
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
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-5-7c6845b86451> in <module>
          4 color = []
          5 for node in G.nodes:
    ----> 6   print(G.node)
          7   if G.node[node]['club'] == 'Mr. Hi':
          8     color.append('r')


    AttributeError: 'Graph' object has no attribute 'node'



```python
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
```


    
![svg](base_nb_files/base_nb_19_0.svg)
    


レコメンドにおけるグラフは、通常2部グラフとして表現されます。


```python
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
```

    ['red', 'red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']



    ---------------------------------------------------------------------------

    AmbiguousSolution                         Traceback (most recent call last)

    <ipython-input-24-1b3a087710d2> in <module>
         10   node_color.append('blue')
         11 print(node_color)
    ---> 12 print(nx.bipartite.sets(B))
         13 top = nx.bipartite.sets(B)[0]
         14 pos = nx.bipartite_layout(B, top)


    ~/anaconda3/lib/python3.8/site-packages/networkx/algorithms/bipartite/basic.py in sets(G, top_nodes)
        199         if not is_connected(G):
        200             msg = "Disconnected graph: Ambiguous solution for bipartite sets."
    --> 201             raise nx.AmbiguousSolution(msg)
        202         c = color(G)
        203         X = {n for n, is_top in c.items() if is_top}


    AmbiguousSolution: Disconnected graph: Ambiguous solution for bipartite sets.



```python
B = nx.Graph()
B.add_nodes_from([1,2,3,4], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(['abc','bcd','cef'], bipartite=1)
B.add_edges_from([(1,'abc'), (1,'bcd'), (2,'bcd'), (2,'cef'), (3,'cef'), (4,'cef')])

top = nx.bipartite.sets(B)[0]
pos = nx.bipartite_layout(B, top)
nx.draw(B, pos=pos, with_labels=True, node_color=['green','green','green','green','blue','blue','blue'])
plt.show()
```


    
![svg](base_nb_files/base_nb_22_0.svg)
    



```python
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
```


    
![svg](base_nb_files/base_nb_23_0.svg)
    


レコメンドにおいて、嗜好行列


|  | item_1 | item_2 | item_3 | item_4 | item_5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| user_1 | 0 | 1 | 1 | 0 | 0 |
| user_2 | 0 | 0 | 0 | 1 | 0 |
| user_3 | 1 | 0 | 0 | 0 | 1 |
| user_4 | 1 | 1 | 1 | 1 | 0 |
| user_5 | 0 | 0 | 0 | 0 | 1 |
| user_6 | 0 | 0 | 1 | 0 | 0 |



$$
\begin{equation}
\mathbf{R}_{u, i}=\left\{\begin{array}{l}
1 \\
0
\end{array}\right.
\end{equation}
$$

$$
R_{u, i}=\left\{\begin{array}{lr}
1, & \text { if }(u, i) \text { interaction is observed } \\
0, & \text { otherwise }
\end{array}\right.
$$

$$
\boldsymbol{R} \in\{0,1\}^{|\mathcal{U}| \times|\mathcal{I}|}
$$

$$
\mathbf{R}_{u, i}=\left\{\begin{array}{lr}
1, & \text { if }(u, i) \text { interaction is observed } \\
0, & \text { otherwise }
\end{array}\right.
$$

レコメンドにおいて、嗜好行列


|  | item_1 | item_2 | item_3 | item_4 | item_5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| user_1 | 0 | 1 | 1 | 0 | 0 |
| user_2 | 0 | 0 | 0 | 1 | 0 |
| user_3 | 1 | 0 | 0 | 0 | 1 |
| user_4 | 1 | 1 | 1 | 1 | 0 |
| user_5 | 0 | 0 | 0 | 0 | 1 |
| user_6 | 0 | 0 | 1 | 0 | 0 |




$$
A=\left(\begin{array}{cc}
0 & R \\
R^{T} & 0
\end{array}\right)
$$

$$
\tilde{R}=D_{U}^{-\frac{1}{2}} R D_{I}^{-\frac{1}{2}}
$$


```python

```


```python
A = nx.adjacency_matrix(B).todense()
A
```




    matrix([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=int64)



networkxにはグラフラプラシアン$L$を求める関数が酔いされています。
$$
L=D-A
$$


```python
L = nx.laplacian_matrix(B).todense()
L
```




    matrix([[ 2,  0,  0,  0, -1, -1,  0,  0,  0,  0],
            [ 0,  2,  0,  0,  0, -1, -1,  0,  0,  0],
            [ 0,  0,  1,  0,  0,  0, -1,  0,  0,  0],
            [ 0,  0,  0,  4, -1,  0,  0, -1, -1, -1],
            [-1,  0,  0, -1,  2,  0,  0,  0,  0,  0],
            [-1, -1,  0,  0,  0,  2,  0,  0,  0,  0],
            [ 0, -1, -1,  0,  0,  0,  2,  0,  0,  0],
            [ 0,  0,  0, -1,  0,  0,  0,  1,  0,  0],
            [ 0,  0,  0, -1,  0,  0,  0,  0,  1,  0],
            [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  1]], dtype=int64)



次数行列は以下の様に求める事が出来ます。

$$
D = L + A
$$


```python
D = L + A
D
```




    matrix([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=int64)




```python
import numpy as np
import numpy.linalg as LA

Lambda = np.linalg.eig(L)[0]
```


```python
U = np.linalg.eig(L)[1]
```


```python
U @ np.diagflat(Lambda) @ U.T
```




    matrix([[ 2.00000000e+00, -2.46981255e-16, -3.10515502e-16,
              2.42969707e-15, -1.00000000e+00, -1.00000000e+00,
              7.15465014e-16, -8.91935267e-16, -8.43993182e-16,
             -7.99586751e-16],
            [-1.36067373e-16,  2.00000000e+00, -1.70002901e-16,
              1.26461341e-15,  1.83186799e-15, -1.00000000e+00,
             -1.00000000e+00, -3.03656075e-16, -3.87629330e-16,
             -4.08529279e-16],
            [-3.66026653e-16, -2.81025203e-16,  1.00000000e+00,
              2.01227923e-16, -2.68882139e-16,  5.37764278e-17,
             -1.00000000e+00, -1.39667317e-16, -7.70490482e-17,
             -9.55338606e-17],
            [ 2.49214711e-15,  1.27849120e-15,  2.04697370e-16,
              4.00000000e+00, -1.00000000e+00, -1.89084859e-15,
              2.79290480e-16, -1.00000000e+00, -1.00000000e+00,
             -1.00000000e+00],
            [-1.00000000e+00,  1.88737914e-15, -1.56125113e-16,
             -1.00000000e+00,  2.00000000e+00, -1.86049093e-16,
              1.20563282e-16,  2.36884101e-17,  5.05086767e-16,
              5.74508954e-16],
            [-1.00000000e+00, -1.00000000e+00,  1.09287579e-16,
             -1.91513472e-15, -3.80771803e-16,  2.00000000e+00,
             -1.45022883e-15,  7.10896422e-16,  3.64826845e-16,
              3.39811090e-16],
            [ 6.11490025e-16, -1.00000000e+00, -1.00000000e+00,
              3.62557206e-16,  1.12757026e-16, -1.28889954e-15,
              2.00000000e+00, -2.57069936e-17,  5.76046858e-17,
              7.21857164e-17],
            [-8.98982582e-16, -3.19268586e-16, -1.46606211e-16,
             -1.00000000e+00,  6.27196883e-17,  6.90079740e-16,
             -2.91764406e-17,  1.22140609e+00,  1.77800863e-01,
             -3.99206957e-01],
            [-8.64809864e-16, -3.78955712e-16, -7.01101543e-17,
             -1.00000000e+00,  5.02484682e-16,  3.57887951e-16,
              6.10741328e-17,  1.77800863e-01,  6.00793043e-01,
              2.21406094e-01],
            [-8.20403432e-16, -3.99855662e-16, -8.85949667e-17,
             -1.00000000e+00,  5.71906868e-16,  3.32872196e-16,
              7.56551633e-17, -3.99206957e-01,  2.21406094e-01,
              1.17780086e+00]])




```python

```

## グラフラプラシアンの正規化

$$
\tilde{L}=I-\tilde{A}
$$

$$
\Delta f=\frac{\partial^{2} f}{\partial x^{2}}+\frac{\partial^{2} f}{\partial y^{2}}+\frac{\partial^{2} f}{\partial z^{2}}
$$


```python
nD = np.diagflat(np.power(np.array(np.sum(D, axis=0))[0] + 1e-15, -1/2))
nA = nD @ A @ nD
nL = np.eye(len(nA)) - nA
nLambda = np.linalg.eig(nL)[0]
nU = np.linalg.eig(nL)[1]
nU @ np.diagflat(nLambda) @ nU.T
```




    matrix([[ 1.00000000e+00,  6.46184495e-17, -5.05238212e-17,
              1.24856722e-15, -5.00000000e-01, -5.00000000e-01,
             -3.61039323e-16, -6.19510098e-16, -6.59214287e-16,
             -7.23905449e-16],
            [ 5.07406617e-17,  1.00000000e+00, -9.19403442e-16,
             -3.03576608e-16,  1.70002901e-16, -5.00000000e-01,
             -5.00000000e-01,  2.04890223e-16,  1.53518028e-17,
              1.89152714e-16],
            [-2.28766658e-17, -9.22872889e-16,  1.00000000e+00,
             -6.41847686e-17, -4.76181594e-16,  1.11022302e-16,
             -7.07106781e-01,  7.66347748e-17, -1.01105375e-16,
              2.25698524e-16],
            [ 1.25528928e-15, -2.48065457e-16, -9.71445147e-17,
              1.00000000e+00, -3.53553391e-01, -4.29344060e-16,
              2.77555756e-17, -5.00000000e-01, -5.00000000e-01,
             -5.00000000e-01],
            [-5.00000000e-01,  1.70002901e-16, -4.48426019e-16,
             -3.53553391e-01,  1.00000000e+00,  7.73252989e-16,
              4.51028104e-16,  2.60613411e-16,  2.69603637e-16,
              2.90740837e-16],
            [-5.00000000e-01, -5.00000000e-01,  1.04950770e-16,
             -4.21537805e-16,  7.73252989e-16,  1.00000000e+00,
             -5.37764278e-17,  1.59189670e-16,  2.01807468e-16,
              2.70442207e-16],
            [-3.46944695e-16, -5.00000000e-01, -7.07106781e-01,
             -3.46944695e-18,  4.50160742e-16, -5.29090660e-17,
              1.00000000e+00, -1.24436896e-17,  2.02146761e-16,
              4.58595776e-18],
            [-6.26448991e-16,  1.70195754e-16,  6.27569870e-17,
             -5.00000000e-01,  2.60613411e-16,  1.86945246e-16,
             -8.97424262e-18,  1.01158431e+00, -1.48312106e-02,
              3.24690156e-03],
            [-6.69622627e-16,  2.22906967e-17, -8.72275876e-17,
             -5.00000000e-01,  2.83047744e-16,  2.26093596e-16,
              1.81330079e-16, -1.48312106e-02,  1.00324690e+00,
              1.15843090e-02],
            [-7.27483316e-16,  1.88285353e-16,  2.10953374e-16,
             -5.00000000e-01,  2.91174518e-16,  2.93860974e-16,
             -2.14348944e-17,  3.24690156e-03,  1.15843090e-02,
              9.85168789e-01]])




```python
embedding = torch.nn.Embedding(num_embeddings=len(nA), embedding_dim=len(nA))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-e065ab5f8c7f> in <module>
    ----> 1 embedding = torch.nn.Embedding(num_embeddings=len(nA), embedding_dim=len(nA))
    

    NameError: name 'torch' is not defined


$$
\Delta h_{\omega}(x)=\lambda_{\omega} h_{\omega}(x)
$$

$$
h_{\omega}(x)=e^{i \omega x}
$$

$$
g(t) * f(t)=\sum_{\tau} g(\tau) f(t-\tau)
$$

$$
F[g(t) * f(t)]=F[g(t)] F[f(t)]
$$

$$
\begin{gathered}
\hat{\mathbf{x}}=\mathbf{U}^{T} \mathbf{x} \\
\mathbf{x}=\mathbf{U} \widehat{\mathbf{x}}
\end{gathered}
$$

$$
\mathcal{H}(\boldsymbol{L})=\operatorname{UDiag}\left(h\left(\lambda_{1}\right), \cdots, h\left(\lambda_{n}\right)\right) U^{T}
$$


```python

```


```python
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

```


    
![svg](base_nb_files/base_nb_52_0.svg)
    



    
![svg](base_nb_files/base_nb_52_1.svg)
    



```python

```


```python

```


```python
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

```


```python
1 + 1
```




    2




```python
np.e
```




    2.718281828459045




```python
a = np.array([[1,2], [3,4]])
e = np.array([[1,0], [0,1]])
np.e ** e

a @ e
```




    array([[1, 2],
           [3, 4]])




```python
np.exp(e)
```




    array([[2.71828183, 1.        ],
           [1.        , 2.71828183]])




```python
A
```




    matrix([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=int64)




```python
# Aを固有値分解して、密度行列を得る

l, p = np.linalg.eig(A)
```


```python
p[:, 0]
```




    matrix([[0.30505495],
            [0.13838447],
            [0.0396419 ],
            [0.6214658 ],
            [0.43721013],
            [0.20925188],
            [0.08400776],
            [0.29325964],
            [0.29325964],
            [0.29325964]])




```python
p[:, 0].T
```




    matrix([[0.30505495, 0.13838447, 0.0396419 , 0.6214658 , 0.43721013,
             0.20925188, 0.08400776, 0.29325964, 0.29325964, 0.29325964]])




```python
p[:, 0] @ p[:, 0].T
```




    matrix([[0.09305852, 0.04221487, 0.01209296, 0.18958122, 0.13337311,
             0.06383332, 0.02562698, 0.0894603 , 0.0894603 , 0.0894603 ],
            [0.04221487, 0.01915026, 0.00548582, 0.08600122, 0.06050309,
             0.02895721, 0.01162537, 0.04058258, 0.04058258, 0.04058258],
            [0.01209296, 0.00548582, 0.00157148, 0.02463609, 0.01733184,
             0.00829514, 0.00333023, 0.01162537, 0.01162537, 0.01162537],
            [0.18958122, 0.08600122, 0.02463609, 0.38621974, 0.27171114,
             0.13004289, 0.05220795, 0.18225084, 0.18225084, 0.18225084],
            [0.13337311, 0.06050309, 0.01733184, 0.27171114, 0.1911527 ,
             0.09148704, 0.03672904, 0.12821609, 0.12821609, 0.12821609],
            [0.06383332, 0.02895721, 0.00829514, 0.13004289, 0.09148704,
             0.04378635, 0.01757878, 0.06136513, 0.06136513, 0.06136513],
            [0.02562698, 0.01162537, 0.00333023, 0.05220795, 0.03672904,
             0.01757878, 0.0070573 , 0.02463609, 0.02463609, 0.02463609],
            [0.0894603 , 0.04058258, 0.01162537, 0.18225084, 0.12821609,
             0.06136513, 0.02463609, 0.08600122, 0.08600122, 0.08600122],
            [0.0894603 , 0.04058258, 0.01162537, 0.18225084, 0.12821609,
             0.06136513, 0.02463609, 0.08600122, 0.08600122, 0.08600122],
            [0.0894603 , 0.04058258, 0.01162537, 0.18225084, 0.12821609,
             0.06136513, 0.02463609, 0.08600122, 0.08600122, 0.08600122]])




```python
for i in range(len(p)):
  _ += (p[:, i] @ p[:, i].T) * (p[:, i] @ p[:, i].T)

# _ = _ / np.sum(_, axis=1)
# np.sum(_, axis=1)
_
```




    matrix([[1.55392348, 0.04957782, 0.04954792, 0.04955595, 0.04955739,
             0.04958729, 0.04955746, 0.04955746, 0.04956761, 0.04956761],
            [0.04957753, 1.55392043, 0.04955018, 0.04955718, 0.04955844,
             0.04958453, 0.0495585 , 0.0495585 , 0.04956735, 0.04956735],
            [0.04954444, 0.04954669, 1.55393626, 0.04958062, 0.04957838,
             0.049532  , 0.04957827, 0.04957827, 0.04956253, 0.04956253],
            [0.04955281, 0.04955416, 0.04958222, 1.55391814, 0.04957334,
             0.04954528, 0.04957327, 0.04957327, 0.04956375, 0.04956375],
            [0.04955435, 0.04955554, 0.04958022, 0.0495736 , 1.5539159 ,
             0.04954773, 0.04957235, 0.04957235, 0.04956398, 0.04956398],
            [0.04958919, 0.04958669, 0.04953507, 0.04954892, 0.04955141,
             1.55394754, 0.04955154, 0.04955154, 0.04956905, 0.04956905],
            [0.04955443, 0.04955561, 0.04958012, 0.04957355, 0.04957236,
             0.04954786, 1.55391579, 0.0495723 , 0.04956399, 0.04956399],
            [0.04955443, 0.04955561, 0.04958012, 0.04957355, 0.04957236,
             0.04954786, 0.0495723 , 1.55391579, 0.04956399, 0.04956399],
            [0.04956562, 0.04956562, 0.04956562, 0.04956562, 0.04956562,
             0.04956562, 0.04956562, 0.04956562, 1.55390944, 0.04956562],
            [0.04956562, 0.04956562, 0.04956562, 0.04956562, 0.04956562,
             0.04956562, 0.04956562, 0.04956562, 0.04956562, 1.55390944]])




```python
len(p)
```




    10




```python
np.array([[1,2], [3,4]]) / np.array([10,20])
```




    array([[0.1, 0.1],
           [0.3, 0.2]])




```python
A
```




    matrix([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=int64)




```python
plt.show()
```


```python
from scipy.linalg import expm

expm(a)
```


```python

```

$$
R_{u, i}=\left\lbrace\begin{array}{lr}
1, & \text { if }(u, i) \text { interaction is observed } \\
0, & \text { otherwise }
\end{array}\right.
$$

$$
\begin{array}{|l|r|r|r|r|}
\hline & \text { item_1 } & \text { item_2 } & \text { item_3 } & \text { item_4 } \\
\hline \text { user_1 } & 0 & 1 & 0 & 0 \\
\hline \text { user_2 } & 0 & 0 & 1 & 1 \\
\hline \text { user_3 } & 1 & 0 & 0 & 0 \\
\hline \text { user_4 } & 0 & 1 & 0 & 0 \\
\hline \text { user_5 } & 1 & 0 & 1 & 0 \\
\hline
\end{array}
$$


```python

```


```python
a = None
print(a)
```

    None



```python
class A():
  def __init__(self):
    self.text = None

  def __repr__(self):
    # __repr__ でNoneを返すとエラー
    return None

    # __repr__ でNoneを返すとエラー
    # これでもダメ
    # return self.text

a = A()
print(a.text)
print(a)
len(a.a)
```

    None



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-33-e8a5c7517652> in <module>
         13 a = A()
         14 print(a.text)
    ---> 15 print(a)
         16 len(a.a)


    TypeError: __str__ returned non-string (type NoneType)


- a.text でNoneの変数をprint出来るが、 a.__repr__ でprint()させようとするとError
