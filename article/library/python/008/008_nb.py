#!/usr/bin/env python
# coding: utf-8

# ## Python PuLP 数理最適化
# 
# これまで、pulpによる最適化計算をしたことがなかったので、基本的な使い方を参考記事に沿って実行してみます。
# 
# 以下が参考にさせていただいた記事になります。とてもわかりやすいです。
# 
# - [samuelladocoさんのqiita記事](https://qiita.com/samuelladoco/items/703bf78ea66e8369c455)
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/python/008/008_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/python/008/008_nb.ipynb)
# 
# ### 筆者の環境

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# ### 1. 線形最適化問題（線形計画問題）
# 
# 高校数学でやった記憶があります。
# 
# #### 最大化
# $$
# x + y 
# $$
# 
# #### 制約条件
# $$
# 2x + y \leq 2 \\\\
# x + 2y \leq 2 \\\\
# x \geq 0 \\\\
# y \geq 0 
# $$
# 

# In[3]:


import pulp
import sys

# 最適化（最大化）を宣言
prob = pulp.LpProblem('test', pulp.LpMaximize)

# 変数の宣言
x = pulp.LpVariable("xx", 0, sys.maxsize, pulp.LpContinuous)
y = pulp.LpVariable("yy", 0, sys.maxsize, pulp.LpContinuous)


# In[4]:


# 目的関数を宣言
prob += ( x + y, "Objective" )

# 制約条件を宣言
prob += ( 2 * x + y <= 2 , "Constraint_1" )
prob += ( x + 2 * y <= 2 , "Constraint_2" )


# In[5]:


prob


# In[6]:


result = prob.solve()


# In[7]:


print("計算結果")
print("*" * 8)
print(f"最適性 = {pulp.LpStatus[result]}")
print(f"目的関数値 = {pulp.value(prob.objective)}")
print(f"解 x = {pulp.value(x)}")
print(f"　 y = {pulp.value(y)}")
print("*" * 8)


# ### 2. 整数最適化問題（整数計画問題）
# 
# #### 最小化
# $$
# \sum_{i \in I} \sum_{j \in J} c_{ij}x_{ij}
# $$
# 
# #### 制約条件
# $$
# \sum_{j\in J}x_{ij} \leq 1 \\\\
# \sum_{i\in I}x_{ij} = 1 \\\\
# x_{ij} \in {0,1}
# $$

# In[8]:


import pulp
import time

# 作業員の集合（便宜上、リストを用いる）
I = ["Aさん", "Bさん", "Cさん"]

print(f"作業員の集合 I = {I}")

# タスクの集合（便宜上、リストを用いる）
J = ["仕事イ", "仕事ロ", "仕事ハ"]

print(f"タスクの集合 J = {J}")

# 作業員 i を タスク j に割り当てたときのコストの集合（一時的なリスト）
cc = [
    [ 1,  2,  3],
    [ 4,  6,  8],
    [10, 13, 16],
   ]

# cc はリストであり、添え字が数値なので、
# 辞書 c を定義し、例えばcc[0][0] は c["Aさん","仕事イ"] でアクセスできるようにする
c = {} # 空の辞書
for i in I:
  for j in J:
    c[i,j] = cc[I.index(i)][J.index(j)]

print("コスト c[i,j]: ")
for i in I:
  for j in J:
    print(f"c[{i},{j}] = {c[i,j]:2d},  ", end = "")
  print("")
print("")


# In[9]:


# 数理最適化を宣言
# pulp.LpMinimize => 最小化
# pulp.LpMaximize => 最大化

prob = pulp.LpProblem('prob', pulp.LpMinimize)


# In[10]:


# 変数集合を表す辞書
x = {} # 空の辞書
     # x[i,j] または x[(i,j)] で、(i,j) というタプルをキーにしてバリューを読み書き

# 0-1変数を宣言
for i in I:
  for j in J:
    x[i,j] = pulp.LpVariable(f"x({i},{j})", 0, 1, pulp.LpInteger)
    # 変数ラベルに '[' や ']' や '-' を入れても、なぜか '_' に変わる…？
# lowBound, upBound を指定しないと、それぞれ -無限大, +無限大 になる

# 内包表記も使える
# x_suffixes = [(i,j) for i in I for j in J]
# x = pulp.LpVariable.dicts("x", x_suffixes, cat = pulp.LpBinary) 

# pulp.LpContinuous : 連続変数
# pulp.LpInteger  : 整数変数
# pulp.LpBinary   : 0-1変数


# In[11]:


# 目的関数を宣言
prob += pulp.lpSum(c[i,j] * x[i,j] for i in I for j in J), "TotalCost"
# problem += sum(c[i,j] * x[i,j] for i in I for j in J)


# In[12]:


# 制約条件の宣言
for i in I:
  prob += sum(x[i,j] for j in J) <= 1, f"Constraint_leq_{i}"
  # 制約条件ラベルに '[' や ']' や '-' を入れても、なぜか '_' に変わる…？

# 各タスク j について、割り当てられる作業員数はちょうど1人
for j in J:
  prob += sum(x[i,j] for i in I) == 1, f"Constraint_eq_{j}"


# In[13]:


# 問題の式全部を表示
print("問題の式")
print(f"-" * 8)
print(prob)
print(f"-" * 8)
print("")


# In[14]:


# 計算
# ソルバー指定
solver = pulp.PULP_CBC_CMD()
# pulp.PULP_CBC_CMD() : PuLP付属のCoin-CBC
# pulp.GUROBI_CMD()   : Gurobiをコマンドラインから起動 (.lpファイルを一時生成)
# pulp.GUROBI()   : Gurobiをライブラリーから起動 (ライブラリーの場所指定が必要)

# 時間計測開始
time_start = time.perf_counter()

result_status = prob.solve(solver)
# solve()の()内でソルバーを指定できる
# 何も指定しない場合は pulp.PULP_CBC_CMD()

# 時間計測終了
time_stop = time.perf_counter()


# In[15]:


# （解が得られていれば）目的関数値や解を表示
print("計算結果")
print(f"*" * 8)
print(f"最適性 = {pulp.LpStatus[result_status]}, ", end="")
print(f"目的関数値 = {pulp.value(prob.objective)}, ", end="")
print(f"計算時間 = {time_stop - time_start:.3f} (秒)")
print("解 x[i,j]: ")
for i in I:
  for j in J:
    print(f"{x[i,j].name} = {x[i,j].value()},  ", end="")
  print("")
print(f"*" * 8)


# 1行1行、とても勉強になりました。ぜひリンク先に飛んでオリジナルの記事で勉強してみてください。
# 
# - [samuelladocoさんのqiita記事](https://qiita.com/samuelladoco/items/703bf78ea66e8369c455)
