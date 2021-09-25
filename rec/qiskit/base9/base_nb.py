#!/usr/bin/env python
# coding: utf-8

# ## HHLアルゴリズム
# 
# qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
# 個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。
# 
# qiskitのウェブサイト通りに勉強を進めています。
# 
# - https://qiskit.org/textbook/ja/ch-applications/hhl_tutorial.html
# 
# 私の拙いブログでqiskitがRec（推薦システム）のカテゴライズしいたのは、すべてHHLを理解するためでした。現在、推薦システムに興味があり、開発などを行っていますが、そこで重要なのが連立一次方程式を解く事です。連立一次方程式は、数理モデルをコンピュータを利用して解く場合に高い確率で利用されますが、推薦システムもUser-Item行列から如何にしてユーザーエンゲージメントの高い特徴量を抽出出来るかという事が重要になってきます。
# 
# よって、量子コンピュータを利用して高速に連立一次方程式を解く事を目標に量子アルゴリズムの復習を開始したわけですが、ようやく目的までたどり着きました。
# 
# 
# ### github
# - jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base9/base_nb.ipynb)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base9/base_nb.ipynb)
# 
# ### 筆者の環境

# In[2]:


get_ipython().system('sw_vers')


# In[3]:


get_ipython().system('python -V')


# 基本的なライブラリをインポートしそのバージョンを確認しておきます。

# In[105]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import qiskit
import json

import matplotlib.pyplot as plt
import numpy as np
import math

from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from qiskit.visualization import plot_histogram

dict(qiskit.__qiskit_version__)


# ## 共役勾配法
# 
# 復習の意味を込めて、古典アルゴリズムである共役勾配法の復習をします。
# 正定値行列である$A$を係数とする連立一次方程式、
# 
# $$
# A \boldsymbol{x}=\boldsymbol{b}
# $$
# 
# の解$x$を反復法を用いて数値計算的に解く方法になります。反復法ですので、計算の終了を判定する誤差$(\epsilon)$が必要になります。

# $A$,$x$,$b$は以下の様な行列になります。
# 
# $$
# A = \left(\begin{array}{cccc}
# a_{11} & a_{12} & \cdots & a_{1 n} \\
# a_{21} & a_{22} & \cdots & a_{2 n} \\
# \vdots & \vdots & \ddots & \vdots \\
# a_{n 1} & a_{n 2} & \cdots & a_{n n}
# \end{array}\right),\quad x=\left(\begin{array}{c}
# x_{1} \\
# x_{2} \\
# \vdots \\
# x_{n}
# \end{array}\right), \quad b=\left(\begin{array}{c}
# b_{1} \\
# b_{2} \\
# \vdots \\
# b_{n}
# \end{array}\right)
# $$

# 行列の表式で書くと以下の通りです。
# 
# $$
# \left(\begin{array}{cccc}
# a_{11} & a_{12} & \cdots & a_{1 n} \\
# a_{21} & a_{22} & \cdots & a_{2 n} \\
# \vdots & \vdots & \ddots & \vdots \\
# a_{n 1} & a_{n 2} & \cdots & a_{n n}
# \end{array}\right)\left(\begin{array}{c}
# x_{1} \\
# x_{2} \\
# \vdots \\
# x_{n}
# \end{array}\right)=\left(\begin{array}{c}
# b_{1} \\
# b_{2} \\
# \vdots \\
# b_{n}
# \end{array}\right)
# $$

# 次に次のように定義される関数$f(x)$を考えます。
# 
# $$
# f(\boldsymbol{x})=\frac{1}{2}(\boldsymbol{x}, A \boldsymbol{x})-(\boldsymbol{b}, \boldsymbol{x})
# $$
# 
# $(-,-)$は、ベクトルの内積を計算する演算子です。
# 
# $$
# (\boldsymbol{x}, \boldsymbol{y})=\boldsymbol{x}^{T} \boldsymbol{y}=\sum_{i=1}^{n} \boldsymbol{x}_{i} \boldsymbol{y}_{i}
# $$
# 
# 成分で表示すると以下の様になります。
# 
# $$
# f(x)=\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} a_{i j} x_{i} x_{j}-\sum_{i=1}^{n} b_{i} x_{i}
# $$

# ここで、　$x_k$で微分すると、
# 
# $$
# \frac{\partial f(x)}{\partial x_{k}}=\frac{1}{2} \sum_{i=1}^{n} a_{i k} x_{i}+\frac{1}{2} \sum_{j=1}^{n} a_{k j} x_{j}-b_{k}
# $$
# 
# となります。

# $A$はエルミート行列なので、
# 
# $$
# \frac{\partial f(x)}{\partial x_{i}}=\sum_{j=1}^{n} a_{i j} x_{j}-b_{i}=0
# $$
# 
# となります。

# これを一般化すると、
# 
# $$
# \nabla f(x)=\left(\begin{array}{c}
# \frac{\partial f}{\partial x_{1}} \\
# \vdots \\
# \frac{\partial f}{\partial x_{n}}
# \end{array}\right)=A\boldsymbol{x}-b = 0
# $$
# 
# となり、関数$f(x)$の最小値となる$x$を求める事が、$A\boldsymbol{x}-b = 0$を解く事と同じである事が分かります。

# ### アルゴリズム
# 
# 上記の通り、共役勾配法(CG法)は、関数$f(x)$を最小化することに帰着されます。
# 
# 通常の勾配法だと、等高線の垂直方向に探索するため効率は悪くなります。
# 
# $$
# -\nabla f(x)=\left(\begin{array}{c}
# \frac{\partial f}{\partial x_{1}} \\
# \vdots \\
# \frac{\partial f}{\partial x_{n}}
# \end{array}\right)=-b + A\boldsymbol{x}
# $$
# 
# $f(x)$は楕円を形成しますが、その楕円に線形変換をかけて、楕円を円に変換すると、等高線の垂直方向が最小点への最短距離となるため、効率的な解を求めることが出来るという考えです。
# 
# ![svg](base_nb_files_local/cg.svg)

# そのために、ある$x^{(0)}$を出発点に、以下の漸化式に従って最小値とする$x$を求めます。
# 
# $$
# x^{(k+1)}=x^{(k)}+\alpha_{k} p^{(k)}
# $$
# 
# ここで、$p^{(k)}$は解を探索する方向ベクトルです。また、$r^{(k)}$は解との残差ベクトルです。
# 
# $$
# r^{(0)}=b-A x^{(0)}, \quad {p}^{(0)}={r}^{(0)}
# $$
# 
# $k+1$回目のステップでの$f(x^{(k+1)})$を最小にする$\alpha_k$を求めます。
# 
# $$
# \begin{aligned}
# &f(x^{(k+1)}) \\
# &=f\left(x^{(k)}+\alpha_{k} p^{(k)}\right) \\
# &=\frac{1}{2} \alpha_{k}{ }^{2}\left(p^{(k)}, A p^{(k)}\right)-\alpha_{k}\left(p^{(k)}, b-A x^{(k)}\right)+f\left(x^{(k)}\right)
# \end{aligned}
# $$
# 
# これは単純な二次関数なので、$\alpha_k$を求める事が出来ます。
# 
# 
# $$
# \alpha_{k}=\frac{\left(p^{(k)}, b-A x^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}=\frac{\left(p^{(k)}, r^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}
# $$
# 
# 

# 次に$r$を更新するために、$r^{(k+1)}$を求めます。
# $$
# \begin{aligned}
# &r^{(k+1)}=b-A x^{(k+1)} \\
# &r^{(k)}=b-A x^{(k)} \\
# &r^{(k+1)}-r^{(k)}=A x^{(k+1)}-A x^{(k)}=\alpha_{k} A p^{(k)}
# \end{aligned}
# $$
# 
# よって、
# 
# $$
# r^{(k+1)}=r^{(k)}-\alpha_{k} A p^{(k)}
# $$
# 
# となります。

# この$r^{(k+1)}$がある一定の閾値以下であれば、探索を終了します。
# 
# $$
# \left\|\boldsymbol{r}^{(k+1)}\right\|<\varepsilon
# $$

# 閾値以下以外の場合、次に$p$を更新するために、$p^{(k+1)}$を求めます。$p^{(k+1)}$は、残差$r^{(k+1)}$と$p^{(k)}$の定数倍のベクトルで構成されます。
# 
# $$
# p^{(k+1)}=r^{(k+1)}+\beta_{k} p^{(k)}
# $$
# 
# この定数$\beta_k$は、$p^{(k)}$と$p^{(k+1)}$が$A$に対して共役になるように取ります。
# 
# $$
# \left(p^{(k+1)}, A p^{(k)}\right)=\left(r^{(k+1)}+\beta_{k} p^{(k)}, A p^{(k)}\right)=\left(r^{(k+1)}, A p^{(k)}\right)+\beta_{k}\left(p^{(k)}, A p^{(k)}\right)
# $$
# 
# これより、
# $$
# \beta_{k}=-\frac{\left(r^{(k+1)}, A p^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}
# $$
# 
# となります。
# 
# これを残差がある一定の閾値以下に収束するまで繰り返します。

# ここで、
# 
# $$
# \left(p^{(i)}, A p^{(j)}\right)=0 \quad (i \neq j)
# $$
# 
# が成立するので、共役関係を持つベクトルを方向ベクトルととする勾配法という事で、共役勾配法という名前なのでしょうか。
# 
# また、残差ベクトルについても、
# 
# $$
# \left(r^{(i)}, r^{(j)}\right)=0 \quad (i \neq j)
# $$
# 
# という直交関係が成立します。

# 一次独立である$r_k$の個数は高々$N$個しかないので、計算は$N$回以内に収束します。
# 
# この他に、前処理付き共役勾配法やクリロフ部分空間と言った解析的な話、収束性についての詳細は以下を参考にしています。
# 
# - [Matrix Computations](https://www.amazon.co.jp/Computations-Hopkins-Studies-Mathematical-Sciences/dp/1421407949)
# - [数値解析入門](https://www.amazon.co.jp/%E5%A4%A7%E5%AD%A6%E6%95%B0%E5%AD%A6%E3%81%AE%E5%85%A5%E9%96%809-%E6%95%B0%E5%80%A4%E8%A7%A3%E6%9E%90%E5%85%A5%E9%96%80-%E9%BD%8A%E8%97%A4-%E5%AE%A3%E4%B8%80/dp/413062959X)

# ### 計算量
# $s$は行列$A$の非0要素の割合、$\kappa$は行列$A$の最大固有値と最小固有値の比$\displaystyle \left| \frac{\lambda_{max}}{\lambda_{min}}\right|$、$\epsilon$  は精度です．
# 
# $$
# O(N s \kappa \log (1 / \varepsilon))
# $$
# 
# これは$N$に比例する形になっているため、$s\sim \log N$の場合、$N\log N$に比例する事になります。

# ## HHLアルゴリズム
# 
# ### HHLの仮定
# 
# HHLはある程度の仮定の下に成立するアルゴリズムです。
# 
# - ローディングを実施する効果的なオラクルが存在
# - ハミルトニアンシミュレーションと解の関数の計算が可能
# - $A$がエルミート行列
# - $\mathcal{O}\left(\log (N) s^{2} \kappa^{2} / \epsilon\right)$で計算可能
# - 古典アルゴリズムが完全解を返すが、HHLは解となるベクトルを与える関数を近似するだけ
# 
# ### HHLのアウトライン
# 
# ![svg](base_nb_files_local/hhl.svg)

# ### 量子回路へのマッピング
# 
# 連立一次方程式を量子アルゴリズムで解くには、$Ax=b$を量子回路にマッピングする必要があります。それは、$b$の$i$番目の成分は量子状態$|b\rangle$の$i$番目の基底状態の振幅に対応させるという方法です。また、当然ですが、その際は$\displaystyle \sum_i |b_i|^2 = 1$という規格化が必要です。
# 
# $$
# Ax=b \rightarrow A|x\rangle = |b\rangle
# $$
# 
# となります。

# ### スペクトル分解
# 
# $A$はエルミート行列なので、スペクトル分解が可能です。$A$のエルミート性を暗幕的に仮定していましたが、
# $$
# A'=\left(\begin{array}{ll}
# 0 & A \\
# A & 0
# \end{array}\right)
# $$
# 
# とすれば、$A'$はエルミート行列となるため、問題ありません。よって、$A$は固有ベクトル$|u_i\rangle$とその固有値$\lambda_i$を利用して、以下の様に展開できます。
# 
# $$
# A=\sum_{j=0}^{N-1} \lambda_{j}\left|u_{j}\right\rangle\left\langle u_{j}\right|
# $$
# 
# よって、逆行列は以下の様になります。
# 
# $$
# A^{-1}=\sum_{j=0}^{N-1} \lambda_{j}^{-1}\left|u_{j}\right\rangle\left\langle u_{j}\right|
# $$

# $u_i$は$A$の固有ベクトルなので、$|b\rangle$はその重ね合わせで表現できます。おそらくこれが量子コンピュータを利用する強い動機になっていると思います。
# 
# $$
# |b\rangle=\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle
# $$
# 
# 本来であれば、$A$の固有ベクトルを計算できなければこの形で$|b\rangle$を用意することが出来ませんが、量子コンピュータでは$|b\rangle$を読み込むことで、自動的にこの状態を用意することが出来ます。
# 
# 最終的には以下の式の形を量子コンピュータを利用して求める事になります。
# 
# $$
# |x\rangle=A^{-1}|b\rangle=\sum_{j=0}^{N-1} \lambda_{j}^{-1} b_{j}\left|u_{j}\right\rangle
# $$

# ### 1. データのロード
# 
# 対象となる$|b\rangle$の各データの振幅を量子ビット$n_b$にロードします。
# 
# $$
# |0\rangle_{n_{b}} \mapsto|b\rangle_{n_{b}}
# $$

# ### 2. QPEの適用
# 
# 量子位相推定を利用して、ユニタリ演算子$U=e^{i A t}$ の$|b\rangle$の位相を推定します。
# 
# $|b\rangle$は上述の様に以下になります。
# 
# $$
# |b\rangle=\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle
# $$

# $U$は展開して整理すると、以下の様になります。
# 
# $$
# \begin{aligned}
# &U=e^{i A t}=\sum_{k=0}^{\infty} \frac{\left(i A t\right)^{k}}{k !}\\
# &=\sum_{k=0}^{\infty} \frac{\left(i t\right)^{k}}{k !}\left(\sum_{j=0}^{N-1} \lambda_{j}|u_{j}\rangle\langle u_{j} |\right)^{k}\\
# &=\sum_{k=0}^{\infty} \frac{(i t)^{k}}{k !} \sum_{j=0}^{N-1} \lambda_{j}^{k}\left|u_{j}\right\rangle \langle u_{j} |\\
# &=\sum_{j=0}^{N-1}\left(\sum_{k=0}^{\infty} \frac{\left(i t\right)^{k}}{k_{i}} \lambda_{j}^{k}\right)|u_{j}\rangle \langle u_{j}| \\
# &=\sum_{r=0}^{N-1} e^{i \lambda_{j} t}\left|u_{j}\right\rangle\left\langle u_{j}\right|
# \end{aligned}
# $$

# $U$を$|b\rangle$に作用させると、
# 
# $$
# \begin{aligned}
# U|b\rangle &=U\left(\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle\right) \\
# &=\sum_{j^{\prime}=0}^{N-1} e^{i \lambda j^{\prime} t}\left|h_{j^{\prime}}\right\rangle\left\langle h_{j^{\prime}}\right| \cdot\left(\sum_{j=0}^{N-1} b_{j}\left|h_{j}\right\rangle\right) \\
# &=\sum_{j=0}^{N-1} b_{j} e^{i \lambda_{j} t}\left|u_{j}\right\rangle
# \end{aligned}
# $$
# 
# となり、量子位相推定を利用して、$\lambda_j$の量子状態 $|\tilde{\lambda_j} \rangle_{n_l}$ を求める事が出来ます。
# 
# $\tilde{\lambda_{j}}$は、$\displaystyle 2^{n_{l}} \frac{\lambda_{j} t}{2 \pi}$に対する$n_l$-bitバイナリ近似となります。

# $t=2\pi$とし、$\lambda_l$が、$n_l$ビットで正確に表現できるとすると、量子位相推定は以下の様に表現できます。
# 
# $$
# \operatorname{QPE}\left(e^{i A 2 \pi}, \sum_{j=0}^{N-1} b_{j}|0\rangle_{n_{l}}\left|u_{j}\right\rangle_{n_{b}}\right)=\sum_{j=0}^{N-1} b_{j}\left|\lambda_{j}\right\rangle_{n_{l}}\left|u_{j}\right\rangle_{n_{b}}
# $$

# ### 3. 補助量子ビットの利用
# 
# 天下り的ですが、制御回転ゲートを利用して、$|v_i\rangle$を量子ビットの振幅として取り出す方法を考えます。
# 
# ここでは、以下の文献を参考にしています。
# 
# - [嶋田義皓. 量子コンピューティング](https://www.amazon.co.jp/%E9%87%8F%E5%AD%90%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0-%E5%9F%BA%E6%9C%AC%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E3%81%8B%E3%82%89%E9%87%8F%E5%AD%90%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%BE%E3%81%A7-%E6%83%85%E5%A0%B1%E5%87%A6%E7%90%86%E5%AD%A6%E4%BC%9A%E5%87%BA%E7%89%88%E5%A7%94%E5%93%A1%E4%BC%9A/dp/4274226212)
# - https://www2.yukawa.kyoto-u.ac.jp/~qischool2019/mitaraiCTO.pdf
# 
# 
# 二つの量子ビットを考えます。一つは、制御回転ゲートの制御ビット、二つ目は制御回転ゲートの対象ビットです。
# 
# 唐突ですが、制御ビットに、$\left|b\left(\frac{1}{\pi} \cos ^{-1} v_{i}\right)\right\rangle$の状態を入力し、対象ビットには$|0\rangle$。を入力します。この対象ビットが補助量子ビットになります。
# 
# $b(\cdots)$は二進数表記である事を示し、$b_k(\cdots)$は、二進数表記の$k$ビット目の値を表します。まとめると、以下の様になります。
# 
# $$
# \begin{aligned}
# &b\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)=d_{0} d_{1} d_2 \cdot \cdots d_{m-1} \\
# &b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)=d_k \\
# &\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)=\frac{d_{0}}{2}+\frac{d_{1}}{4}+\frac{d_{2}}{8} \cdots \frac{d_{m-1}}{2^{m}} \quad \left(0 \leqq \frac{2}{\pi} \cos ^{-1}\left(v_{i}\right) \leqq 1\right)
# \end{aligned}
# $$
# 
# これから、
# 
# 
# $$
# \begin{aligned}
# \frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)&=\frac{1}{2} b_{0}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{1}\right)\right)+\frac{1}{4} b_{1}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)+ \cdots \\
# &=\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{n}\right)\right) 2^{-k-1} \cdots (1)
# \end{aligned}
# $$

# $\displaystyle b_{k}\left(\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)\right)$を制御ビットとして、$R_{y}\left(2^{-k-1}\pi\right)$である制御回転ゲートをかけることを考えます。
# 
# $$
# \begin{aligned}
# &\prod_{k=0}^{m-1} R_{y}\left(b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle \\
# &=R_{y}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle \\
# \end{aligned}
# $$

# 回転ゲートは、以下の様になります。
# 
# $$
# R_{y}(\theta)|0\rangle=\cos \frac{\theta}{2}|0\rangle+\sin \frac{\theta}{2}|1\rangle
# $$

# この式と(1)を利用して、
# 
# $$
# \begin{aligned}
# &\cos\frac{1}{2}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right) \\
# &=\cos\left(\frac{1}{2}\times\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)\times \pi\right) = v_i
# \end{aligned}
# $$

# となり、こちらを利用して、
# 
# $$
# R_{y}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle=v_{i}|0\rangle+\sqrt{1-v_{i}^{2}}|1\rangle
# $$
# 
# を得る事ができます。ここで、$\displaystyle v_i = \frac{1}{\lambda_i}$とすることで、補助量子ビットを、
# 
# $$
# \frac{1}{\lambda_{j}}|0\rangle + \sqrt{1-\frac{1}{\lambda_{j}^{2}}}|1\rangle
# $$
# 
# と計算することが出来きます。$\frac{1}{\lambda_j}$が振幅として得られたので、これを利用して連立一次方程式を解けそうです。

# ### 4. 逆量子位相推定を利用
# 
# 量子位相推定の逆変換を行うと、以下の様になります。
# 
# $$
# \sum_{j=0}^{N-1} b_{j}|0\rangle_{n_{l}}\left|u_{j}\right\rangle_{n_{b}}\left(\frac{1}{\lambda_{j}}|0\rangle+\sqrt{1-\frac{1}{\lambda_{j}^{2}}}|1\rangle\right)
# $$
# 

# ### 5. 補助量子ビットの測定
# 
# 補助量子ビットを測定し、$|0\rangle$が測定された場合、
# 
# $$
# \left(\sqrt{\frac{1}{\sum_{j=0}^{N-1}\left|b_{j}\right|^{2} /\left|\lambda_{j}\right|^{2}}}\right) \sum_{j=0}^{N-1} \frac{b_{j}}{\lambda_{j}}|0\rangle_{n_{l}}\left|u_{j}\right\rangle_{n_{b}}
# $$
# 
# となり、解の形となっています。

# ## 計算量の比較
# 
# ### 量子アルゴリズム
# 
# 
# $$
# O(s \kappa \operatorname{poly} \log (s \kappa / \varepsilon)))
# $$
# 
# 行列 $A$に対して、スパース性 $(s \sim O(\operatorname{poly} \log N))$を仮定できる場合、
# 
# $$
# O(s \kappa \operatorname{poly} \log (s \kappa / \varepsilon))) \sim O(s \kappa \operatorname{poly} \log N \operatorname{poly} \log (s \kappa / \varepsilon))
# $$
# 
# となります。

# ### 共役勾配法
# $$
# O(N s \kappa \log (1 / \varepsilon))
# $$
# 
# これより、量子アルゴリズムの方が、指数関数的な速度向上が見込まれます。

# ## qiskitで実装
# 
# qiskitのサイトに従って、実装してみます。
# 
# 結論から言うと、私の環境ではサイト通りの結果になりませんです。DepricateWarningが出ていて、それかなと思って色々やってみたのですが、結果が一致しなかったので、後に詳細な原因を探ろうと思います。

# In[108]:


from qiskit import Aer
from qiskit.circuit.library import QFT
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms import HHL, NumPyLSsolver
from qiskit.aqua.components.eigs import EigsQPE
from qiskit.aqua.components.reciprocals import LookupRotation
from qiskit.aqua.operators import MatrixOperator
from qiskit.aqua.components.initial_states import Custom
import numpy as np

from qiskit import aqua

dict(qiskit.__qiskit_version__)


# In[107]:


print(aqua.__version__)


# In[85]:


def create_eigs(matrix, num_ancillae, num_time_slices, negative_evals):
    ne_qfts = [None, None]
    if negative_evals:
        num_ancillae += 1
        ne_qfts = [QFT(num_ancillae - 1), QFT(num_ancillae - 1).inverse()]

    return EigsQPE(MatrixOperator(matrix=matrix),
                   QFT(num_ancillae).inverse(),
                   num_time_slices=num_time_slices,
                   num_ancillae=num_ancillae,
                   expansion_mode='suzuki',
                   expansion_order=2,
                   evo_time=None, # np.pi*3/4, #None,  # This is t, can set to: np.pi*3/4
                   negative_evals=negative_evals,
                   ne_qfts=ne_qfts)


# In[86]:


def fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    fidelity = state_fidelity(solution_hhl_normed, solution_ref_normed)
    print("Fidelity:\t\t %f" % fidelity)


# In[87]:


matrix = [[1, -1/3], [-1/3, 1]]
vector = [1, 0]


# In[95]:


orig_size = len(vector)
matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

# Initialize eigenvalue finding module
eigs = create_eigs(matrix, 3, 100, False)
num_q, num_a = eigs.get_register_sizes()

# Initialize initial state module
init_state = Custom(num_q, state_vector=vector)

# Initialize reciprocal rotation module
reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
           init_state, reci, num_q, num_a, orig_size)


# In[96]:


result = algo.run(QuantumInstance(Aer.get_backend('statevector_simulator')))
print("Solution:\t\t", np.round(result['solution'], 5))

result_ref = NumPyLSsolver(matrix, vector).run()
print("Classical Solution:\t", np.round(result_ref['solution'], 5))

print("Probability:\t\t %f" % result['probability_result'])
fidelity(result['solution'], result_ref['solution'])


# In[92]:


print("circuit_width:\t", result['circuit_info']['width'])
print("circuit_depth:\t", result['circuit_info']['depth'])
print("CNOT gates:\t", result['circuit_info']['operations']['cx'])


# qiskitのサイトだと、
# 
# ```text
# Solution:		 [1.13586-0.j 0.40896+0.j]
# Classical Solution:	 [1.125 0.375]
# ```
# 
# となっています、今回の結果は、
# 
# ```text
# Solution:		 [ 0.66576-0.j -0.38561+0.j]
# Classical Solution:	 [1.125 0.375]
# ```
# 
# となり、あまり良い結果が得られていないようです。。。

# メモがてら、DepricateのログにあるCustomとQuantumCircuit.initializeのdocを残しておきます。

# In[42]:


get_ipython().run_line_magic('pinfo', 'Custom')

init signature:
Custom(
    num_qubits: int,
    state: str = 'zero',
    state_vector: Union[numpy.ndarray, qiskit.aqua.operators.state_fns.state_fn.StateFn, NoneType] = None,
    circuit: Union[qiskit.circuit.quantumcircuit.QuantumCircuit, NoneType] = None,
) -> None


# In[48]:


get_ipython().run_line_magic('pinfo', 'QuantumCircuit.initialize')

Args:
    params (str or list or int):
        * str: labels of basis states of the Pauli eigenstates Z, X, Y. See
            :meth:`~qiskit.quantum_info.states.statevector.Statevector.from_label`.
            Notice the order of the labels is reversed with respect to the qubit index to
            be applied to. Example label '01' initializes the qubit zero to `|1>` and the
            qubit one to `|0>`.
        * list: vector of complex amplitudes to initialize to.
        * int: an integer that is used as a bitmap indicating which qubits to initialize
           to `|1>`. Example: setting params to 5 would initialize qubit 0 and qubit 2
           to `|1>` and qubit 1 to `|0>`.
    qubits (QuantumRegister or int):
        * QuantumRegister: A list of qubits to be initialized [Default: None].
        * int: Index of qubit to initialized [Default: None].

Returns:
    qiskit.circuit.Instruction: a handle to the instruction that was just initialized


# ## まとめ
# 
# このHHLアルゴリズムを元に、量子推薦アルゴリズムやそれにインスパイアされた古典アルゴリズムが発表されています。
# 
# - https://arxiv.org/pdf/1603.08675.pdf
# - https://arxiv.org/pdf/1807.04271.pdf
# 
# インスパイアされた古典アルゴリズムでは、推薦システムで利用されるUser-Item行列はしばしば低ランク近似が利用可能（莫大な数のユーザーのカテゴリはその数に比べてはるかに少ない）であるため、それを利用して解となる状態を高速でサンプリングすることが出来るというのが概要です。
# 
# いずれ近いうちにその内容もまとめたいと思います。

# ## 参考文献
# - [嶋田義皓. 量子コンピューティング](https://www.amazon.co.jp/%E9%87%8F%E5%AD%90%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0-%E5%9F%BA%E6%9C%AC%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E3%81%8B%E3%82%89%E9%87%8F%E5%AD%90%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%BE%E3%81%A7-%E6%83%85%E5%A0%B1%E5%87%A6%E7%90%86%E5%AD%A6%E4%BC%9A%E5%87%BA%E7%89%88%E5%A7%94%E5%93%A1%E4%BC%9A/dp/4274226212)
# - https://www2.yukawa.kyoto-u.ac.jp/~qischool2019/mitaraiCTO.pdf
# - [Matrix Computations](https://www.amazon.co.jp/Computations-Hopkins-Studies-Mathematical-Sciences/dp/1421407949)
# - [数値解析入門](https://www.amazon.co.jp/%E5%A4%A7%E5%AD%A6%E6%95%B0%E5%AD%A6%E3%81%AE%E5%85%A5%E9%96%809-%E6%95%B0%E5%80%A4%E8%A7%A3%E6%9E%90%E5%85%A5%E9%96%80-%E9%BD%8A%E8%97%A4-%E5%AE%A3%E4%B8%80/dp/413062959X)
