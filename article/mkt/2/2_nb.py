#!/usr/bin/env python
# coding: utf-8

# ## 準備
#   
# 教科書では計算は主にエクセルによる関数で実行されています。エクセルはGUI上の操作性は抜群なのですが、外部のWebシステムと連携するためのAPIのライブラリやデータ分析ツールとの連携が十分でないため、本サイトではpythonにより教科書と同じ計算を行います。そのための準備です。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/mkt/2/2_nb.md)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/mkt/2/2_nb.ipynb)
# 
# ### 筆者の環境
# 筆者の環境です。

# In[1]:


get_ipython().system('sw_vers')


# In[2]:


get_ipython().system('python -V')


# 必要なライブラリを読み込みます。

# In[3]:


import numpy as np
import scipy
from scipy.stats import binom

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print("numpy version :", np.__version__)
print("matplotlib version :", matplotlib.__version__)
print("sns version :",sns.__version__)


# ## 概要
# 
#  巻末解説2では、筆者たちがよく利用する6つのツールについて、具体例を踏まえて解説してくれています。実際のビジネスの場で必要になったらこれを参考にして適用すると良いでしょう。
# 
# 1. ガンマ・ポアソン・リーセンシーモデル
# - 「最近いつ買ったか」、「最近いつ訪れたか」というデータ（最近の購入時期：Recency）から、相対的にどのブランド、どの施設、どの時期に資源を集中すべきか教えてくれます。
# 
# 2. 負の二項分布  
# - 消費者世帯パネルの自社ブランドのデータと、実際の売り上げ高との違いを補正してくれるツールです。予測時のトライアル率、リピート率、購入回数のベンチマークにも非常に有効という事です。
# 
# 3. カテゴリーの進出順位モデル
# - 新しく作られたカテゴリーに、どの程度のシェアが取れるか教えてくれます。また、マーケティングの計画に基づくシェアのシミュレーションが出来ます。
# 
# 4. トライアルモデル・リピートモデル
# - コンセプト・テスト、コンセプト・ユース・テストのデータと世帯パネル・データを使い、新製品の発売①年目の売り上げを予測できます。
# 
# 5. 平均購入額・量モデル (VPP Model: Volume per Purchase)
# - 製品のサイズを決める手助けをしてくれます。
# 
# 6. デリシュレーNBDモデル
# - 巻末解説1の1−6で説明した、教科書の表1−4のコルゲートの四半期購入率、四半期購入回数、100％ロイヤル顧客の割合の予測、NBDのカテゴリーKのデリシュレーSがどのような計算がされているか、その具体的な例が示されています。
# 
# ## 2-1. ガンマ・ポアソン・リーセンシー・モデル
# 
# 「最近いつ買ったか」、「最近いつ訪れたか」というデータから$m$と$k$を計算し、NBDモデルを構築する事が出来ます。NBDモデルを記述する式は、何度も紹介したとおり、以下の様に計算できます。
# 
# $$
# P\left(r \right) = \frac{\left(1 + \frac{M}{K} \right)^{-K} \cdot \Gamma\left(K + r \right)}{\Gamma\left(r + 1 \right)\cdot \Gamma\left(K \right)} \cdot \left(\frac{M}{M+K} \right)^r
# $$
# 
# ある製品の期間$t$の対応する平均値$M$を$mt$とし、$K$を$k$と表記します。浸透率は100%から一度もこの製品を購入しない人を引けば良いので、
# 
# <div>
# $$
# \begin{aligned}
# P(t) &=1-P_0\left(r=0 \right) \\
# &= 1 - \frac{\left(1 + \frac{mt}{k} \right)^{-k} \cdot \Gamma\left(k + 0 \right)}{\Gamma\left(0 + 1 \right)\cdot \Gamma\left(k \right)} \cdot \left(\frac{mt}{mt+k} \right)^0  \\
# &= 1 - \left(1 + \frac{mt}{k} \right)^{-k}
# \end{aligned}
# $$
# </div>
# 
# となります。よってある期間$t$と$t-1$における浸透率は、
# 
# $$
#  P\left(t \right) - P\left(t-1 \right) = 
# \left(1+\frac{m\times t}{k} \right)^{-k} - \left(1+\frac{m\times \left(t-1 \right)}{k} \right)^{-k}
# $$
# 
# となります。
# 
# これを任意の期間に適用するために、二つの変数$t_1$と$t_2$を用いて、以下の様に$f \left(x \right)$と定義します。
# 
# $$
# f\left(t_1,t_2,m,k \right) = \left(1+\frac{m\times t_1}{k} \right)^{-k} - \left(1+\frac{m\times t_2}{k} \right)^{-k}
# $$
# 
# 
# 教科書の表10-1は共通の関数$f\left(x \right)$を用いて、以下の様に表現することが出来ます。
# <div style="display:table;margin: 0 auto;width:70%">
# <table>
#   <tr>
#     <th>ガンマ分布</th>
#     <th>実測値</th>
#   </tr>
#   <tr>
#     <td>$\displaystyle  f\left(t_1=0,t_2= \frac{14}{31}\right) $ </td>
#     <td>43.9%</td>
#   </tr>
#   <tr>
#     <td>$\displaystyle  f\left(t_1=\frac{14}{31},t_2=1 \right) $ </td>
#     <td>25.6%</td>
#   </tr>
#   <tr>
#     <td>$\displaystyle  f\left(t_1=1,t_2=2 \right) $ </td>
#     <td>19.1%</td>
#   </tr>
#   <tr>
#     <td>$\displaystyle  f\left(t_1=2,t_2=3 \right) $ </td>
#     <td>5.1%</td>
#   </tr>
#   <tr>
#     <td>$\displaystyle  f\left(t_1=3,t_2=4 \right) $ </td>
#     <td>1.5%</td>
#   </tr>
#   <tr>
#     <td>$\displaystyle  f\left(t_1=4,t_2=5 \right) $ </td>
#     <td>0.7%</td>
#   </tr>
#   <tr>
#     <td>$\displaystyle  f\left(t_1=5,t_2=6 \right) $ </td>
#     <td>1.4%</td>
#   </tr>
#   <tr>
#     <td>$\displaystyle  f\left(t_1=6,t_2=\infty \right) $ </td>
#     <td>2.7%</td>
#   </tr>
# </table>
# </div>
# 
# 
# ### scipyのcurve_figによる m,k の導出
# 
#  一般に非線形の関数に対して最小二乗法によるfittingを行うには、[scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)のcurve_fitモジュールを利用します。scipyの[ウェブサイト](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)によると
# 
# ```python
#   scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-inf, inf), method=None, jac=None, **kwargs)[source]
# ```
# となっており、さらに、$xdata$と$ydata$は
# 
# ```bash
#   xdata : array_like
#   The independent variable where the data is measured. Must be an M-length sequence or an (k,M)-shaped array for functions with k predictors.
#   
#   ydata : array_like
#   The dependent data, a length M array - nominally f(xdata, ...).
# ```
# と定義されています。教科書のfittingの問題を解くためには、
# 
# $$ f\left(t_1,t_2,m,k \right) =\left(1+\frac{m\times t_1}{k} \right)^{-k} -  \left(1+\frac{m\times t_2}{k} \right)^{-k} $$
# 
# という上記で定義した関数 $\displaystyle f\left(x \right)$に対して、２変数関数のfitting問題を解く事になります。２変数は期間を指定する２変数（２週間〜１ヶ月の購入数を求める際は、$ \displaystyle t_1=\frac{14}{31}, t_2=1$となります）が必要で、以下のように二次元配列で定義します。
# 
# ```python
# x = np.array([
#   [0.0 ,14/31 ,1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ],
#   [14/31 ,1.0  ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 , 10000.0]
# ])
# ```
# 
# x[0]が$t_1$の配列、x[1]が$t_2$の配列となります。x[1,7]=10000.0としているのは、本来は$\infty$となりますが、実際の数値計算では無限大は受けいられないので、事実上無限大となる10000を入れています。この値は100程度でも問題ありません。
# 
# fittingする実際のコードは以下の通りです。

# In[4]:


import json
import numpy as np

from scipy.optimize import curve_fit 
from scipy.special import gamma

def _get_delta_nbd(x, m, k):
  return (1 + m * x[0] / k )**(-k) - (1 + m * x[1] / k )**(-k) 

x = np.array([
  [0.0   ,14/31 ,1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ],
  [14/31 ,1.0   ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 , 10000.0]
])

y = [0.439, 0.256, 0.191, 0.051, 0.015, 0.007, 0.014, 0.027]

parameters, covariances = curve_fit(_get_delta_nbd, x, y)
print('parameters  : ', parameters)
print('covariances : ', covariances)


# 結果として得られた$m$と$k$は、
# 
# <div>
# $$
# \begin{aligned}
# m&= 1.378 \\
# k&= 4.144
# \end{aligned}
# $$
# </div>
# 
# となり、筆者が利用した$m$と$k$
# 
# <div>
# $$
# \begin{aligned}
# m&= 1.37552 \\
# k&= 4.061
# \end{aligned}
# $$
# </div>
# 
# とほぼ等しい値になっています。

# ## 2-2. 負の二項分布
# 
# 本項では、実際の売上高のデータとパネル・データから得られるデータの差分を用いて、パネル・データを補正する方法が解説されています。
# 
# まず、パネル・データによってわかるのは以下の通りです。
# 
# - (A) : 世帯数
# - (B) : 浸透率
# - (C) : 平均購入回数
# - (D) : 平均購入個数
# - (E) : 平均購入単価
# 
# 具体的な値は下記の表10−2を参照してください。教科書ではP281です。ここから、
# 
# <div class="box26">
#   <span class="box-title">パネル・データによる売上高</span>
#   売上高=世帯数×浸透率×平均購入回数×平均購入個数×平均購入単価
# </div>
# としてパネル・データ上の売上高が求められます。
# 
# また、実績としてわかっているのは以下の通りです。
# - 売上高
# 
# このパネル・データ上の売上高と実績としての売上高の比を利用して、パネル・データや様々なパラメタの補正を行います。
# 
# そのために、教科書ではいくつの重要な仮定が設定されています。この仮定をしっかり頭に入れることが以後の計算をスムーズに理解する一助となります。

# ### 仮定
# - 今売上の実績：58.9億円
# - パネルデータによる売上高：41.2億円(売り上げ実績比: 70%, AxBxCxDxEによって求められる)
# - 1回当たり平均購入個数はパネル・データと現実において同一   
# - 1回当たり平均購入単価はパネル・データと現実において同一   
# - $K$はパネル・データと現実において同一
# 
# となり、繰り返しますが、あくまでも実績としてわかっているのは、「売上高」のみです。教科書の例で言うと、売上が58.9億円という事だけがわかっています。
# 
# <div style="display:table;margin: 0 auto;width:70%">
# <table>
#   <caption>表10-2 あるブランドのパネル・データの補正</caption>
#   <caption>一年間の世帯パネル・データの補正</caption>
# 
#   <tr>
#     <th></th>
#     <th>項目</th>
#     <th>補正前</th>
#     <th>補正後</th>
#   </tr>
# 
#   <tr>
#     <td>(A)</td>
#     <td>平成20年の総世帯数（千）</td>
#     <td>49973</td>
#     <td>49973</td>
#   </tr>
# 
#   <tr>
#     <td>(B)</td>
#     <td>浸透率</td>
#     <td>15.0%</td>
#     <td>17.4%</td>
#   </tr>
# 
#   <tr>
#     <td>(C)</td>
#     <td>平均購入回数</td>
#     <td>2.50</td>
#     <td>3.07</td>
#   </tr>
# 
#   <tr>
#     <td>(D)</td>
#     <td>1回当たり平均購入個数</td>
#     <td>1.10</td>
#     <td>1.10</td>
#   </tr>
# 
#   <tr>
#     <td>(E)</td>
#     <td>1回当たり平均購入単価</td>
#     <td>200円</td>
#     <td>200円</td>
#   </tr>
# 
#   <tr>
#     <td>(F)</td>
#     <td>2回以上購入者比率</td>
#     <td>50%</td>
#     <td>55%</td>
#   </tr>
# 
#   <tr>
#     <td>(G)</td>
#     <td>年間売り上げ高(AxBxCxDxE)</td>
#     <td>41.2億円</td>
#     <td>58.9億円</td>
#   </tr>
# 
#   <tr>
#     <td>(H)</td>
#     <td>実績に対するGの比率</td>
#     <td>70%</td>
#     <td>100%</td>
#   </tr>
# 
# </table>
# </div>
# 
# 
# 
# <div style="display:table;margin: 0 auto;width:70%">
# <table>
#   <caption>表10-3 補正の計算</caption>
#   <caption>1年間の世帯パネル・データ</caption>
# 
#   <tr>
#     <th></th>
#     <th>項目</th>
#     <th>補正前</th>
#     <th>補正後</th>
#   </tr>
# 
#   <tr>
#     <td>(I)</td>
#     <td>ブランドの$m$:(BxCxD)</td>
#     <td>0.4125</td>
#     <td>0.5893</td>
#   </tr>
# 
#   <tr>
#     <td>(J)</td>
#     <td>ブランドの$k$</td>
#     <td>0.09899</td>
#     <td>0.09899</td>
#   </tr>
# 
#   <tr>
#     <td>(K)</td>
#     <td>$P_0$(1回も買わない確率)</td>
#     <td>85.00%</td>
#     <td>82.53%</td>
#   </tr>
# 
#   <tr>
#     <td>(L)</td>
#     <td>$P_1$(1回買う確率)</td>
#     <td>6.79%</td>
#     <td>7.00%</td>
#   </tr>
# 
#   <tr>
#     <td>(M)</td>
#     <td>$P_{+2}=100\%-P_0-P_1$</td>
#     <td>8.21%</td>
#     <td>10.47%</td>
#   </tr>
# 
#   <tr>
#     <td>(N)</td>
#     <td>モデルによる2回以上の購入者比率:$\left(\frac{M}{B} \right)$</td>
#     <td>54.76%</td>
#     <td>59.95%</td>
#   </tr>
# 
# </table>
# </div>

# ### 補正のステップ
# 1. ブランドの$m=$浸透率×平均購入回数×平均購入個数
# 2. ブランドの$k$
# $$ P\left(r \right) = \frac{\left(1 + \frac{M}{K} \right)^{-K} \cdot \Gamma\left(K + r \right)}{\Gamma\left(r + 1 \right)\cdot \Gamma\left(K \right)} \cdot \left(\frac{M}{M+K} \right)^r $$
# に、$\displaystyle K=k, M=m=0.4125, r=0$を代入することにより、
# 
# $$
# P_0=\frac{\left(1+\frac{m}{k} \right)^{-k}\cdot \Gamma\left(k+0 \right)}{\Gamma\left(0+1 \right)\cdot \Gamma\left(k \right)}=\left(1+\frac{0.4125}{k}  \right)^{-k}  =0.85
# $$
# 
# という非線形方程式を得ます。ここで$\displaystyle P_0$は一回も購入していない確率なので、(1-浸透率)から計算できて、
# 
# $$
# P_0=1 - 015 = 0.85
# $$
# 
# となることを利用しています。

# #### 非線形方程式を数値計算によって解く
# $k$を求める方程式
# 
# $$
# \left(1+\frac{0.4125}{k}  \right)^{-k}  =0.85
# $$
# 
# は非線形であり、解析に解く事は出来ません。数値計算法により、コンピュータを用いて解く事になります。ここではpythonのnewton法によって解を得ます。教科書では、エクセルによって$k$の値を得ていますが、どちらでも構いません。
# 
# 結果として、
# $$
# k=0.09899
# $$
# を得ます。

# #### python code
# pythonのコードは以下の通りです。

# In[5]:


from scipy.optimize import newton

MIN_k = 0
MAX_k = 1.0

def check_k(k):
  if MIN_k < k and k < MAX_k:
    return True
  else:
    return False

def get_k(m, P0):
  
  def func(k, m=m, P0=P0):
    return (1 + m / k) ** (-1 * k) - P0

  k = None
  try:
    for initial_k in [(i + 1) * 0.01 for i in range(100)]:
      k = newton(func, initial_k)
      if check_k(k):
        return k
    else:
      if not check_k(k):
        return None
  except:
    return None

m = 0.4125
P0 = 0.85

print("k = {:,.5f}".format(get_k(m, P0)))


# となり、pythonを用いても教科書とほぼ等しい値が得られています。

# #### 3. 1回買う確率であるP_1
# $P_1$は$P_0$と同じように、
# 
# $$ P\left(r \right) = \frac{\left(1 + \frac{m}{k} \right)^{-k} \cdot \Gamma\left(k + r \right)}{\Gamma\left(r + 1 \right)\cdot \Gamma\left(k \right)} \cdot \left(\frac{m}{m+k} \right)^r $$ 
# 
# に、$\displaystyle k=0.09899, m=0.4125, r=1$を代入するだけです。ただ、ガンマ関数が含まれているので、pythonやエクセルによる計算が必要です。 
# 
# <div>
# $$ 
# \begin{aligned}
# P_1&=P\left(1 \right) \\
# &= \frac{\left(1 + \frac{0.4125}{0.09899} \right)^{-0.09899} \cdot \Gamma\left(0.09899 + 1 \right)}{\Gamma\left(1 + 1 \right)\cdot \Gamma\left(0.09899 \right)} \\
# & \quad  \quad  \quad \times \left(\frac{0.4125}{0.4125+0.09899} \right)^1 \\
# &= 0.0679
# \end{aligned}
# $$ 
# </div>
# 
# となります。
# 
# #### 4. 2回以上買う確率P_{2+}
# 1から$P_0$と$P_1$を引けば良いので、
# 
# $$
# P\_{2+}=1-0.85-0.06709=0.0821
# $$
# 
# となります。
# 
# #### 5. モデルによる購入者の中で2回以上買う人の割合（補正前）
# これは単純に比を取るだけです。
# 
# $$
# \frac{P\_{2+}}{1-P_0} =\frac{0.0821}{1-0.85} =\frac{0.0821}{0.1500}=0.5476
# $$
# となります。
# 
# ### 具体的な補正の計算
# #### 6. $P_0$の計算
# $m$は実際の売上高とパネル・データ上の売上の比(0.7)によって補正されます。補正後の$m$を$m'$とします。
# 
# $$
# m'=\frac{m}{0.7}=\frac{0.4125}{0.7}=0.5893
# $$
# 
# と単純に補正されます。事前の仮定より$k$はパネル・データでも実際のデータでも共通なので、$k'=k=0.09899$となります。$k'$は補正後の$k$と言う意味です。この$k'$を用いて、$P_0$は以下の様に補正されます。
# 
# $$
# P_0=\left(1+\frac{0.5893}{0.09899} \right)^{-0.09899}=0.8253
# $$
# 
# また、補正後の浸透率($\tau'$と定義、補正前の浸透率を$\tau$と定義)も
# 
# $$
# \tau'=1-0.8253=0.1747
# $$
# 
# と計算できます。
# 
# #### 7. 補正後の平均購入回数
# 1.で求めたように、$m=$浸透率×平均購入回数×平均購入個数なので、
# 
# <div>
# $$
# \begin{aligned}
# \text{補正後の平均購入回数}&=\frac{補正後のm}{補正後の浸透率\rm \times 1回当たり平均購入個数} \\
# &=\frac{m'}{\tau' \times 1.1} = \frac{0.5893}{0.1747 \times 1.1}=3.07
# \end{aligned}
# $$
# </div>
# となります。
# 
# #### 8. 購入者の中で2回以上買う人の比率
# これも同様に補正後の$P_0$と$P_1$を計算するだけです。計算には、
# $$ P\left(r \right) = \frac{\left(1 + \frac{m^\prime}{k^\prime} \right)^{-k^\prime} \cdot \Gamma\left(k^\prime+ r \right)}{\Gamma\left(r + 1 \right)\cdot \Gamma\left(k^\prime \right)} \cdot \left(\frac{m^\prime}{m^\prime+k^\prime} \right)^r $$
# 
# を利用します。
# それぞれ補正後の値に$'$をつけると、
# 
# <div>
# $$
# \begin{aligned}
# P_{0}' &= 0.8253 \\
# P_{1}' &= 0.0700 \\
# P_{2+}' &=0.1047  \\
# \end{aligned}
# $$
# </div>
# 
# よって、
# 
# $$
# \frac{P\_{2+}^\prime}{1-P_0^\prime} =\frac{0.1047}{1-0.8253} =0.5995
# $$
# となります。
# 
# #### 9. 2回以上購入者比率
# こちらは、パネル・データの値に、モデルによる2回以上の購入者比率の比で補正するだけです。
# 
# <div>
# $$
# \begin{aligned}
# & \text{2回以上購入者比率} \\
# &=\text{パネル・データの値} \\
# &\quad \quad \times \frac{\text{補正後のモデルによる2回以上購入者比率}}{\text{補正前のモデルによる2回以上購入者比率}} \\
# &=0.5 \times \frac{0.5995}{0.5476}=0.5474
# \end{aligned}
# $$
# </div>
# 
# となり、補正されます。

# ## 2-3. カテゴリーの進出順位モデル
# 
# 本節では、
# 
# <div class="box26">
#   <span class="box-title">モデルの意味</span>
#   マーケットのシェアに関するシミュレーションが可能 
# </div>
# 
# となる公式が示されています。対応可能な商品のカテゴリは、
# 
# - 柔軟剤
# - 衣料用液体洗剤
# - フリーズドライ
# - コーヒー
# 
# となります。
# 
# ### 公式
# <div>
# $$
# \begin{aligned}
# & \text{マーケットシェアのパイオニアブランドに対する比率} \\
# &= \left(a\right)^{-0.49} \times \left(b\right)^{1.11} \times \left(c\right)^{0.28} \times \left(d\right)^{0.07} 
# \end{aligned}
# $$
# </div>
# 
# ここで、
# - a : 参入順位
# - b : 相対的好意度
# - c : 宣伝費の比率
# - d : 間合いの年数
# となります。
# 
# ### 例
# 教科書では具体的な例が示されています。
# 
# - パイオニアブランド(トップのシェアを持つブランド)のシェア:35%
# - 参入順位:4
# - 相対的好意度:0.9
# - 広告宣伝費率:0.7
# - 3番目の製品と同じ年に参入(間合いの年数):1
# 
# <div>
# $$
# \begin{aligned}
# & \text{予測されるシェア} \\
# &=0.35 \times \left(\text{4}\right)^{-0.49} \times \left(\text{0.9}\right)^{1.11} \left(\text{0.7}\right)^{0.28} \times \left(\text{1}\right)^{0.07} = 0.14285
# \end{aligned}
# $$
# </div>
# 
# となり、シェアは14%になります。

# ### python code
# あまり必要ないですが、pythonの計算コードを記載します。

# In[6]:


pioneer_share = 0.35
order = 4
m = 0.9
cost = 0.7
entry = 1

prediceted_share = pioneer_share*order**(-0.49)*m**(1.11)*cost**(0.28)*entry**(0.07)

print('予想されるシェア率 = {:,.3f}'.format(prediceted_share))


# これが実際に正しく予測できるかという問題は別として、この公式から、今後新しく市場に参入した際のシェアを予測できるという点でかなり有意義の高い公式になります。
# 
# ## 2-4. トライアルモデル・リピートモデル
# 
# 本節では、
# 
# - コンセプト・テスト
# - コンセプト・ユース・テスト
# - 世帯パネル・データ
# 
# の値から、新製品の1年目の売上を予測する事について説明がなされています。
# 
# ### a) トライアル・モデル、リピートモデル
# 
# <div class="box1">
#   売上高=年間のトライアルによる売上＋年間のリピーターによる売上
# </div>
# 
# ### 定義
# - トライアルによる売上= (Pop) ×(トライアル率) ×(トライアルVPP)
# - リピーターによる売上= (Pop) x (トライアル率) x (リピート率) x (リピート回数) x (リピートVPP)
# 
# 
# ### b) 各項目の説明
# - Pop： 消費者全体・全体世帯数の数
# - トライアル率：一年間に初めて対象の製品を購入した人のPopの割合
# - リピート率：一年間に初めて購入した人の内、もう一度一年間に購入した人の割合
# - リピート回数：リピートした人の平均購入回数から一回（トライアル分）を除いた回数
# - トライアルVPP：トライアル時の平均購入金額
# - リピートVPP：リピート時の平均購入金額
# 
# ### c) 例
# 
# #### 条件
# - 全世帯の10%が発売から1年間にある新製品のシャンプーを購入
# - 購入者の30%が期間内に少なくとももう一度購入
# - リピーターの平均購入回数は2.5回
# - トライアル時の平均購入金額は383円(365円x1.05)
# - リピート時の平均購入金額は475円(431円x1.10)
# 
# <div class="box1">
#   売上高 <br>
#   = 4997万世帯x10%x383円 + 4997万世帯x10%x30%x1.5x475円<br>
#   = 19.1億円 + 10.7億円 = 29.8億円
# </div>
# 
# 本節は、トライアル率さえパネルデータから導き出せれば、それほど理解することは難しくないと思われます。

# ## 2-5. 平均購入額・量モデル (VPP Model: Volume per Purchase)
# 本節は特に数学的な面の説明も必要ないと思われるため、省略します。

# ## 2-6. デリシュレーNBDモデル 
# ### 更新中
