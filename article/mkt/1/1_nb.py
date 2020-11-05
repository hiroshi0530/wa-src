
# coding: utf-8

# ## 準備
#   
# 教科書では計算は主にエクセルによる関数で実行されています。エクセルはGUI上の操作性は抜群なのですが、外部のWebシステムと連携するためのAPIのライブラリやデータ分析ツールとの連携が十分でないため、本サイトではpythonにより教科書と同じ計算を行います。そのための準備です。
# 
# ### github
# - githubのjupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/mkt/1/1_nb.md)
# 
# ### google colaboratory
# - google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/mkt/1/1_nb.ipynb)
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

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print("numpy version :", np.__version__)
print("matplotlib version :", matplotlib.__version__)
print("sns version :",sns.__version__)


# ## 概要
# 限られた時間の中で生きている我々が知ることが出来るのは、平均値とその分散（標準偏差）のみです。マーケティングの世界で例えると、一日あたりの来店者数、一日あたりの商品の売り上げ数などです。では、その限られた情報から何を知ることが出来るでしょうか？今日の来店者数が１０００人だった場合、明日の来店者数が５００の確率はどのぐらいでしょうか？また、１５００人である確率はどのぐらいでしょうか？その答えを教えてくれるのが確率分布になります。
# 
# 「確率思考の戦略論」では、確率分布の形状を決める平均値と標準偏差に相当する数字を $M$と$K$という二つのパラメタで表現できるとしています。$M$が消費者のプリファレンス（相対的好意度）そのものであり、$K$は$M$の関数です。本書では一貫して以下の事を主張しています。
# 
# プリファレンス（相対的好意度）がブランドのマーケット・シェア、浸透率、購入回数を支配している
# 理由は以下の3点です（教科書の写し書きです）。
# 
# 1. プリファレンスは消費者の頭の中にあり、人の購買行動を支配している。直接的な証拠は、消費者のプリファレンスに基づく「BP-10シェアモデル」が現実のシェアを比較的高い精度で予測できる事である。<font color="MediumVioletRed">消費者のプリファレンスがシェアを支配し、売り上げを支配している。</font>言い換えれば、100%の認知、100%の配下率、十分な時間があればプリファレンスとユニット・シェアは同じものになる。プレファレンスは消費者の頭の中にあり、それが現実に現れたのがユニット・シェアである。
# 2. 負の二項分布モデルにより、カテゴリーおよびブランドの浸透率と回数別分布は、MとKの二つのパラメタのみで現実に非常に近い正確な予測が出来る。$M$も$K$もプリファレンスの関数である。
# 3. カテゴリーのMとK、ユニット・シェア、デリシュレーSをインプットとして、デリシュレーNBDモデルは各ブランドの現実に非常に近い浸透率と回数別購入率の分布を正確に予測する事が出来る。ブランド間のスイッチングも正確に予測できる。デリシュレー$S$も$K$もプリファレンスの関数である。
# 
# 本サイトでは、「確率思考の戦略論」にそって
# 
# 1. 二項分布 
# 2. ポアソン分布 
# 3. 負の二項分布 
# 4. ポアソン分布と負の二項分布のまとめ 
# 5. 売り上げを支配する重要な式 
# 6. デリシュレーNBDモデル
# 
# の順に沿って解説を行います。

# ## 1-1. 二項分布 (Binomicl Distribution)
# 
# ### 1. 二項分布の式
# 二項分布は成功確率$\displaystyle p$の試行を $\displaystyle N$回行い、その成功回数$r$を確率変数とする確率分布の事です。一般的には以下の様な確率質量関数として定義されています。正規分布などの説明に用いられる確率密度関数ではなく、確率質量関数となっているのは$r$が正の整数しか取らない離散値だからです。
# 
# $$ \frac{N!}{r! \(N-r\)!} \times p^r \times \left(1-p\right)^{N-r}  \cdot \cdot \cdot \cdot  \left(1\right) $$
# 
# 本章では、二項分布をくじ引きを例に説明しています。くじが全部で$n$個あり、その中の$\theta$個が当たりとします。そうすると、一回目のくじ引きであたりのくじを引く確率は$\displaystyle \frac{\theta}{n}$となります。はずれのくじを引く確率は$\displaystyle 1-\frac{\theta}{n}$ となります。あたりのくじを引く回数が$r$回、はずれのくじを引く回数は$\displaystyle N-r$回なので、例えば、最初の$r$回連続であたりのくじを引き、その後、$1-r$回連続ではずれのくじを引く確率は、
# 
# $$ \left(\frac{\theta}{n}\right)^r \times \left(\frac{n-\theta}{n}\right)^{N-r}  \cdot \cdot \cdot \cdot  \left(2\right) $$
# 
# となります。後は、その組み合わせを考える必要があります。その組み合わせは高校数学で習ったとおり、$N$中、$r$回のあたりを引く確率なので、$\displaystyle {}_n \mathrm{C}_r =  \frac{N!}{r! \(N-r\)!} $となり、
# 
# $$ \frac{N!}{r! \(N-r\)!} \times \left(\frac{\theta}{n}\right)^r \times \left(\frac{n-\theta}{n}\right)^{N-r}  \cdot \cdot \cdot \cdot  \left(3\right) $$
# となります。

# ### 2. pythonによる計算例

# In[4]:


x = np.arange(100)

n = 100
p = 0.3
mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
print("平均    : ", mean)
print("標準偏差 :", var)

plt.xlabel('$r$')
plt.xlabel('$r$')
plt.ylabel('$B(n,p)$')
plt.title('binomial distribution n={}, p={}'.format(n,p))
plt.grid(True)

y = binom.pmf(x,n,p)
plt.scatter(x,y)
# sns.plot(x=x, y=y)  
# sns.scatterplot(data=tips, x='total_bill', y='tip')

plt.show()


# ## 1-2. ポアソン分布 (Poisson Distribution)
# 
# ### 1. ポアソン分布の意味
# ポアソン分布は単位期間あたり$\mu$回起こる<font color="MediumVioletRed">ランダムな事象</font>の発生回数が従う分布になります。ポアソン分布が持つパラメタこの$\mu$一つだけです。数式で表すと、
# 
# $$P\left(r|\mu \right) = \frac{\mu^r}{r!}e^{-\mu}$$
# 
# となります。本書に習って$\mu = 0.6$の場合に$r=0,1,2,3,4$の各場合について計算した結果が以下の通りです。
# 
# <style>.cent td {text-align:center;}</style>
# <style>.cent tr {text-align:center;}</style>
# 
# <div style="width:100%;"> 
# <table class="cent">
#   <tr>
#     <th>$r$</th>
#     <th>0</th>
#     <th>1</th>
#     <th>2</th>
#     <th>3</th>
#     <th>4</th>
#   </tr>
#   <tr>
#     <td>確率</td>
#     <td>54.88%</td>
#     <td>32.92%</td>
#     <td>9.88%</td>
#     <td>1.98%</td>
#     <td>0.30%</td>
#   </tr>
# </table>
# </div>
# 
# グラフと計算に利用したpythonのコードを以下に示します。

# In[5]:


from scipy.stats import poisson

x = np.arange(10)

mu = 0.6
mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
print("平均    : ", mean)
print("標準偏差 :", var)

y = poisson.pmf(x,mu)

plt.xlabel('$r$')
plt.ylabel('$P(r|\mu)$')
plt.title('poisson distribution mu=%.1f' % (mu))
plt.grid(True)

plt.plot(x,y)

plt.show()


# ## 1-3. 負の二項分布 (Negative Binomical Distribution)
# 
# この章では、最初に重要な結論が書かれています。
# 
# ```text
# 個人個人の購買行動はポアソン分布をしているが、消費者全体を見ると「負の二項分布」をしている
# ```
# 
# なぜ、消費者全体を見ると負の二項分布をしているかが書かれておらず、以後、負の二項分布を仮定して議論が進みます。しかし、先の章の話になってしまいますが、1-4.の「ポアソン分布」と「負の二項分布」のまとめのP254の下の方に、
# 
# - あ）個人レベルでポアソン分布している
# - い）長期平均値 $\mu $は消費者全体で見たときガンマ分布している
# 
# と書かれており、さらに、
# 
# - この二つの前提が成り立つとき、消費者全体で見たときにある期間の実際の購入確率は負の二項分布すると覚えてください
# 
# と書かれています。つまり、消費者全体で見たときに負の二項分布をすると言うのは、（あ）と（い）の結果でしかないのです。
# 
# 
# その結果として、消費者全体によってあるカテゴリー、もしくは、あるブランドが選ばれる確率は
# 
# $$ P\left(r \right) = \frac{\left(1 + \frac{M}{K} \right)^{-K} \cdot \Gamma\left(K + r \right)}{\Gamma\left(r + 1 \right)\cdot \Gamma\left(K \right)} \cdot \left(\frac{M}{M+K} \right)^r \cdots \left(1 \right)$$
# 
# と計算できると述べています。その後に、その負の二項分布が、「成功が成功を呼ぶ分布」としてのガンマ分布を仮定することによって$\left(1 \right)$が導かれる事が証明されていますが、この理解は後で良いかと思います。繰り返しますが、重要なのは、
# 
# ```text
# ポアソン分布とガンマ分布を仮定することによって、負の二項分布が導かれる
# ```
# 
# という事です。
# 
# ※実際に個人レベルのポアソン分布と消費者全体でのガンマ分布を仮定することによって負の二項分布が導かれることは1.6で証明します。

# ### アプリでの動作確認
# 
# 負の二項分布の挙動の確認は以下のアプリで確認することが出来ます。
# <div class="container">
# <a href="https://app.wayama.io/article/mkt/nbd" class="btn-border">負の二項分布</a>
# </div>

# ## 1-4. ポアソン分布と二項分布のまとめ
# 
# 1.4 では「ポアソン分布と二項分布のまとめ」と主題でまとめていますが、重要な点は繰り返しになりますが以下の通りです。
# 
# - 消費者があるカテゴリーを選ぶ場合もあるブランドを選ぶ場合もその仕組みは同じである。すなわち、消費者がどのカテゴリーを選ぶかという問題と、どのブランドを選ぶかという問題は同じ確率分布を用いて解く事が出来る。
# - 消費者個人の購買の分布は「ポアソン分布」している。
# - 上記2点の結果として、ある期間における消費者全体の購入回数の分布は「負の二項分布」に従う 
# 
# ※ここで述べていることは、教科書と原因と結果の因果関係が異なります。教科書では、ポアソン分布と負の二項分布の結果からガンマ分布を導かれると書かれていますが、同じページの下の部分では、ポアソン分布とガンマ分布から負の二項分布が導かれると書かれているので、ここではそちらの理解の立場を取ります。

# ### ガンマ関数の表記
# 一般的にガンマ分布を表す数式は、形状を決定するパラメタ $\alpha, \beta$を用いて、
# 
# $$ f\left(x|\alpha, \beta \right) =\frac{\beta^{\alpha}x^{\alpha - 1}e^{-\beta x}}{\Gamma\left(\alpha \right)} \cdot\cdot\cdot\cdot\left(1\right) $$
# 
# と表現されます。また、$\displaystyle \beta =\frac{1}{\theta} $として、
# 
# $$ f\left(x|\alpha, \theta \right) =\frac{x^{\alpha - 1}e^{-\frac{x}{\theta}}}{\Gamma\left(\alpha \right)\theta^{\alpha}} \cdot\cdot\cdot\cdot\left(2\right) $$
# 
# <a href="https://en.wikipedia.org/wiki/Gamma_distribution" target="_blank">wikipedia</a>でもこの二通りの数式で表記されています。ここで、(1)と(2)における確率分布の平均と標準偏差は以下の通りです。
# 
# <style>.cent td {text-align:center;}</style>
# <style>.cent tr {text-align:center;}</style>
# 
# <div style="width:100%;"> 
# <table>
#   <tr>
#     <th>ガンマ分布</th>
#     <th>$\displaystyle E[x]$</th>
#     <th>$\displaystyle V[x]$</th>
#   </tr>
#   <tr>
#     <td>$ \displaystyle \frac{\beta^{\alpha}x^{\alpha - 1}e^{-\beta x}}{\Gamma\left(\alpha \right)} $</td>
#     <td>$\displaystyle \frac{\alpha}{\beta}$</td>
#     <td>$\displaystyle \frac{\alpha}{\beta^2}$</td>
#   </tr>
#   <tr>
#     <td>$\displaystyle \frac{x^{\alpha - 1}e^{-\frac{x}{\theta}}}{\Gamma\left(\alpha \right)\theta^{\alpha}} $</td>
#     <td>$\displaystyle \alpha\theta$</td>
#     <td>$\displaystyle \alpha\theta^2$</td>
#   </tr>
# </table>
# </div>
# 
# 本書では、$\displaystyle Gamma \left(K,\frac{M}{K} \right)$がどちらの表記を用いているか明示されていませんが、$\displaystyle Gamma \left(1,5 \right)$、$\displaystyle Gamma \left(3,\frac{5}{3} \right)$、$\displaystyle Gamma \left(15,\frac{5}{15} \right)$の平均値はすべて5と述べているので、(2)の表記を用いていると思われます。

# ### python code
# ガンマ分布を記述するpythonのコードです。moduleとしてscipy、numpy、matplotlibなど機械学習ではおなじみのライブラリを用いてます。
# 
# ### ガンマ分布のpython code
# 上記のコードの実行結果です。
# 
# - $\displaystyle \left(K,\frac{M}{K} \right) = \left(1,5 \right) , \left(3,\frac{5}{3} \right), \left(15,\frac{5}{15} \right) $の三通りについてプロットしています。教科書のP61の図2-2と同じようなグラフが得られています
# 

# In[6]:


from scipy.stats import gamma

x = np.linspace(0,50,1000)

a = 1.0 
b = 5.0
mean, var, skew, kurt = gamma.stats(a, scale=b, moments='mvsk')
y1 = gamma.pdf(x, a, scale=b)
print('a : {}, b : {:,.3f}, mean : {}'.format(a,b,mean))

a = 3.0
b = 5.0/3.0
mean, var, skew, kurt = gamma.stats(a, scale=b, moments='mvsk')
print('a : {}, b : {:,.3f}, mean : {}'.format(a,b,mean))
y2 = gamma.pdf(x, a, scale=b)

a = 15.0
b = 1.0/3.0
mean, var, skew, kurt = gamma.stats(a, scale=b, moments='mvsk')
print('a : {}, b : {:,.3f}, mean : {}'.format(a,b,mean))
y3 = gamma.pdf(x, a, scale=b)

plt.grid()
plt.ylim([-0.01,0.40])
plt.xlim([0,15])

plt.plot(x, y1, x, y2, x, y3)

plt.show()


# ### アプリでの動作確認
# 
# ガンマ分布の挙動の確認は以下のアプリで確認することが出来ます。
# <div class="container">
# <a href="https://app.wayama.io/article/mkt/gamma" class="btn-border">ガンマ分布</a>
# </div>
# 

# ## 1-5. 売り上げを支配する重要な式
# 
# この章で重要なのは、表９−５であり、消費者全体がどの「カテゴリー」、もしくは、「ブランド」を選ぶかは負の二項分布に従い、それぞれの平均購入回数$M$や、分布を決定するパラメタ$K$がどのように決定されるか示されています。「カテゴリー」を表現した式も、「ブランド」を表現した式もほぼ同じような数式で表現されています。違うのは添字ぐらいでしょうか。
# 
# 本書を理解する上で重要だと思われる点はパラメタ$K$の説明がされているP258の中段の説明になります。
# 
# - パラメタ$\left(k_j \right) $は、最初にあった赤玉の$\displaystyle \left(\theta_j \right) $を1回ずつ袋に足す玉の数$\displaystyle\left(d_i \right) $割った値で$\theta_j$ の関数です。分散の式が示すように、赤玉が多くなる（プリファレンスが増える）事により分散が大きくなり、浸透率$\theta$に対しての微分が示すように、より多くの人に広がります。市場の構造上のプリファレンスが高くなれば、より人々に広がります。
# 
# 　すなわち、$K$というのは、プリファレンスの関数であり、プリファレンスが増加するごとに、増加するする値である事が説明されています。
# 
# 　負の二項分布は、
# 
# $$
# P\left(r \right) = \frac{\left(1 + \frac{M}{K} \right)^{-K} \cdot \Gamma\left(K + r \right)}{\Gamma\left(r + 1 \right)\cdot \Gamma\left(K \right)} \cdot \left(\frac{M}{M+K} \right)^r \cdots \left(1 \right)
# $$
# 
# と表され、$M$はプリファレンスであり、$K$がプリファレンスの関数であれば、$P\left(r \right)$は$M$のみをパラメタに持つ関数という事になります。これは本書で筆者たちがいい関して主張している結論になります。

# ## 1-6. デリシュレーNBDモデル
# 
# ### デリシュレーNBDモデルとは
# デリシュレーNBDモデルとは、あるカテゴリーの中のブランド間の関係を教えてくれる確率分布になります。この分布を用いてわかる具体例として本書で上げられているのが、P31の表1-4になります。与えら得た式(後述の式１)に対して、どのようにパラメタを計算しているのかが具体的に示されています。
# 
# 本章の内容はかなり高度なレベルとなっています。少しずつ読み解いてみます。
# 
# ### 勝手な注釈
# 私ごときが注釈をつけるのは失礼に当たるのですが、私と同じレベルの読者が少々戸惑う場面もあると思いますので、少し追記しておきます。
# 
# #### デリシュレーについて
# 著者は「Dirichlet」をデリシュレーと表記していますが、おそらく一般的にはディリクレと表記される場合が多いかと思います。統計の分野でもディリクレ分布、数値計算の分野でも境界値問題でディリクレ問題などと言いますので、混同しないように気をつけた方が良いかと思います。私も普段はディリクレと言いますが、以下ではデリシュレーに統一します。
# 
# ### 概要
# 
# まず結論からです。デリシュレーNBDは以下の様な数式で表せると結論づけています。
# 
# <div>
# $$
# \begin{aligned}
# P_{r} \left(R,r_1, r_2, \cdots , r_g \right) &=\frac{\Gamma\left(S \right)}{\displaystyle \prod_{i=1}^{N} \Gamma\left(\alpha_j \right)} \cdot \frac{\displaystyle \prod_{i=1}^{N} \Gamma\left(r_j + \alpha_j \right)}{\Gamma\left(S+R \right)} \cdot \frac{1}{\displaystyle \prod_{j=1}^{g} r_j!} \cdot \\ 
# &\qquad \frac{\Gamma\left(R + K \right)}{\displaystyle \Gamma\left(K \right)} \cdot \left(1 + \frac{K}{MT} \right)^{-R} \cdot\left(1 + \frac{MT}{K} \right)^{-K} \cdot\cdot\cdot\cdot\left(1\right) \\ 
# \\
# R&=\sum_{j=i}^{g}r_j  \cdot\cdot\cdot\cdot\left(2\right) 
# \\
# \\
# \alpha_j &=S \times\left(ブランドjの購入頻度に基づくマーケットシェア \right)\cdot\cdot\cdot\cdot\left(3\right)  \\
# \\
# S &=\sum_{j=1}^g \alpha_j \cdot\cdot\cdot\cdot\left(4\right) 
# \end{aligned}
# $$
# </div>
# 
# そして、数式が成り立つ前提は以下の様になっています。
# 
# 1. 消費者各自の購買行動は独立事象
# 2. 購入行動はランダムに発生
# 3. 購入者各自$\left(C_i \right)$は、一定のカテゴリーに対して、長期的購入回数の平均値$\mu_i$を持つ。購入者各自$\left(C_i \right)$の単位時間のカテゴリ購入回数$R_i$はポアソン分布している。
# $$R_i \sim Poisson\left(\mu_i \right)$$
# 4. カテゴリーの長期平均購入回数$\left(\mu \right)$は消費者間で異なり、ガンマ分布している。
# $$\mu \sim Gamma\left(K,\frac{M}{K} \right)$$
# 5. 期間$T$における各ブランドの購入回数$\left(r_j \right)$は、ガンマ分布$Gamma\left(\alpha_j,\beta \right)$に従う。$\alpha$はブランド間で異なるが、$\beta$はブランド間で同一。この購入回数に関する過程は結果としてブランドを選ぶ確率$p$がデリシュレー分布する。
# 本来、(1)から(4)のカテゴリーに関する仮定をブランドに関する仮定に当てはめると、各ブランドの購入回数は負の二項分布になる。よって、この仮定はガンマ分布でNBDを近似している事に相当。
# ※個人的には、このカテゴリーの理論をそのまま当てはめれば、ブランドもNBDになるが、ガンマ分布で仮定するという事は重要なポイントだと思っています。
# 6. 各消費者は、各ブランドに対して一定の購買確率を持っており、ブランドの購入$\left(r \right)$ は多項分布に従う。各々のカテゴリーの購入時のブランドの購入確率$\left(p \right)$は、それぞれのブランドについて長期的に見ると決まっている。ただ、カテゴリー購入時にどれを選ぶかはランダムである。
# 7. 異なる人々の各々のカテゴリー平均購入回数と、人々のそれぞれのブランドを選択する確率とは、互いに独立。すなわち、特定のカテゴリー購入回数の人が、特定のブランドを特定の確率で購入しているような事が起こらない。
# 
# 
# ### 式(17)の意味
# 
# 仮定1〜7をまとめたものが式(17)になっています。以下では教科書の式(17)を式(17')として、少し変えて表記しています。
# 
# <div>
# $$
# \begin{aligned}
# &P(R,r_1,\cdots,r_g) = \\ \int &Multi(r|p,R) Dir(p|\alpha)dp \int Poisson(R|\mu T) Gamma\left(\mu|K,\frac{M}{K}\right) d\mu \quad \cdots 式(17')
# \end{aligned}
# $$
# </div>
# 
# 式(17')は二つの積分から構成されています。
# 
# <div>
# $$
# \int Poisson(R|\mu T)\cdot Gamma\left(\mu|K,\frac{M}{K}\right) d\mu \quad \cdots (パート1)
# $$
# </div>
# 
# と
# 
# <div>
# $$
# \int Multi(r|p,R)\cdot Dir(p|\alpha)dp \quad \cdots （パート2）
# $$
# </div>
# 
# です。以下それぞれの式の意味について説明します。
# 
# #### パート1について
# これは先ほど説明したとおり、カテゴリーを選択する時の確率分布、NBDになります。
# 
# - 個人レベルでの購入回数は平均購入回数$\mu$をパラメタに持つポアソン分布に従う
# - ポアソン分布のパラメタ$\mu$は消費者全体で見た時$\displaystyle \left(K, \frac{M}{K}\right)$をパラメタに持つガンマ分布に従う
# 
# ポアソン分布とガンマ分布の積を$\mu$に対して積分すると、カテゴリーの購入回数別の消費者全体の確率を得ることが出来ます。
# 
# #### パート2について
# 
# 各ブランドが選ばれる確率を求める分布になります。
# 
# - 各ブランドの購入回数$(r_j)$はそのブランドが選ばれる確率$(p_j)$をパラメタに持つ多項分布に従う
# - ブランドが選ばれる確率$(p_j)$はパラメタ$(\alpha)$を持つデリシュレー分布に従う
# 
# パート1とパート2の詳細な計算は別途以下で行います。
# 
# また、式(17)が生成されるモデルを私が勝手に解釈した図を以下に置いておきます。
# 
# ![png](1_nb_files_local/nbd_1.png)

# ### パート1
# 
# #### (ア)ガンマ分布と$S$の正体：
# 
# #### ガンマ分布の定性的理解
# 
# 教科書によると、ガンマ分布は確率の発生が、さらにその確率を高めていく分布という事です。正直私の今の知識ではさっとは理解できませんが、おそらく、負の二項分布を導出したときの、赤玉と白玉の結果から理解することが出来ます。赤玉と白玉が入った袋から、赤が出たら、その玉を袋に戻しつつ、さらに赤玉を追加するという処理を数式化すると負の二項分布が導かれます。つまり、赤を引けば、次に赤を引く確率は増えていることになります。
# 
# 次で示しますが、ポアソン分布とガンマ分布の混合分布からも負の二項分布が導かれます。これより、結果として、ガンマ分布が確率を高めていく分布である事が理解できます。(あくまでも定性的理解で、間違っているかもしれません。)
# 
# #### ガンマ分布の基本的な性質
# 
# ガンマ分布の表記と平均、分散が示されています。
# 
# <div>
# $$
# Gamma\left(r | \alpha, \beta \right) = \frac{r^{\alpha - 1}e^{-\frac{r}{\beta}}}{\Gamma\left(\alpha \right)\beta^\alpha } 
# $$
#     
# $$
# E[r] = \alpha \beta
# $$
# 
# $$
# Var[r] = \alpha \beta^2
# $$
# </div>
# 
# #### ガンマ分布の加法性（再生性）
# 
# ガンマ分布には加法性（本書では加法性と書かれているが、一般的には再生性と言われることが多い）という性質があります。$r_1$と$r_2$が
# $$
# r_i \sim Gmma(\alpha_i, \beta)
# $$
# $$
# r_j \sim Gmma(\alpha_j, \beta)
# $$
# というガンマ分布に従って生じるとき、$r_i$と$r_j$は、
# 
# $$
# r_i + r_j \sim Gmma(\alpha_i + \alpha_j, \beta)
# $$
# に従って生じるという特性です。
# 
# 仮定（5）に従い、ブランドの購入回数$r_i$はガンマ分布に従います。よって、カテゴリーの購入回数$R$は$$R=\sum_{j=0}^gr_j$$なので、
# $$ R \sim Gamma\left(\sum_{j=0}^g \alpha_j, \beta\right) $$
# となります。
# 
# また、
# $$S= \sum_{j=0}^g\alpha_j$$
# という新しいパラメタ$S$を設定すると、$R$の期待値はガンマ分布の特性から$S\beta$になります。一方、$R$はカテゴリーの購入回数であり、負の二項分布に従い、期待値は$MT$となります。
# よって、$$S\beta=MT$$となります。
# 
# さらに、個々のブランドの購入回数の期待値についても同様に考えることが出来て、
#  $$\alpha_j\beta=m_jT$$となり、二つの式から$\beta$を削除すると$$\alpha_j=S\times \frac{m_j}{M}$$となります。これがガンマ分布のパラメタである$\alpha$の意味になります。
#  
# #### Sの意味
# ガンマ分布の特性から$$R\sim Gamma(S,\beta)$$となります。また、$$R\sim NBD(K,MT)$$となります。以上から、私は、$S$を以下ように理解しています。
# 
# <div class="box1">
# 仮定（5）で述べられている、「カテゴリーの購入回数はNBDに従うが、ブランドの購入回数はガンマ分布に従う」と仮定した場合にその二つをつなぐパラメタ
# </div>
# 

# #### (イ) ポアソン分布とガンマ分布から負の二項分布へ：
# ポアソン分布とガンマ分布の混合分布から負の二項分布を導きます。個人のカテゴリーの購入回数はポアソン分布に従い、ポアソン分布の平均購入回数$\mu$はガンマ分布に従うという仮定をおきます。
# 
# <div>
# $$
# Poisson(R|\mu T) = \frac{\left(\mu T \right)^R\cdot e^{-\mu T}}{R!}
# $$
# $$
# Gamma\left(\mu|K, \frac{M}{K}\right) = \frac{\mu^{K-1}}{\Gamma(K)\cdot\left(\frac{M}{K}\right)^K}e^{-\mu\frac{K}{M}}
# $$
# </div>
# 
# この表記から、$\mu$に対して積分を行います。購入回数の期待値を計算します。
# 
# <div>
# $$
# \begin{aligned}
# P\left(R \right) &=\int_0^{\infty} Poisson\left(R|\mu T \right) Gamma\left(\mu | K, \frac{M}{K} \right)  d\mu \\
# &= \int_0^{\infty} \frac{\left(\mu T \right)^R\cdot e^{-\mu T}}{R!}\frac{\mu^{K-1}}{\Gamma(K)\cdot\left(\frac{M}{K}\right)^K}e^{-\mu\frac{K}{M}} d \mu \\
# &= \frac{K^KT^R}{R!\Gamma\left(K \right)M^K}  \int_0^{\infty} \mu^{R+K-1}e^{-\mu\left(T +\frac{K}{M}  \right)} d\mu 
# \end{aligned}
# $$
# </div>
# 
# ここで、ガンマ関数が、
# $$
# \Gamma\left(r \right) = \int_0^{\infty}t^{r-1}e^{-t}dt
# $$
# 
# と定義できる事を念頭に、
# $$
# \mu \left(T +\frac{K}{M}  \right)  = t \rightarrow \mu =  \frac{M}{MT + K } t
# $$
# 
# と変数変換すると、
# 
# <div>
# $$
# \begin{aligned}
# P\left(R \right) &= \frac{K^KT^R}{R!\Gamma\left(K \right)M^K}  \int_0^{\infty} \mu^{R+K-1}e^{-\mu\left(T +\frac{K}{M}  \right)} d\mu \\
# &= \frac{K^KT^R}{R!\Gamma\left(K \right)M^K} \int_0^{\infty} \left(\frac{M}{MT+K}\right)^{R+K-1} t^{R+K-1}e^{-t  } \left(\frac{M}{MT+K}\right) dt \\
# &= \frac{K^KT^R}{R!\Gamma\left(K \right)M^K}\left(\frac{M}{MT+K}\right)^{R+K}  \int_0^{\infty} t^{R+K-1}e^{-t} dt \\
# &= \frac{K^KT^R}{R!\Gamma\left(K \right)M^K}\left(\frac{M}{MT+K}\right)^{R+K}\Gamma(R+K) \\
# &= \frac{\left(1 + \frac{MT}{K} \right)^{-K} \cdot \Gamma\left(K + R \right)}{R!\cdot \Gamma\left(K \right)} \cdot \left(\frac{MT}{MT+K} \right)^R 
# \end{aligned}
# $$
# </div>
# 
# 以上から、
# 
# $$ P\left(R \right) = \left(1 + \frac{MT}{K} \right)^{-K}\frac{\Gamma\left(K + R \right)}{R! \cdot \Gamma\left(K \right)} \cdot \left(\frac{MT}{MT+K} \right)^R $$
# 
# となり、式(21)となります。また、1-3の負の二項分布の章で示した以下の式と一致します。以上から、
# 
# <div class="box1">
# 消費者個人の購買活動がポアソン分布しており、その長期平均購入回数がガンマ分布していることを仮定することによって、ある期間における消費者全体の購入回数は負の二項分布している
# </div>
# 
# という事が導けました。
# 

# ### パート2
# #### (ウ) ガンマ分布からデリシュレー分布へ：
# 
# 各ブランドの購入回数$r_1,\cdots,r_g$は独立であり、それぞれガンマ分布に従うという仮定から、各ブランドの購入確率の分布式を導きます。ブランドの購入回数の関数は仮定から以下の様になります。
# 
# <div>
# $$
# G(r_1,r_2,\cdots ,r_g | \alpha_1,\alpha_2,\cdots,\alpha_g) = \prod_{j=1}^{g} \frac{r_j^{\alpha_{j-1}}e^{-\frac{r_j}{\beta}}}{\Gamma(\alpha_j)\beta^{\alpha_j}}
# $$
# </div>
# 
# この式の$r_j$をブランド$j$が選ばれる確率$p_j$に変数変換を行います。変換を行うための条件を以下の通りです。教科書ではDからFに射影変換すると説明されています。
# 
# <div>
# $$
# D=\{(r_1,r_2,\dots, r_j \cdots,  r_g): 0 \leqq r_j < \infty , j = 1,2,\cdots j \cdots g \}
# $$
# $$
# F=\{(p_1,p_2,\dots,p_j, \cdots,  p_{g-1}): 0 \leqq p_j < 1 , j = 1,2,\cdots j \cdots g-1 \}
# $$
# $$
# p_j=\frac{r_j}{r_1+r_2+ \cdots +r_g} \quad (j=1,2,\cdots,g-1)
# $$
# </div>
# 
# 教科書では$0 < p_g < \infty$となっていますが、おそらく誤植だと思います。結果として、式(22)が導かれます。
# 
# $$ 
# Dirichlet\left(p|\alpha \right) = \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}\alpha_j\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j\right)} \left( \prod_{j=1}^{g-1}p_j^{\alpha_j-1}\right)\left(1-\sum_{j=1}^{g-1}p_j \right)^{\alpha_g-1}
# $$
# 
# 
# この確率分布は一般的にディリクレ分布と言われることが多いと思います。この式は拘束条件を別の式として、
# 
# <div>
# $$ 
# Dir\left(p_1,p_2, \cdots, p_g| \alpha_1,\alpha_2, \cdots, \alpha_g\right) = \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}\alpha_j\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j\right)} \prod_{j=1}^{g}p_j^{\alpha_j-1}
# $$
#     
# ただし、
# 
# $$
# \sum_{j=1}^{g}p_j = 1
# $$
# 
# と書く方が一般的な気がします。
# 
# 
# 少しややこしいですが、注意深く式を追えば導けると思います。以下に式（22）を導く際の注意点を挙げます。
# 
# #### 注意点1：変数変換の際の拘束条件
# 
# $r_1, \cdots, r_j$は独立試行によりガンマ分布からサンプリングされますが、これを確率$p_1, \cdots, p_g$に変数変換する際、$p_j$には$\sum_j p_j=1$という拘束条件があります。式(22)の導出途中で$1-\sum_{j=1}^{g-1}$となっているのは、この拘束条件を式の中に反映させるためです。
# 
# #### 注意点2：ヤコビアンの計算
# 
# P267の一番下の式に
# 
# <div>
# $$
# \begin{aligned}
# H_{g} \left(p_1, p_2, \cdots , p_{g'} \right) &= \left[\prod_{j=1}^{g-1}\frac{(p_jp_{g'})^{\alpha_j-1}e^{-\frac{p_jp_{g'}}{\beta}}}{\Gamma(\alpha_j)\beta^{\alpha_j}} \right] \cdot \left[ \frac{((1-\sum_{j=1}^{g-1}p_j)p_{g'})^{\alpha_j-1}e^{-\frac{\left(1-\sum_{j=1}^{g-1}p_j\right)p_{g'}}{\beta}}}{\Gamma(\alpha_g)\beta^{\alpha_g}} \right]\cdot |J|
# \end{aligned}
# $$
# </div>
# 
# という$|J|$という式が出てきます。これをヤコビアンと言います。確率分布の変数変換は、単純に変数を入れ替える以外にこの量を考慮する必要があります。変数が一つしかない場合は簡単ですが、変換する変数の量が複数ある場合は、以下の様な計算をする必要があります。
# 
# <div>
# $$
# |J| = \left|
#   \begin{array}{cccc}
#     \displaystyle \frac{\partial r_1}{\partial p_1} & \displaystyle \frac{\partial r_1}{\partial p_2} & \cdots & \displaystyle \frac{\partial r_1}{\partial p_{g'}} \\
#     \displaystyle \frac{\partial r_2}{\partial p_1} & \displaystyle \frac{\partial r_2}{\partial p_2} & \cdots & \displaystyle \frac{\partial r_2}{\partial p_{g'}} \\
#     \vdots & \vdots    & \ddots & \vdots \\
#   \displaystyle \frac{\partial r_g}{\partial p_1} & \displaystyle \frac{\partial r_g}{\partial p_2} & \cdots & \displaystyle \frac{\partial r_g}{\partial p_{g'}} 
#   \end{array}
# \right|
# $$
# </div>
# 
# ヤコビアンの意味を簡単に説明します。例えば、$p(x,y)$を変数$x,y$を確率変数とする連続確率分布を表すとします。普段はあまり意識することはないですが、連続確率分布はその値だけでは実は意味をなしません。例えば、$\displaystyle p(2,3)=\frac{1}{3}$という結果は、暗黙的に$\displaystyle p(2,3)dxdy=\frac{1}{3}$という事を意味していて、$x,y$が$x=2 + dx, y=3+dy$の範囲に存在する確率を意味します。よって、この$dxdy$というが確率分布の解釈には必要で、ヤコビアンは$dxdy$を変数変換する時に必要になります。正確な数学的な表現はルベーグ積分や測度論の知識が必要となりますが、マーケティングの世界では不要かと思います。ただ、このように数式をしっかり追うときに知識として必要になります。
# 
# これは私の経験から来る感想ですが、実務レベルになるとこの辺の理解の重要性を実感します。コンピュータに積分を計算させようとするとき、この$dxdy$を考慮しないと訳のわからない数値になり、モデルが意味をなさず、何をしているのか分からなくなります。
# 
# #### ヤコビアンの計算の例
# 
# よく利用されるガウス積分を例にヤコビアンの計算をしてみます。$x=r\cos \theta, y=r\sin \theta$という極座標への変数変換を例に取ります。
# 
# 
# <div>
# $$
# dxdy = |J|drd\theta
# $$
# </div>
# 
# なので、
# 
# <div>
# $$
# \mathrm{det}J = |J| = \left|
#   \begin{array}{cc}
#     \displaystyle \frac{\partial x}{\partial r} & \displaystyle \frac{\partial x}{\partial \theta} \\
#     \displaystyle \frac{\partial y}{\partial r} & \displaystyle \frac{\partial y}{\partial \theta}
#   \end{array}
# \right| = \left| 
#   \begin{array}{cc}
#     \cos \theta & -r\sin \theta\\
#     \sin \theta & r\cos \theta\\
#   \end{array}
# \right| = r
# $$
# </div>
# 
# 
# <div>
# $$
# p(x,y)dxdy = p(r,\theta)rdrd\theta
# $$
# </div>
# 
# 
# <div>
# $$
# \int_{-\infty}^{\infty}dx\int_{-\infty}^{\infty}dye^{-(x^2+y^2)} = \int_0^{\infty}dr\int_0^{2\pi}d\theta re^{-r^2} = \pi
# $$
# </div>
# 
# これより、
# 
# <div>
# $$
# \int_{-\infty}^{\infty}e^{-x^2}dx = \sqrt{\pi} 
# $$
# </div>
# 
# となり、ガウス積分の有名な公式が導けます。この辺の説明がないので、数学や確率・統計の知識があまりないと完全に理解するのは厳しいかもしれません。
# 
# #### 行列式の計算
# 
# ヤコビアンは$\displaystyle \frac{\partial r_1}{\partial p_1} $を計算する必要がありますが、$\displaystyle p_j=\frac{r_j}{r_1+r_2 + \cdots + r_g}$から計算することが可能です。簡単な計算により、以下の様になります。
# 
# <div>
# $$
# |J| = \left|
#   \begin{array}{cccc}
#     p_{g'} & 0 & \cdots & p_1 \\
#     0 & p_{g'} & \cdots & p_2 \\
#     \vdots & \vdots & \ddots & \vdots \\
#     -p_{g'} & -p_{g'} & \cdots & \displaystyle 1- \sum_{j=1}^{g-1}\
#   \end{array}
# \right|
# $$
# </div>
# 
# 
# #### 行列式の計算の性質
# 
# 
# 行列式の特徴として、ある行にある行を足しても変わらないという特性があります。
# 
# ２次の正方行列の場合の具体的な計算をしてみます。
# 
# <br>
# 
# 式で書くと以下の通りです。
# 
# 
# <div>
# $$
# \left|
#   \begin{array}{cc}
#     a & b \\
#     c & d
#   \end{array}
# \right| = ad-bc
# $$
# </div>
# 
# 
# <div>
# $$
# \left|
#   \begin{array}{cc}
#     a & b \\
#     c + a & d + b
#   \end{array}
# \right| = a(d + b) - b(c + a) = ad - bc +ab -ab = ad -bc
# $$
# </div>
# 
# となり、
# 
# <div>
# $$
# \left|
#   \begin{array}{cc}
#     a & b \\
#     c & d
#   \end{array}
# \right| = \left|
#   \begin{array}{cc}
#     a & b \\
#     c + a & d + b
#   \end{array}
# \right|
# $$
# </div>
# 
# となることが分かります。3次以上の場合も同様に計算できます。
# 
# <br>
# 
# P267の下から2行目の「行を足しても変わらない」といっているのは、
# 
# - 1行目をg行目に加える <br>
# - 2行目をg行目に加える <br>
# - $\cdots$
# - g-1行目をg行目に加える <br>
# 
# というように、g行目にそれ以外の行の値をすべて加えることを示しています。1行目にg行目を加えると、1列目の$p_{g'}$と$-p_{g'}$が0となります。この0によりかなり楽に行列式を計算することが出来ます。
# 
# <div>
# $$
# |J| = \left|
#   \begin{array}{cccc}
#     p_{g'} & 0 & \cdots & p_1 \\
#     0 & p_{g'} & \cdots & p_2 \\
#     \vdots & \vdots & \ddots & \vdots \\
#     0 & 0 & \cdots & 1
#   \end{array}
# \right|=(p_{g'})^{g-1}
# $$
# </div>
# 
# となります。この辺の行列の式の変形もある程度知識がないときついかもしれません。
# 
# 
# 

# #### (エ) 多項分布とデリシュレー分布を合体：
# #### 多項分布
# 
# 1から$g$までのブランドがあり、それぞれが選ばれる確率が$p_1,p_2,\cdots,p_g$とします。それぞれのブランドが選ばれる回数を$r_1, r_2, \cdots, r_g$とすると、$r$が従う確率分布は多項分布となります。多項分布は二項分布を多変数に拡張した確率分布になります。
# 
# 
# <div>
# $$
# Multi\left(r_1,r_2, \cdots, r_g | p_1,p_2, \cdots, p_g, R\right) = \frac{R!}{\prod_{j=1}^gr_j!}\prod_{j=1}^{g}p_j^{r_j}
# $$
# </div>
# 
# ただし、
# 
# <div>
# $$
# \sum_{j=1}^g p_j = 1
# $$
# </div>
# 
# となります。これを変形すると$p_g =1-\sum_{j=1}^{g-1} p_j$となり、教科書では$\prod$の最後の$g$の項をこの値に置き換えています。
# 
# このような表式をする場合、変数が$r$で$p,R$はあくまでもパラメタであると理解することが重要です。
# 
# #### 多項分布とデリシュレー分布を合体
# 
# ブランドが選ばれる回数$(r_1,r_2, \cdots, r_j)$はそのブランドが選ばれる確率をパラメタに持つ多項分布に従い、その確率はデリシュレー分布に従うと考えて、二つを掛け合わせ$p$に対して積分する事により$r_1, \cdots ,r_g$となる確率分布を得ることが出来ます。
# 
# 繰り返しになりますが、デリシュレー分布は以下の通りです。
# 
# <div>
# $$ 
# Dir\left(p_1,p_2, \cdots, p_g| \alpha_1,\alpha_2, \cdots, \alpha_g\right) = \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}\alpha_j\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j\right)} \prod_{j=1}^{g}p_j^{\alpha_j-1}
# $$
# </div>
# 
# <div>
# $$
# \sum_{j=1}^g p_j = 1
# $$
# </div>
# 
# この場合は、変数が$p$でパラメタが$\alpha$であると理解することが重要です。
# 
# <!--
# #### ブランドの購入回数が多項分布に従い、購入確率がデリシュレー分布に従うと考える場合
# 
# <div>
# $$
# p_i \sim Dir\left(\alpha_1,\alpha_2, \cdots, \alpha_g\right)
# $$
# </div>
# 
# 正直言うと、このモデリングでいいような気がしますが、どうでしょうか。おそらく、ガンマ分布を仮定してからの流れがマーケティングではよく利用されているのかもしれません。
# 
# 多項分布のパラメタである確率に対して、ディリクレ分布（あえてここではディリクレ分布と表記します）を仮定するのは、ベイズ統計における共益事前分布に慣れてる身からするとかなり自然な感じを受けます。
# 
# <div>
# $$ 
# Multi\left(r_1,r_2, \cdots, r_g| p_1,p_2, \cdots, p_g\right) = \frac{R!}{\displaystyle \prod_{j=1}^{g}r_j!} \prod_{j=1}^{g}p_j^{r_j}
# $$
# </div>
# 
# $\alpha$はハイパーパラメタになります。
# 
# <div>
# $$ 
# Dir\left(p_1,p_2, \cdots, p_g| \alpha_1,\alpha_2, \cdots, \alpha_g\right) = \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}\alpha_j\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j\right)} \prod_{j=1}^{g}p_j^{\alpha_j-1}
# $$
#     
# ただし、
# 
# $$
# \sum_{j=1}^{g}p_j = 1
# $$
# $$
# \sum_{j=1}^{g}r_j = R
# $$
# </div>
# 
# -->
# 
# <div>
# $$
# \begin{aligned}
# &P(R, r_1, \cdots, r_g) \\
# &= \int Multi(r_1, \cdots, r_g | p_1, \cdots, p_g) \cdot Dir(p_1, \cdots, p_g | \alpha_1 , \cdots \alpha_g) dp_1\cdots dp_g \quad \cdots (1)\\
# &=\displaystyle \int \frac{R!}{\displaystyle \prod_{j=1}^{g}r_j!} \prod_{j=1}^{g}p_j^{r_j} \cdot \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}p_j\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(p_j\right)} \prod_{j=1}^{g}p_j^{\alpha_j-1} dp_1\cdots dp_g \quad  \cdots (2)\\
# &=\displaystyle \frac{R!}{\displaystyle \prod_{j=1}^{g}r_j!} \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}\alpha_j\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j\right)} \int \prod_{j=1}^{g}p_j^{r_j + \alpha_j-1} dp_1\cdots dp_g \quad  \cdots (3) \\
# &=\displaystyle \frac{R!}{\displaystyle \prod_{j=1}^{g}r_j!} \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}\alpha_j\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j\right)} \frac{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j + r_j\right)}{\Gamma\left(\displaystyle\sum_{j=1}^{g}(\alpha_j + r_j)\right)} \int \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}(\alpha_j + r_j)\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j + r_j\right)}    \prod_{j=1}^{g}p_j^{r_j + \alpha_j-1} dp_1\cdots dp_g \quad  \cdots (4) \\
# &=\displaystyle \frac{R!}{\displaystyle \prod_{j=1}^{g}r_j!} \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}\alpha_j\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j\right)} \frac{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j + r_j\right)}{\Gamma\left(\displaystyle\sum_{j=1}^{g}(\alpha_j + r_j)\right)} \quad \cdots (5) \\
# &=\displaystyle \frac{R!}{\displaystyle \prod_{j=1}^{g}r_j!} \frac{\Gamma(S)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j\right)} \frac{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j + r_j\right)}{\Gamma\left(S+R\right)} \quad \cdots (6) 
# \end{aligned}
# $$
# </div>
# となり、教科書の式(25)の表記が得られます。
# 
# #### (1)から(2)への変形
# 多項分布とデリシュレー分布を代入しただけです。
# 
# #### (2)から(3)への変形
# $p$に関する部分を積分記号の中に残し、それ以外のものを外に出しています。
# 
# #### (3)から(4)への変形
# デリシュレー分布の積分
# 
# <div>
# $$
# \int \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}(\alpha_j + r_j)\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j + r_j\right)}    \prod_{j=1}^{g}p_j^{r_j + \alpha_j-1} dp_1\cdots dp_g = 1
# $$
# </div>
# 
# の式を適用するために、定数部分
# 
# <div>
# $$
# \frac{\Gamma\left(\displaystyle\sum_{j=1}^{g}(\alpha_j + r_j)\right)}{\displaystyle \prod_{j=1}^{g}\Gamma\left(\alpha_j + r_j\right)}
# $$
# </div>
# 
# を分子と分母に無理矢理作り出しています。
# 
# #### (4)から(5)への変形
# デリシュレー分布の積分を実行して積分部分が消去されます。
# 
# #### (5)から(6)への変形
# $S$と$R$を用いて式を簡素化します。
# 
# <div>
# $$
# S =\sum_{j=1}^g \alpha_j 
# $$
# $$
# R =\sum_{j=1}^g r_j
# $$
# </div>

# ### パート3
# 
# #### デリシュレーNBDモデル：
# 
# パート1の式(21)からカテゴリーの購入回数の分布がわかり、パート2の式(25)で各ブランドの購入回数の分布が定式化出来たので、これらの積を取ることにより、ブランド別の購入分布が導かれます。
# 
# <div>
# $$
# \begin{aligned}
# &P(R, r_1, \cdots, r_g) \\ &=\frac{\Gamma\left(S \right)}{\displaystyle \prod_{i=1}^{N} \Gamma\left(\alpha_j \right)} \cdot \frac{\displaystyle \prod_{i=1}^{N} \Gamma\left(r_j + \alpha_j \right)}{\Gamma\left(S+R \right)} \cdot \frac{1}{\displaystyle \prod_{j=1}^{g} r_j!} \frac{\Gamma\left(R + K \right)}{\displaystyle \Gamma\left(K \right)} \cdot \left(1 + \frac{K}{MT} \right)^{-R} \cdot\left(1 + \frac{MT}{K} \right)^{-K}
# \end{aligned}
# $$
# </div>
# 
# $g=2$の場合は、デリシュレー分布はベータ分布となります。一般的にベータ分布の多変数化がデリシュレー分布（ディリクレ分布）となります。
# この式を用いた具体的な計算は[巻末解説2](/article/mkt/2/#python)を参照してください。
# 

# ## まとめ
# 1-6に関してはポアソン分布や負の二項分布などといった基本的な確率分布の他に、ガンマ分布やベータ分布、連続確率分布の変数変換、ヤコビアンの計算、行列式の性質、多項分布とデリシュレー分布（ディリクレ分布）、ガンマ関数の性質などといった一連の数学の知見がないと理解が厳しいかと思います。読者の方の理解の一助になれば幸いです。
# 
# ※P267の中段部"$p_j\times p_{g'}$。"となっている丸印が合成写像を表す記号の丸だと思って数式を追うのにかなり時間がかかりました。実際は日本語の文末につける丸印でした。日本語の全角の丸はややこしいです･･･
