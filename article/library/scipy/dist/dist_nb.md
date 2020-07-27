
## scipyによる確率分布と特殊関数

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/scipy/dist/dist_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/scipy/dist/dist_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G2022



```python
!python -V
```

    Python 3.7.3


基本的なライブラリをインポートしそのバージョンを確認しておきます。


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
```

    matplotlib version : 3.0.3
    scipy version : 1.4.1
    numpy version : 1.16.2


## 基本的な確率分布

ここではベイズ統計を理解するために必要な基本的な確率分布と必要な特殊関数の説明を行います。基本的な性質をまとめています。また、サンプリングするpythonのコードも示しています。基本的にはscipyモジュールの統計関数を利用するだけです。

### 正規分布

正規分布は最も良く使われる確率分布であり、その詳細は教科書に詳細に書かれているので割愛します。ここではscipyによる利用方法にとどめます。

#### 表式
$
\displaystyle P\left( x \| \mu, \sigma \right) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(x-\mu)^2}{2 \sigma^2} \right)
$

#### 平均
$
E\left[x \right]=\mu
$

#### 分散
$
V\left[x\right]=\sigma^2
$

#### 確率密度関数の取得
pythonで正規分布の確率密度関数の生成方法は以下の通りです。中心0、標準偏差1のグラフになります。


```python
from scipy.stats import norm

mu = 0
sigma = 1

X = np.arange(-5, 5, 0.1)
Y = norm.pdf(X, mu, sigma)

plt.xlabel('$x$')
plt.ylabel('$P$')
plt.grid()
plt.plot(X, Y)
```




    [<matplotlib.lines.Line2D at 0x111fb8b00>]




![svg](dist_nb_files/dist_nb_6_1.svg)


実際に定量的なデータ分析の現場では、確率密度関数自体はそれほど使いません。統計モデリングなどでは、仮定された確率分布からデータを仮想的にサンプリングすることがしばしば行われます。確率分布を仮定した場合のシミュレーションのようなものです。

#### サンプリング
標準正規分布から1000個データをサンプリングして、ヒストグラム表示してみます。ちゃんと確率密度関数を再現できていることがわかります。


```python
from scipy.stats import norm

mu = 0
sigma = 1

x = norm.rvs(mu, sigma, size=1000)
plt.grid()
plt.hist(x, bins=20)
```




    (array([  2.,   2.,   8.,  15.,  28.,  47.,  69.,  90., 130., 152., 116.,
            120.,  92.,  70.,  29.,  18.,   8.,   3.,   0.,   1.]),
     array([-3.32475325, -2.97972419, -2.63469512, -2.28966605, -1.94463699,
            -1.59960792, -1.25457885, -0.90954979, -0.56452072, -0.21949165,
             0.12553741,  0.47056648,  0.81559555,  1.16062461,  1.50565368,
             1.85068275,  2.19571181,  2.54074088,  2.88576995,  3.23079901,
             3.57582808]),
     <a list of 20 Patch objects>)




![svg](dist_nb_files/dist_nb_9_1.svg)


#### help
これ以外にもscipy.stats.normには様々な関数が用意されています。

```python
norm?
```

これを実行するとどのような関数が用意されているかわかります。
`rvs: random variates`や`pdf : probability density function`、`logpdf`、`cdf : cumulative distribution function`などを利用する場合が多いと思います。

```txt

rvs(loc=0, scale=1, size=1, random_state=None)
    Random variates.
pdf(x, loc=0, scale=1)
    Probability density function.
logpdf(x, loc=0, scale=1)
    Log of the probability density function.
cdf(x, loc=0, scale=1)
    Cumulative distribution function.
logcdf(x, loc=0, scale=1)
    Log of the cumulative distribution function.
sf(x, loc=0, scale=1)
    Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
logsf(x, loc=0, scale=1)
    Log of the survival function.
ppf(q, loc=0, scale=1)
    Percent point function (inverse of ``cdf`` --- percentiles).
isf(q, loc=0, scale=1)
    Inverse survival function (inverse of ``sf``).
moment(n, loc=0, scale=1)
    Non-central moment of order n
stats(loc=0, scale=1, moments='mv')
    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
entropy(loc=0, scale=1)
    (Differential) entropy of the RV.
fit(data, loc=0, scale=1)
    Parameter estimates for generic data.
expect(func, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
    Expected value of a function (of one argument) with respect to the distribution.
median(loc=0, scale=1)
    Median of the distribution.
mean(loc=0, scale=1)
    Mean of the distribution.
var(loc=0, scale=1)
    Variance of the distribution.
std(loc=0, scale=1)
    Standard deviation of the distribution.
interval(alpha, loc=0, scale=1)
    Endpoints of the range that contains alpha percent of the distribution

```

### ベルヌーイ分布
 コイン投げにおいて、表、もしくは裏が出る分布のように、試行の結果が2通りしか存在しない場合の分布を決定します。

#### 表式
$
P\left( x \| \mu \right) = \mu^{x}\left(1-\mu \right)^{1-x} \quad \left(x = 0 \text{ or }1 \right) 
$

もしくは、

<div>
$$
 P\left(x | \mu \right)= \begin{cases}
   1 - \mu &\text{if } x = 0 \\
   \mu &\text{if } x = 1
\end{cases}
$$
</div>

となります。$x$が0か1しか取らないことに注意してください。

#### 平均
$
E\left[x \right]=\mu
$

#### 分散
$
V\left[x\right]=\mu\left(1-\mu \right)
$
#### 確率質量関数


```python
from scipy.stats import bernoulli

mu=0.3

print(bernoulli.pmf(0, mu))
print(bernoulli.pmf(1, mu))
```

    0.7000000000000001
    0.3


#### サンプリング
1000個サンプリングして、ヒストグラムで表示してみます。


```python
from scipy.stats import bernoulli

mu=0.3
size=1000

x = bernoulli.rvs(mu, size=size)
plt.hist(x, bins=3)
```




    (array([695.,   0., 305.]),
     array([0.        , 0.33333333, 0.66666667, 1.        ]),
     <a list of 3 Patch objects>)




![svg](dist_nb_files/dist_nb_15_1.svg)


### 二項分布
コイン投げを複数回行った結果、表が出来る回数の確率が従う分布になります。回数が1の時はベルヌーイ分布に一致します。$n$がコインを投げる回数、$p$が表が出る確率、$k$が表が出る回数で確率変数になります。詳細な計算は別途教科書を参照してください。

#### 表式
$ \displaystyle
P\left(k | n,p \right) = \frac{n!}{k!\left(n-k \right)!} p^{k}\left( 1-p\right)^{n-k}
$

#### 平均
$ \displaystyle
E\left[k \right] = np
$

#### 分散
$ \displaystyle
V\left[k \right] = np\left(1-p \right) 
$

#### 確率質量関数

$p=0.3, 0.5, 0.9$の場合の二項分布の確率質量関数になります。


```python
import numpy as np
import scipy
from scipy.stats import binom
import matplotlib.pyplot as plt

n = 100
x = np.arange(n)

# case.1
p = 0.3
y1 = binom.pmf(x,n,p)

# case.2
p = 0.5
y2 = binom.pmf(x,n,p)

# case.3
p = 0.9
y3 = binom.pmf(x,n,p)

# fig, ax = plt.subplots(facecolor="w")
plt.plot(x, y1, label="$p=0.3$")
plt.plot(x, y2, label="$p=0.5$")
plt.plot(x, y3, label="$p=0.9$")

plt.xlabel('$k$')
plt.ylabel('$B(n,p)$')
plt.title('binomial distribution n={}'.format(n))
plt.grid(True)

plt.legend()
```




    <matplotlib.legend.Legend at 0x11ddf7780>




![svg](dist_nb_files/dist_nb_17_1.svg)


#### サンプリング

$n=100, p=0.3$のパラメタを持つ二項分布からサンプリングを行い、確率質量関数が正しいことを確認します。


```python
from scipy.stats import binom

n = 100
p = 0.3

x = binom.rvs(n,p,size=10000)

plt.xlim(0, 60)
plt.grid()
plt.hist(x, bins=15)
```




    (array([  18.,   61.,  208.,  816., 1148., 1516., 1777., 2314., 1023.,
             615.,  329.,  136.,   26.,    9.,    4.]),
     array([15.        , 17.26666667, 19.53333333, 21.8       , 24.06666667,
            26.33333333, 28.6       , 30.86666667, 33.13333333, 35.4       ,
            37.66666667, 39.93333333, 42.2       , 44.46666667, 46.73333333,
            49.        ]),
     <a list of 15 Patch objects>)




![svg](dist_nb_files/dist_nb_19_1.svg)


### マルチヌーイ分布（カテゴリ分布）
ベルヌーイ分布の多変数化になります。ベルヌーイ分布がコインの裏表が出る確率分布を表現するのであれば、サイコロの面が出る確率分布を表現します。次に説明する多項分布の$N=1$の場合に相当します。

#### 表式

$ \displaystyle
P\left(x_1, x_2, \cdots x_n | n,p_1,p_2 \cdots p_n \right) = p_1^{x_1}p_2^{x_2}\cdots p_n^{x_n}
$

ただし、$p_i$と$x_i$は $ \displaystyle \sum_i p_i = 1 , \sum_i x_i = 1 $を満たす。

#### 平均
$ \displaystyle
E\left[x_i \right] = p_i 
$

#### 分散
$ \displaystyle
V\left[x_i \right] = p_i(1-p_i)
$

#### 確率質量関数

scipyではマルチヌーイ分布専用の関数はないようなので、多項分布の$N=1$の場合を利用します。


```python
from scipy.stats import multinomial

rv = multinomial(1,[0.1,0.2,0.25,0.15,0.2,0.1])
print('確率 : ',str(rv.pmf([1,2,1,3,2,1]))[:8])

```

    確率 :  0.0


### 多項分布
二項分布の多変数版です。いかさまの可能性があるサイコロを複数回振って、それぞれの目が出る回数が従う確率分布になります。サイコロの面が$n$個あり、それぞれが一回の試行で出る確率が$p_1,p_2, \cdots , p_n$で、$N$回そのサイコロを振ったとき、それぞれの面が出る確率を$x_1,x_2, \cdots , x_n$とします。

#### 表式
$ \displaystyle
P\left(x_1, x_2, \cdots x_n | n,p_1,p_2 \cdots p_n \right) = \frac{N!}{x_1!x_2! \cdots x_n!} p_1^{x_1}p_2^{x_2}\cdots p_n^{x_n}
$

ただし、$p_i$と$x_i$は $ \displaystyle \sum_i p_i = 1 , \sum_i x_i = N $を満たす。

#### 平均
$ \displaystyle
E\left[x_i \right] = Np_i 
$

#### 分散
$ \displaystyle
V\left[x_i \right] = Np_i\left(1-p_i \right) 
$

#### 確率質量関数
6面体のサイコロで、1から6それぞれの目が出る確率は、$0.1,0.2,0.25,0.15,0.2,01$とします。そのサイコロを10回振って、それぞれの目が1回,2回,1回,3回,2回,1回出る確率を計算します。


```python
from scipy.stats import multinomial

rv = multinomial(10,[0.1,0.2,0.25,0.15,0.2,0.1])
print('確率 : ',str(rv.pmf([1,2,1,3,2,1]))[:8])
```

    確率 :  0.002041


#### サンプリング
上記のサイコロを10回投げて、それぞれの目が出る回数をサンプリングします。とりあえず一回だけサンプリングしてみます。


```python
multinomial.rvs(10, [0.1,0.2,0.25,0.15,0.2,0.1], size=1)
```




    array([[1, 4, 2, 0, 2, 1]])



10000回サンプリングしてみます。


```python
from collections import Counter

array = multinomial.rvs(10, [0.1,0.2,0.25,0.15,0.2,0.1], size=10000)
array = map(str, array)

print('出現する上位10個のデータ')
for i in Counter(array).most_common()[:10]:
  print(i)
```

    出現する上位10個のデータ
    ('[1 2 2 2 2 1]', 63)
    ('[1 2 3 1 2 1]', 55)
    ('[0 3 2 2 2 1]', 52)
    ('[0 2 3 2 2 1]', 50)
    ('[1 1 3 2 2 1]', 46)
    ('[1 2 2 1 3 1]', 46)
    ('[1 2 3 2 2 0]', 43)
    ('[1 3 2 1 2 1]', 43)
    ('[1 2 3 2 1 1]', 41)
    ('[1 3 2 2 2 0]', 40)


結果として、一番出現する確率する3の目が出る出現する目のセットが多いことがわかります。

### ベータ分布

#### 表式
$ \displaystyle
P\left( x | \alpha, \beta \right) = \frac{x^{\alpha - 1}(1-x)^{\beta - 1}}{B \left(\alpha, \beta \right)}
$

ここで$\displaystyle B(\alpha, \beta)$はベータ関数で$\alpha, \beta > 0$となります。

#### 平均
$ \displaystyle
E\left[x \right] = \frac{\alpha}{\alpha + \beta}
$

#### 分散
$ \displaystyle
V\left[x \right] = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}
$


#### 確率質量関数


```python
from scipy.stats import beta

alpha_list = [2,3,4]
beta_list = [2,2,3]

for _alpha,_beta in zip(alpha_list, beta_list):
  x = np.linspace(0,1,100)[1:-1]
  y = beta.pdf(x, _alpha, _beta)
  plt.plot(x,y,label="$\\alpha={}, \\beta={}$".format(_alpha,_beta))

plt.xlabel("$x$")
plt.ylabel("$p(x)$")
plt.grid()
plt.legend()
```




    <matplotlib.legend.Legend at 0x11df12f60>




![svg](dist_nb_files/dist_nb_30_1.svg)


#### サンプリング

$\alpha=2, \beta=2$のベータ関数に対してサンプリングを行いヒストグラム表示して、確率密度関数が確かに正しいことを確認します。


```python
from scipy.stats import beta

_alpha = 2
_beta = 2

plt.grid()
plt.hist(beta.rvs(_alpha, _beta, size=100000))
```




    (array([ 2930.,  7630., 11152., 13576., 14832., 14657., 13517., 11179.,
             7611.,  2916.]),
     array([0.00309102, 0.10256636, 0.20204171, 0.30151706, 0.40099241,
            0.50046776, 0.59994311, 0.69941845, 0.7988938 , 0.89836915,
            0.9978445 ]),
     <a list of 10 Patch objects>)




![svg](dist_nb_files/dist_nb_32_1.svg)


となり、上記の$\alpha=2, \beta=2$のベータ関数と形状が一致する事が分かります。

### ガンマ分布

#### 表式
$ \displaystyle
P(x|\alpha, \beta) = \frac{\beta^\alpha x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}
$

もしくは、

$ \displaystyle
P(x|\alpha, \theta) = \frac{x^{\alpha-1}e^{-\frac{x}{\theta}}}{\Gamma(\alpha)\theta^\alpha }
$

#### 平均
$ \displaystyle
E\left[x \right] = \frac{\alpha}{\beta}
$

#### 分散
$ \displaystyle
V\left[x \right] = \frac{\alpha}{\beta^2}
$

#### 確率質量関数


```python
from scipy.stats import gamma

_alpha_list = [1.0, 2.0, 3.0, 9.0]
_beta_list = [2.0, 2.5, 4.5, 5]

for _alpha, _beta in zip(_alpha_list, _beta_list):
  x = np.linspace(0,4,100)
  y = gamma.pdf(x,_alpha,scale=1/_beta)
  plt.plot(x,y,label="$\\alpha = {}, \\beta = {}$".format(_alpha, _beta))

plt.grid()
plt.legend()
```




    <matplotlib.legend.Legend at 0x11e40d470>




![svg](dist_nb_files/dist_nb_35_1.svg)


#### サンプリング
ベータ分布と同様に、サンプリングを行い、ヒストグラムを作成し、上記の確率密度関数が正しいことを確認します。$\alpha=2.0, \beta=2.5$のヒストグラムを作成します。


```python
_alpha = 2.0
_alpha = 2.5

plt.grid()
plt.hist(gamma.rvs(_alpha, _beta, size=10000), bins=10)
```




    (array([2.865e+03, 3.786e+03, 2.093e+03, 8.100e+02, 2.980e+02, 9.600e+01,
            3.000e+01, 1.100e+01, 9.000e+00, 2.000e+00]),
     array([ 5.03448303,  6.44804252,  7.86160201,  9.27516149, 10.68872098,
            12.10228047, 13.51583995, 14.92939944, 16.34295893, 17.75651841,
            19.1700779 ]),
     <a list of 10 Patch objects>)




![svg](dist_nb_files/dist_nb_37_1.svg)


### カイ二乗分布

標準正規分布に従う確率変数の二乗和の従う分布だそうです。正直あまり使う機会はないですが、区間推定やカイ二乗検定などに利用します。

$x_i \sim N(0,1)$であり、

$$ \displaystyle
Z = \sum_{i=1}^{k}x_i^2
$$

が従う分布という事になります。

#### 表式

$ \displaystyle
f(x|k) = \frac{x^{\frac{k}{2}-1}e^{-\frac{x}{2}}}{2^{\frac{k}{2}}\Gamma\left(\frac{k}{2}\right)} (x \gt 0)
$

#### 平均
$ \displaystyle
E\left[x \right] =  k
$

#### 分散
$ \displaystyle
V\left[x \right] = 2k
$ 


#### 確率質量関数


```python
from scipy.stats import chi2

k = 3.0

x = np.linspace(0, 20 ,100)
y = chi2.pdf(x, k)

plt.grid()
plt.plot(x, y)
plt.show()
```


![svg](dist_nb_files/dist_nb_39_0.svg)


#### サンプリング
$k=3$の場合について、ヒストグラムを作成し、確率密度関数の様子を確認してみます。


```python
plt.grid()
plt.hist(chi2.rvs(3,size=1000),bins=12)
plt.show()
```


![svg](dist_nb_files/dist_nb_41_0.svg)


### ステューデントのt分布

単にt分布という事が多いようです。こちらもt検定などの検定に利用されます。サラリーマン時代、先輩にこの確率分布を紹介され、データ分析に利用したことを覚えています。得られるデータ数が少ない場合に適用可能で、製造業の工場などでも利用できると思います。

#### 表式

$ \displaystyle
p(x | \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{x^2}{\nu}\right)^{-\left(\frac{\nu+1}{2}\right)}
$

#### 平均
$ \displaystyle
E\left[x \right] = 0 \cdots (k > 1)
$

#### 分散
$ \displaystyle
V\left[x \right] = \frac{k}{k-2} \cdots (k > 2)
$


#### 確率質量関数


```python
from scipy.stats import t

nu_list = [1,3,5]

plt.grid()

for nu in nu_list:

  x = np.linspace(-5,5,100)
  y = t.pdf(x,nu)
  
  plt.plot(x,y,label="$\\nu = {}$".format(nu))

plt.title("t-distribution")
plt.legend()
plt.show()
```


![svg](dist_nb_files/dist_nb_43_0.svg)


#### サンプリング

$\nu = 3$の場合について、確率密度関数からサンプリングを行い、ヒストグラム表示してみます。


```python
from scipy.stats import t

nu = 3

plt.grid()
plt.xlim(-10,10)
plt.hist(t.rvs(nu, size=10000), bins=80)
plt.show()
```


![svg](dist_nb_files/dist_nb_45_0.svg)


## 多変数ガウス分布

統計モデリングでは相関関係のある多変数ガウス分布からサンプリングする事があります。

$$ \displaystyle
P(x| \mu, \Sigma) = \frac{1}{\sqrt{(2 \pi)^{k} \det\Sigma}} \exp \left( - \frac{(x-\mu)^{T}\Sigma^{-1}(x-\mu)}{2} \right)
$$




```python
from scipy.stats import multivariate_normal

mu = np.array([0,0])
sigma = np.array([[1,0],[0,1]])

x1 = np.linspace(-3,3,100)
x2 = np.linspace(-3,3,100)

X = np.meshgrid(x1,x2)

print(X[0].shape)
print(X[1].shape)
print(np.ravel(X[0]))
X = np.array([np.ravel(X[0]), np.ravel(X[1])])
XX = []
for i,j in zip(np.ravel(X[0]), np.ravel(X[1])):
  XX.append([i,j])

Z = multivariate_normal.pdf(XX, mu,sigma)

```

    (100, 100)
    (100, 100)
    [-3.         -2.93939394 -2.87878788 ...  2.87878788  2.93939394
      3.        ]



```python
sample = multivariate_normal.rvs(mu, sigma, 1000)
sample
```




    array([[ 0.4369545 ,  1.25467804],
           [ 0.21130174, -0.54498372],
           [ 2.71295515,  1.24101205],
           ...,
           [-0.92113178, -1.63351393],
           [ 0.47291447,  0.65840656],
           [-1.14134527, -1.91980601]])



## 指数型分布族

データ分析やマーケティングの分析の際に出てくる確率分布は、正規分布、二項分布、ポアソン分布など多岐にわたりますが、そのほとんどは、指数型分布属といわれる形をしています。


#### 表式

$ \displaystyle
f(x|\theta) = h(x)\exp (\eta(\theta)\cdot T(x) - A(\theta))
$


```python
%matplotlib nbagg

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
x = np.arange(0, 10, 0.1)

ims = []
for a in range(50):
  y = np.sin(x - a)
  im = plt.plot(x, y, "r")
  ims.append(im)

ani = animation.ArtistAnimation(fig, ims)
plt.show()
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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8z3Wh4o1wl5vYps8VeMFNi0+LHkB+FKxKHPXWdYArihuN45n/mJ2RLf2ExgogMVCzLF2a/zXhZjbZHwNYG0yUC1oZlwFiwc349PTS4GYujQWfIzV2eMB7nh42x/mLRiYDKeMhRvPLJ4Nik9TY82Q3ltcDY193KopYACr3rjGz+6Y/WyhYk2wWP8rZkGM29EmFWsG/aF63e64HsXdKBGoY13ON4F4Fiye2421/tYuFmGO24l91q+9Xhoxq2V8xTZHsZZfPMMXVzRjiyUHYqmgri32HVus9Xd+8SFqXPUcUfx9BPX4d+9WEgEDWEkGoqTNGF5cFYlbVT5d3NISa4wc6BWSko5Yfc3qevM9uaOdsro+23Y5ygDWLiM1uHbGIr1x9StC19zFG/S4zTRmx3M2tMFZlnnvqYu1Ob9ZrPUUk2/EXQwRumNmxD+XufG2rU+BgX4vP1lcwe55cFz9iiVGYrbD6YpHSc4oXgMf6VwuAQNYucbD1iiggAIKKKCAAgoooECFBQxgFR5cu6aAAgoooIACCiiggALlEjCAlWs8bI0CCiiggAIKKKCAAgpUWMAAVuHBtWsKKKCAAgoooIACCihQLgEDWLnGw9YooIACCiiggAIKKKBAhQUMYBUeXLumgAIKKKCAAgoooIAC5RIwgJVrPGyNAgoooIACCiiggAIKVFjAAFbhwbVrCiiggAIKKKCAAgooUC4BA1i5xsPWKKCAAgoooIACCiigQIUFDGAVHly7poACCiiggAIKKKCAAuUSMICVazxsjQIKKKCAAgoooIACClRYwABW4cG1awoooIACCiiggAIKKFAuAQNYucbD1iiggAIKKKCAAgoooECFBQxgFR5cu6aAAgoooIACCiiggALlEjCAlWs8bI0CCiiggAIKKKCAAgpUWMAAVuHBtWsKKKCAAgoooIACCihQLgEDWLnGw9YooIACCiiggAIKKKBAhQUMYBUeXLumgAIKKKCAAgoooIAC5RIwgJVrPGyNAgoooIACCiiggAIKVFjAAFbhwbVrCiiggAIKKKCAAgooUC4BA1i5xsPWKKCAAgoooIACCiigQIUFDGAVHly7poACCiiggAIKKKCAAuUSMICVazxsjQIKKKCAAgoooIACClRYwABW4cG1awoooIACCiiggAIKKFAuAQNYucbD1iiggAIKKKCAAgoooECFBQxgFR5cu6aAAgoooIACCiiggALlEjCAlWs8bI0CCiiggAIKKKCAAgpUWMAAVuHBtWsKKKCAAgoooIACCihQLgEDWLnGw9YooIACCiiggAIKKKBAhQUMYBUeXLumgAIKKKCAAgoooIAC5RIwgJVrPGyNAgoooIACCiiggAIKVFjAAFbhwbVrCiiggAIKKKCAAgooUC4BA1i5xsPWKKCAAgoooIACCiigQIUFDGAVHly7poACCiiggAIKKKCAAuUSMICVazxsjQIKKKCAAgoooIACClRYwABW4cG1awoooIACCiiggAIKKFAuAQNYucbD1iiggAIKKKCAAgoooECFBQxgFR5cu6aAAgoooIACCiiggALlEjCAlWs8bI0CCiiggAIKKKCAAgpUWMAAVuHBtWsKKKCAAgoooIACCihQLgEDWLnGw9YooIACCiiggAIKKKBAhQUMYBUeXLumgAIKKKCAAgoooIAC5RIwgJVrPGyNAgoooIACCiiggAIKVFjg/wEZPcHVc7esewAAAABJRU5ErkJggg==" width="432">

