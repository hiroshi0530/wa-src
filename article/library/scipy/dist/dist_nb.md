
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
    BuildVersion:	18G6020



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




    [<matplotlib.lines.Line2D at 0x112cdfa58>]




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




    (array([  1.,   1.,   8.,  20.,  28.,  52.,  73., 104., 122., 132., 130.,
            107.,  83.,  62.,  37.,  24.,   9.,   3.,   3.,   1.]),
     array([-3.22122581, -2.89149803, -2.56177025, -2.23204247, -1.90231468,
            -1.5725869 , -1.24285912, -0.91313134, -0.58340355, -0.25367577,
             0.07605201,  0.40577979,  0.73550758,  1.06523536,  1.39496314,
             1.72469092,  2.05441871,  2.38414649,  2.71387427,  3.04360205,
             3.37332984]),
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




    (array([709.,   0., 291.]),
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




    <matplotlib.legend.Legend at 0x11b425668>




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




    (array([1.100e+01, 7.900e+01, 2.070e+02, 8.220e+02, 1.134e+03, 2.335e+03,
            1.673e+03, 2.117e+03, 8.170e+02, 5.900e+02, 1.370e+02, 6.900e+01,
            7.000e+00, 0.000e+00, 2.000e+00]),
     array([14.        , 16.53333333, 19.06666667, 21.6       , 24.13333333,
            26.66666667, 29.2       , 31.73333333, 34.26666667, 36.8       ,
            39.33333333, 41.86666667, 44.4       , 46.93333333, 49.46666667,
            52.        ]),
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




    array([[0, 5, 2, 0, 1, 2]])



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
    ('[1 2 3 1 2 1]', 68)
    ('[1 2 3 2 1 1]', 47)
    ('[1 3 3 1 2 0]', 47)
    ('[0 2 3 2 2 1]', 46)
    ('[0 2 3 1 3 1]', 45)
    ('[1 2 2 1 2 2]', 45)
    ('[1 3 2 1 2 1]', 42)
    ('[1 2 3 2 2 0]', 42)
    ('[0 2 4 2 1 1]', 42)
    ('[1 2 4 1 2 0]', 40)


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




    <matplotlib.legend.Legend at 0x11b6ed208>




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




    (array([ 2835.,  7639., 11106., 13515., 14798., 14875., 13482., 11280.,
             7557.,  2913.]),
     array([0.00161106, 0.10122898, 0.20084689, 0.30046481, 0.40008273,
            0.49970065, 0.59931856, 0.69893648, 0.7985544 , 0.89817232,
            0.99779023]),
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




    <matplotlib.legend.Legend at 0x11ba66240>




![svg](dist_nb_files/dist_nb_35_1.svg)


#### サンプリング
ベータ分布と同様に、サンプリングを行い、ヒストグラムを作成し、上記の確率密度関数が正しいことを確認します。$\alpha=2.0, \beta=2.5$のヒストグラムを作成します。


```python
_alpha = 2.0
_alpha = 2.5

plt.grid()
plt.hist(gamma.rvs(_alpha, _beta, size=10000), bins=10)
```




    (array([2.746e+03, 3.692e+03, 2.138e+03, 9.090e+02, 3.360e+02, 1.260e+02,
            3.700e+01, 1.000e+01, 4.000e+00, 2.000e+00]),
     array([ 5.05073076,  6.41202124,  7.77331172,  9.13460219, 10.49589267,
            11.85718315, 13.21847363, 14.5797641 , 15.94105458, 17.30234506,
            18.66363554]),
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

$\Sigma$は共分散行列、$\mu$はそれぞれの確率変数の平均です。2次元以上の場合はベクトルになります。

$\Sigma$を操作することで、変数間の相関を設定すること出来ます。

$$
  \Sigma = \left(
    \begin{array}{cc}
      \sigma_{x_1}^2 & \sigma_{x_1x_2} \\\\
      \sigma_{x_1x_2} & \sigma_{x_2}^2
    \end{array}
  \right)
$$

共分散行列は半正定値対称行列の条件を満たし、すべての固有値は非負などと言った便利な性質があります。数値計算では良く出てくる性質の行列です。

### 完全非相関のプロット

$\sigma_{x_1x_2}=0$の場合は二つの確率変数に相関関係はありません。
わかりやすいように、z軸から見たグラフを示します。


```python
from scipy.stats import multivariate_normal

mu = np.array([0,0])
sigma = np.array([[1,0],[0,1]])

x1 = np.linspace(-3,3,100)
x2 = np.linspace(-3,3,100)

X = np.meshgrid(x1,x2)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]
Z = multivariate_normal.pdf(X, mu,sigma).reshape(100, -1)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-90, azim=0)
surf = ax.plot_surface(X1, X2, Z, cmap='bwr', linewidth=0)
fig.show()

```

    /Users/hiroshi/anaconda3/lib/python3.7/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
      % get_backend())



![svg](dist_nb_files/dist_nb_47_1.svg)


### 正の相関を持つガウス分布

$\sigma_{x_1x_2} = 0.8$の正の相関を持つ二元ガウス分布のグラフです。


```python
from scipy.stats import multivariate_normal

mu = np.array([0,0])
sigma = np.array([[1,0.8],[0.8,1]])

x1 = np.linspace(-3,3,100)
x2 = np.linspace(-3,3,100)

X = np.meshgrid(x1,x2)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]
Z = multivariate_normal.pdf(X, mu,sigma).reshape(100, -1)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-90, azim=0)
surf = ax.plot_surface(X1, X2, Z, cmap='bwr', linewidth=0)
fig.show()
```


![svg](dist_nb_files/dist_nb_49_0.svg)


### 負の相関を持つガウス分布

$\sigma_{x_1x_2} = -0.8$の負の相関を持つ二元ガウス分布のグラフです。


```python
from scipy.stats import multivariate_normal

mu = np.array([0,0])
sigma = np.array([[1,-0.8],[-0.8,1]])

x1 = np.linspace(-3,3,100)
x2 = np.linspace(-3,3,100)

X = np.meshgrid(x1,x2)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]
Z = multivariate_normal.pdf(X, mu,sigma).reshape(100, -1)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-90, azim=0)
surf = ax.plot_surface(X1, X2, Z, cmap='bwr', linewidth=0)
fig.show()
```


![svg](dist_nb_files/dist_nb_51_0.svg)


## 指数型分布族

データ分析やマーケティングの分析の際に出てくる上記で示した確率分布は、正規分布、二項分布、ポアソン分布など多岐にわたりますが、そのほとんどは、指数型分布族といわれる形をしています。

#### 表式

$ \displaystyle
f(x|\theta) = h(x)\exp (\eta(\theta)\cdot T(x) - A(\theta))
$

この形を持つ分布関数は共益事前分布を持つなどという重要な性質を持ちます。詳細は機会が来たときに書こうと思います。
