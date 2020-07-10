
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


### 主要確率分布まとめ

<!-- <div style="width:100%; margin: 10px 40px 10px 40px;"> -->
<style>.cent td {text-align:center;}</style>
<style>.cent tr {text-align:center;}</style>

<div style="width:100%;"> 
<table class="cent">
  <tr>
    <th>名前</th>
    <th>確率密度関数</th>
    <th>確率変数</th>
    <th>params</th>
    <th>$\displaystyle E[x]$</th>
    <th>$\displaystyle V[x]$</th>
    <th>概要</th>
  </tr>
  <tr>
    <td>二項分布</td>
    <td>$\displaystyle \binom{n}{k}p^k\left(1-p\right)^{n-k}$</td>
    <td>$k$</td>
    <td>$n,p$</td>
    <td>$np$</td>
    <td>$np(1-p)$</td>
    <td>成功確率$\displaystyle p $の試行を$n$回行い、その成功回数が従う確率分布 </td>
  </tr>
  <tr>
    <td>ポアソン分布</td>
    <td>$\displaystyle \dfrac{\lambda^ke^{-\lambda}}{k!}$</td>
    <td>$k$</td>
    <td>$\lambda$</td>
    <td>$\lambda$</td>
    <td>$\lambda$</td>
    <td align="left">単位時間あたり$\displaystyle \lambda$回起こる事象の、単位時間あたりの発生回数が従う確率分布</td>
  </tr>
  <tr>
    <td>正規分布</td>
    <td>$\displaystyle \dfrac{1}{\sqrt{2\pi \sigma^2}}\exp\left(-\dfrac{\left(x-\mu\right)^2}{2\sigma^2}\right)$</td>
    <td>$x$</td>
    <td>$\mu,\sigma$</td>
    <td>$\mu$</td>
    <td>$\sigma^2$</td>
    <td></td>
  </tr>
  <tr>
    <td>幾何分布</td>
    <td>$\displaystyle p\left(1-p\right)^k$</td>
    <td>$k$</td>
    <td>$p$</td>
    <td>$\displaystyle \dfrac{1-p}{p}$</td>
    <td>$\displaystyle \dfrac{1-p}{p^2}$</td>
    <td align="left">成功確率$\displaystyle p $の試行を行い、はじめての成功を得られるまでに必要な失敗の回数が従う確率分布</td>
  </tr>
  <tr>
    <td>指数分布</td>
    <th>$\lambda e^{-\lambda x} $</th>
    <td>$\displaystyle x $</td>
    <td>$\displaystyle \lambda $</td>
    <td>$\displaystyle \dfrac{1}{\lambda} $</td>
    <td>$\displaystyle \dfrac{1}{\lambda^2} $</td>
    <td>単位時間あたり$\displaystyle \lambda$回起こる事象において、始めて発生する時間が従う確率分布</td>
  </tr>
  <tr>
    <td>負の二項分布</td>
    <td>$\displaystyle \binom{n+k-1}{k-1}p^n\left(1-p\right)^{k}$</td>
    <td>$k$</td>
    <td>$n,p$</td>
    <td>$\displaystyle \dfrac{n}{p}$</td>
    <td>$\displaystyle \dfrac{n\left(1-p\right)}{p^2}$</td>
    <td align="left">成功確率$\displaystyle p $の試行を行い、$n$回の成功を得られるまでに必要な失敗の回数が従う確率分布 (定義は他にもあり)</td>
  </tr>
  <tr>
    <td>ガンマ分布</td>
    <td>$\displaystyle \dfrac{x^{n-1}\lambda^{n}}{\Gamma\left(n\right)}e^{-\lambda x} $ <br>for $x > 0$</td>
    <td>$x$</td>
    <td>$n,\lambda$</td>
    <td>$\displaystyle \dfrac{n}{\lambda}$</td>
    <td>$\displaystyle \dfrac{n}{\lambda^2}$</td>
    <td>単位時間あたり$\displaystyle \lambda$回起こる事象において、$n$回発生する時間が従う確率分布</td>
  </tr>
</table>
</div>
