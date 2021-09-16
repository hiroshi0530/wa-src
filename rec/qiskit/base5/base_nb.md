## 量子フーリエ変換

qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。

qiskitのウェブサイト通りに勉強を進めています。

- https://qiskit.org/textbook/ja/ch-algorithms/quantum-fourier-transform.html

次は、量子フーリエ変換の復習をしてみます。学生時代に理解したつもりでしたが、全く忘れているので、１から勉強し直しです。

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base5/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base5/base_nb.ipynb)

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

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('pandas version :', pd.__version__)
```

    matplotlib version : 3.3.2
    scipy version : 1.5.2
    numpy version : 1.19.2
    pandas version : 1.1.3



```python
import qiskit
import json

dict(qiskit.__qiskit_version__)
```




    {'qiskit-terra': '0.17.4',
     'qiskit-aer': '0.8.2',
     'qiskit-ignis': '0.6.0',
     'qiskit-ibmq-provider': '0.13.1',
     'qiskit-aqua': '0.9.1',
     'qiskit': '0.26.2',
     'qiskit-nature': None,
     'qiskit-finance': None,
     'qiskit-optimization': None,
     'qiskit-machine-learning': None}



## 離散フーリエ変換の復習

フーリエ変換は、関数$f(x)$は様々な波数を持つ三角関数の和として表現できるというフーリエの考え方から来ています。

$$
f(x)=\frac{a_{0}}{2}+a_{1} \cos \frac{\pi x}{L}+b_{1} \sin \frac{\pi x}{L}+a_{2} \cos \frac{2 \pi x}{L}+b_{2} \sin \frac{2 \pi x}{L}+\cdots
$$

ここで、　$a_1$や$b_1$などは、それぞれの波数を持つ三角関数の振幅であり、関数＄f(x)＄を三角関数で分解したときの重みと考えられます。これらの振幅を求めるには三角関数の公式を利用して以下の様に求める事が出来ます。

$$
a_{n}=\frac{1}{L} \int_{-L}^{L} f(x) \cos \frac{n \pi x}{L} d x \quad(n=0,1,2, \cdots)
$$

$$
b_{n}=\frac{1}{L} \int_{-L}^{L} f(x) \sin \frac{n \pi x}{L} d x \quad(n=1,2, \cdots)
$$

それぞれの振幅をフーリエ係数と呼ぶようです。

これを$L \rightarrow \infty$としてあげて、$\cos$と$\sin$をオイラー数を用いて変換してあげると、以下の公式が導かれます。

$$
\begin{aligned}
&{F}(k)=\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{\infty} f(x) e^{-i k x} d x \\
&f(x)=\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{\infty} \hat{F}(k) e^{i k x} d k
\end{aligned}
$$

今関数$f(x)$は連続関数という仮定がありましたが、これを離散化してあげると、

$$
\begin{aligned}
y_{k}=\frac{1}{\sqrt{2^{n}}} \sum_{j=0}^{2^{n}-1} x_{j} \exp \left(i \frac{2 \pi k j}{2^{n}}\right) \\
x_{k}=\frac{1}{\sqrt{2^{n}}} \sum_{j=0}^{2^{n}-1} y_{j} \exp \left(-i \frac{2 \pi k j}{2^{n}}\right) 
\end{aligned}
$$

と変換できます。これは、ベクトル$\left(x_{0}, \ldots, x_{N-1}\right)$からベクトル$\left(y_{0}, \ldots, y_{N-1}\right)$へのマッピングに相当します。

フーリエ変換は実世界では欠かせない技術で、サラリーマン時代、アナログ回路設計や半導体設計をしてきましたが、これなしではデバイスを開発することは出来ません。

## 量子フーリエ変換

先ほどの議論を量子状態の変換に応用します。$|\boldsymbol{x}\rangle$、$|\boldsymbol{y}\rangle$を以下の様な重ね合わせの状態と定義します。

$$
|\boldsymbol{x}\rangle=\sum_{j=0}^{2^{n}-1} x_{j}|j\rangle 
$$

$$
|\boldsymbol{y}\rangle=\sum_{k=0}^{2^{n}-1} y_{k}|k\rangle
$$

$|\boldsymbol{x}\rangle \rightarrow |\boldsymbol{y}\rangle$への変換を考えます。$|\boldsymbol{y}\rangle$は、以下の様に変換できます。

$$
\begin{aligned}
|\boldsymbol{y}\rangle&=\sum_{k=0}^{2^{n}-1} y_{k}|k\rangle \\
&=\frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n}-1} \sum_{j=0}^{2^{n}-1} x_{j} \exp \left(i \frac{2 \pi k j}{2^{n}}\right)|k\rangle \\
&=\sum_{j=0}^{2^{n}-1} x_{j}\left(\frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n}-1} \exp \left(i \frac{2 \pi k j}{2^{n}}\right)|k\rangle\right)
\end{aligned}
$$

つまり、

$$
|\boldsymbol{j}\rangle \rightarrow \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n}-1} \exp \left(i \frac{2 \pi k j}{2^{n}}\right)|k\rangle
$$

という変換が量子フーリエ変換になることが分かります。一般的にこの$\exp$の部分を

$$
W=w^{k j}=\exp \left(i \frac{2 \pi kj}{2^{n}}\right)
$$

と書き、この$W$は

$$
W^{\dagger} W=W W^{\dagger}=I
$$

を満たすので、量子フーリエ変換はユニタリ変換になります。


```python

```

$$
\begin{aligned}
& \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n}-1} \exp \left(i \frac{2 \pi k j}{2^{n}}\right)|k\rangle \\
&=\frac{1}{\sqrt{2^{n}}} \sum_{k_{1}=0}^{1} \sum_{k_{2}=0}^{1} \cdots \sum_{k_{n}=0}^{1} \exp \left(i \frac{2 \pi j\left(k_{n} k_{n-1} \cdots k_{1}\right)_{2}}{2^{n}}\right)|k_{n}k_{n-1} \cdots k_{1}\rangle \\
&=\frac{1}{\sqrt{2^{n}}} \sum_{k_{1}={0}}^1 \sum_{k_{2}=0}^{1} \cdots \sum_{k_{n}=0}^{1} \exp \left(i \cdot 2 \pi j\left(0. k_{n} k_{n-1} \cdots k_{1}\right)_{2}\right)|k_{n} k_{n-1} \cdots k_{1}\rangle\\
&=\frac{1}{\sqrt{2^{n}}}\left(\sum_{k_{n}=0}^{1} \exp \left(i^{1} \cdot 2 \pi j\left(0. k_{n}\right)_{2}|k_{n}\rangle \right)\right) \otimes\left(\sum_{k_{n}-0}^{1} \exp \left(i \cdot 2 \pi j\left(0.0 k_{n-1}\right)_{2} \mid k_{n-1}\right)\right) \otimes \cdots\\
& \cdots \otimes \left(\sum_{k_{1}=0}^{1} \exp (i \cdot 2 \pi j(0. \underbrace{0 \cdots 0 k_{1}}_{n})|k_{1}\rangle\right)
\end{aligned}
$$

$$
\begin{aligned}
\end{aligned}
$$


```python

```


```python

```

$$
\frac{1}{\sqrt{2^n}}\left(|0\rangle+e^{\frac{2 \pi i}{2} x}|1\rangle\right) \otimes\left(|0\rangle+e^{\frac{2 \pi i}{2^{2}} x}|1\rangle\right) \otimes \ldots \otimes\left(|0\rangle+e^{\frac{2 \pi i}{2^{n-1}} x}|1\rangle\right) \otimes\left(|0\rangle+e^{\frac{2 \pi i}{2^{n}} x}|1\rangle\right)
$$


```python

```


```python

```

## qiskitを用いた量子フーリエ変換回路の作成


```python
qc = QuantumCircuit(3)
```


```python
qc.h(2)
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_18_0.svg)
    




```python
qc.cp(pi/2, 1, 2) # qubit 1 から qubit 2へのCROT 
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_19_0.svg)
    




```python
qc.cp(pi/4, 0, 2) # qubit 0から qubit 2へのCROT 
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_20_0.svg)
    




```python
qc.h(1)
qc.cp(pi/2, 0, 1) # qubit 0からqubit 1へのCROT 
qc.h(0)
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_21_0.svg)
    




```python
qc.swap(0,2)
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_22_0.svg)
    




```python
backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,backend).result().get_statevector()
array_to_latex(final_state, pretext="\\\\text{Statevector} = ")
```


$\displaystyle 
\\text{Statevector} = \begin{bmatrix}
\tfrac{1}{\sqrt{8}} \\
\tfrac{1}{\sqrt{8}} \\
\tfrac{1}{\sqrt{8}} \\
\tfrac{1}{\sqrt{8}} \\
\tfrac{1}{\sqrt{8}} \\
\tfrac{1}{\sqrt{8}} \\
\tfrac{1}{\sqrt{8}} \\
\tfrac{1}{\sqrt{8}}
\end{bmatrix}
$


## 2量子ビットの場合をqiskitで確認

2ビットにおける量子フーリエ変換を手計算の結果が合っているかqiskitで確認してみます。


```python
from numpy import pi
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit_textbook.tools import array_to_latex
```

### 00の場合

$$
\begin{aligned}
\left|00\right\rangle &=\frac{1}{\sqrt{2^{2}}} \sum_{k=0}^{3} \exp \left(i \frac{2 \pi k j}{2^{2}}\right)|k\rangle \\
&=\frac{1}{\sqrt{2^{2}}} \sum_{k=0}^{3} |k\rangle \\
&=\frac{1}{2}\left(|00\rangle+|01\rangle+|10\rangle+|11\rangle \right)
\end{aligned}
$$


```python
qc = QuantumCircuit(2)
qc.h(1)
qc.cp(pi/2, 0, 1)
qc.h(0)
qc.swap(0,1)
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_27_0.svg)
    




```python
backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,backend).result().get_statevector()
array_to_latex(final_state, pretext="\\\\text{Statevector} = ")
```


$\displaystyle 
\\text{Statevector} = \begin{bmatrix}
\tfrac{1}{2} \\
\tfrac{1}{2} \\
\tfrac{1}{2} \\
\tfrac{1}{2}
\end{bmatrix}
$


となり、一致しています。

### 01の場合

$$
\begin{aligned}
|01\rangle &=\frac{1}{\sqrt{2^{2}}} \sum_{k=1}^{3} \exp \left(i \frac{2 \pi}{2^{2}} k\right)|k\rangle \\
&=\frac{1}{2}\left(|00\rangle+i|01\rangle-|10\rangle-i|11\rangle \right)
\end{aligned}
$$


```python
qc = QuantumCircuit(2)
qc.x(0)
qc.h(1)
qc.cp(pi/2, 0, 1)
qc.h(0)
qc.swap(0,1)
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_30_0.svg)
    




```python
backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,backend).result().get_statevector()
array_to_latex(final_state, pretext="\\\\text{Statevector} = ")
```


$\displaystyle 
\\text{Statevector} = \begin{bmatrix}
\tfrac{1}{2} \\
\tfrac{1}{2}i \\
-\tfrac{1}{2} \\
-\tfrac{1}{2}i
\end{bmatrix}
$


となり、一致しています。

### 10の場合

$$
\begin{aligned}
|10\rangle &=\frac{1}{\sqrt{2^{2}}} \sum_{k=0}^{3} \exp \left(i \frac{2 \pi \cdot 2}{2^{2}} k\right)|k\rangle \\
&=\frac{1}{2} \sum_{k=0}^{3} \exp (i \pi k)|k\rangle \\
&=\frac{1}{2}\left(|00\rangle-|01\rangle+|10\rangle-|11\rangle\right)
\end{aligned}
$$


```python
qc = QuantumCircuit(2)
qc.x(1)
qc.h(1)
qc.cp(pi/2, 0, 1)
qc.h(0)
qc.swap(0,1)
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_33_0.svg)
    




```python
backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,backend).result().get_statevector()
array_to_latex(final_state, pretext="\\\\text{Statevector} = ")
```


$\displaystyle 
\\text{Statevector} = \begin{bmatrix}
\tfrac{1}{2} \\
-\tfrac{1}{2} \\
\tfrac{1}{2} \\
-\tfrac{1}{2}
\end{bmatrix}
$


となり、こちらも一致しています。

### 11の場合

$$
\begin{aligned}
|11 \mid &=\frac{1}{\sqrt{2^{2}}} \sum_{k=0}^{3} \exp \left(i \frac{2 \pi}{2^{2}} 3 k\right)|k\rangle \\
&=\frac{1}{2} \sum_{k=0}^{3} \exp \left(i \frac{3}{2} \pi k\right)|k\rangle \\
&=\frac{1}{2}\left(|00\rangle-i|01\rangle-|10\rangle+i|11\rangle \right)
\end{aligned}
$$


```python
qc = QuantumCircuit(2)
qc.x(0)
qc.x(1)
qc.h(1)
qc.cp(pi/2, 0, 1)
qc.h(0)
qc.swap(0,1)
qc.draw('mpl')
```




    
![svg](base_nb_files/base_nb_36_0.svg)
    




```python
backend = Aer.get_backend('statevector_simulator')
final_state = execute(qc,backend).result().get_statevector()
array_to_latex(final_state, pretext="\\\\text{Statevector} = ")
```


$\displaystyle 
\\text{Statevector} = \begin{bmatrix}
\tfrac{1}{2} \\
-\tfrac{1}{2}i \\
-\tfrac{1}{2} \\
\tfrac{1}{2}i
\end{bmatrix}
$


以上より、手計算の結果が確認出来ました。


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
sim = Aer.get_backend("aer_simulator")
qc_init = qc.copy()
qc_init.save_statevector()
statevector = sim.run(qc_init).result().get_statevector()
plot_bloch_multivector(statevector)
```




    
![png](base_nb_files/base_nb_45_0.png)
    




```python

```


```python

```


```python

```








$$
\begin{aligned}
&\frac{1}{\sqrt{2^{n}}} \sum_{k_{1}=0}^{1} \cdots \sum_{k_{n}=0}^{1} \exp \left(i \frac{2 \pi\left(k_{1} 2^{n-1}+\cdots k_{n} 2^{0}\right) \cdot j}{2^{n}}\right)\left|k_{1}\right\rangle\left|k_{2}\right\rangle \cdots\left|k_{n}\right\rangle \\
&=\frac{1}{\sqrt{2^{n}}} \sum_{k_{1}=0}^{1} \cdots \sum_{k_{n}=0}^{1} \exp \left(i 2 \pi j\left(k_{1} 2^{-1}+\cdots k_{n} 2^{-n}\right)\right)\left|k_{1} k_{2} \cdots k_{n}\right\rangle \\
&=\frac{1}{\sqrt{2^{n}}}\left(\sum_{k_{1}=0}^{1} e^{i 2 \pi j k_{1} 2^{-1}}\left|k_{1}\right\rangle\right) \otimes \cdots \otimes\left(\sum_{k_{n}=0}^{1} e^{i 2 \pi j k_{n} 2^{-n}}\left|k_{n}\right\rangle\right) \\
&=\frac{1}{\sqrt{2^{n}}}\left(|0\rangle+e^{i 2 \pi 0 . j_{n}}|1\rangle\right) \otimes\left(|0\rangle+e^{i 2 \pi 0 . j_{n-1} j_{n}}|1\rangle\right) \otimes \cdots \\
&\otimes\left(|0\rangle+e^{i 2 \pi 0 . j_{2} j_{3} \cdots j_{n}}|1\rangle\right) \otimes\left(|0\rangle+e^{i 2 \pi 0 . j_{1} j_{2} \cdots j_{n}}|1\rangle\right)
\end{aligned}
$$


```python

```


```python

```


```python

```


```python

```
