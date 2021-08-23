## サイモンアルゴリズム

qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。

qiskitのウェブサイト通りに勉強を進めています。

- https://qiskit.org/textbook/ja/ch-algorithms/simon.html

今回は、サイモンのアルゴリズムを数式を追って理解を深めようと思います。

ドイチェ-ジョサの問題設定は、関数$f(x)$が、定数型か分布型のどちらか判別するいう事でしたが、サイモンのアルゴリズムの問題設定は、1:1関数か、2:1の関数かのどちらかを判定するという違いです。その違いを判別し、さらに、2:1の関数の周期を求める事になります。

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base3/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base3/base_nb.ipynb)

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




```python
from qiskit import IBMQ, Aer, execute
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile

from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import array_to_latex
```

## 問題設定

$$
f:\lbrace 0,1 \rbrace^{n} \rightarrow \lbrace 0,1 \rbrace^{n-1}
$$

$$
x \in\{0,1\}^{n}
$$

$$
f(x \oplus a)=f(x)
$$

どちらの関数を見極めるには、最大で、$2^{n-1}+1$回の関数の実行が必要です。運良く、異なる入力に対して、同じ出力が2回連続で出た場合は、2対1型の関数だと分かります。

古典コンピューター上で、回数の下限が$\Omega\left(2^{n / 2}\right)$となるアルゴリズムが知られているようですが、それでも$n$に対して指数関数的に増加します。


```python

```

$$
\left(\begin{array}{ll}
\cos (\theta / 2) & -\mathrm{ie}^{-i \varphi} \sin (\theta / 2) \\
-\mathrm{i} e^{i \varphi} \sin (\theta / 2) & \cos (\theta / 2)
\end{array}\right)
$$

$$
|0\rangle \longmapsto \frac{1}{\sqrt{2}^{n}} \sum_{k=0}^{2 n-1}|k\rangle
$$

$$
\begin{aligned}
\frac{1}{\sqrt{2}^{n}} \sum_{k=0}^{2 n-1}|k\rangle \otimes|0\rangle \longmapsto \frac{1}{\sqrt{2^{n}} \displaystyle \sum_{k=0}^{2-1}|k\rangle \otimes|f(k)\rangle}
\end{aligned}
$$

$$
\begin{aligned}
&\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle) \otimes|\psi\rangle  \\
& \longrightarrow \frac{1}{\sqrt{2}}(|0\rangle \otimes|\psi\rangle+|1\rangle \otimes U|\psi\rangle) \\
&=\frac{1}{\sqrt{2}}\left(|0\rangle \otimes|\psi\rangle+e^{i \lambda}|1\rangle \otimes|\psi\rangle\right) \\
&=\frac{1}{\sqrt{2}}\left(|0\rangle+e^{i \lambda}|1\rangle\right) \otimes|\psi\rangle
\end{aligned}
$$




$$
\begin{array}{|r|r|r|r|}
\hline \mathrm{x} & \mathrm{f}(\mathrm{x}) & \mathrm{y}(=\mathrm{x} \oplus \mathrm{b}) & \mathrm{x} \cdot \mathrm{b} \\
\hline 00 & 00 & 11 & 0 \\
01 & 10 & 10 & 1 \\
10 & 10 & 01 & 1 \\
11 & 00 & 00 & 0 \\
\hline
\end{array}
$$

$$
\begin{array}{|r|r|r|r|}
\hline \mathrm{x} & \mathrm{f}(\mathrm{x}) & \mathrm{y}(=\mathrm{x} \oplus \mathrm{b}) & \mathrm{x} \cdot \mathrm{b} \\
\hline 000 & 000 & 110 & 0 \\
001 & 001 & 111 & 0 \\
010 & 100 & 100 & 1 \\
011 & 101 & 101 & 1 \\
100 & 100 & 010 & 1 \\
101 & 101 & 011 & 1 \\
110 & 000 & 000 & 0 \\
111 & 001 & 001 & 0 \\
\hline
\end{array}
$$

$$
\begin{array}{|r|r|r|r|r|r|r|r|}
\hline x_{i} & y_{i} & z_{i} & x_{i} \oplus y_{i} & x_{i} z_{i} & y_{i} z_{i} & \left(x_{i} \oplus y_{i}\right) z_{i} & x_{i} z_{i} \oplus y_{i} z_{i} \\
\hline 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 1 & 0 & 1 & 1 & 1 \\
1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
1 & 0 & 1 & 1 & 1 & 0 & 1 & 1 \\
1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 1 & 1 & 0 & 0 \\
\hline
\end{array}
$$

$$
\begin{aligned}
(x \oplus y) \cdot z &=\bigoplus_{i=0}^{n}\left(x_{i} \oplus y_{i}\right) z_{i} \\
&=\bigoplus_{i=0}^{n} x_{i} z_{i} \oplus y_{i} z_{i} \\
&=\left(\bigoplus_{i=0}^{n} x_{i} z_{i}\right) \oplus\left(\bigoplus_{i=0}^{n} y_{i} z_{i}\right) \\
&=(x \cdot z) \oplus(y \cdot z)
\end{aligned}
$$


```python

```


```python

o
```


```python

```

$$
\begin{array}{|l|}
\hline\left[\begin{array}{cc}
0 & 1 \\
1 & 0
\end{array}\right] \\
\hline\left[\begin{array}{cc}
0 & -i \\
i & 0
\end{array}\right] \\
\hline\left[\begin{array}{cc}
1 & 0 \\
0 & -1
\end{array}\right] \\
\hline \frac{1}{\sqrt{2}}\left[\begin{array}{cc}
1 & 1 \\
1 & -1
\end{array}\right] \\
\hline\left[\begin{array}{ll}
1 & 0 \\
0 & i
\end{array}\right] \\
\hline\left[\begin{array}{cc}
1 & 0 \\
0 & e^{i \pi / 4}
\end{array}\right] \\
\hline\left[\begin{array}{cc}
\cos \frac{\theta}{2} & -i \sin \frac{\theta}{2} \\
-i \sin \frac{\theta}{2} & \cos \frac{\theta}{2}
\end{array}\right] \\
\hline\left[\begin{array}{cc}
\cos \frac{\theta}{2} & -\sin \frac{\theta}{2} \\
\sin \frac{\theta}{2} & \cos \frac{\theta}{2}
\end{array}\right] \\
\hline\left[\begin{array}{cc}
e^{-i \theta / 2} & 0 \\
0 & e^{i \theta / 2}
\end{array}\right] \\
\hline
\end{array}
$$

$$
|i j\rangle \stackrel{\mathrm{CNOT}}{\longrightarrow}|i(i \mathrm{XOR} j)\rangle
$$

$$
\begin{aligned}
&Q_{W}=\left(q_{x}-\frac{\partial q_{x}}{\partial x} \frac{1}{2} \Delta x\right) \Delta y \Delta z \\
&Q_{E}=\left(q_{x}+\frac{\partial q_{x}}{\partial x} \frac{1}{2} \Delta x\right) \Delta y \Delta z
\end{aligned}
$$

$$
\begin{aligned}
&{\left[\left(q_{x}-\frac{\partial q_{x}}{\partial x} \frac{1}{2} \Delta x\right)-\left(q_{x}+\frac{\partial q_{x}}{\partial x} \frac{1}{2} \Delta x\right)\right] \Delta y \Delta z} \\
&=-\frac{\partial q_{x}}{\partial x} \Delta x \Delta y \Delta z=\frac{\partial}{\partial x}\left(k \frac{\partial T}{\partial x}\right) \Delta x \Delta y \Delta z
\end{aligned}
$$

$$
\begin{aligned}
U &=1+(i A)+\frac{(i A)^{2}}{2 !}+\cdots=\sum_{k} \frac{1}{k !}(i A)^{k}=\sum_{k} \frac{i^{k}}{k !}\left(v \Theta V^{\dagger}\right)^{k} \\
&=\sum_{k} \frac{i^{k}}{k !} V \Theta^{k} V^{\dagger}=V \sum_{k} \frac{i^{k}}{k !} \Theta^{k} V^{\dagger}=V e^{i \theta} V^{\dagger}
\end{aligned}
$$


```python

```


```python

```


```python

```


```python
import pandas as pd
```


```python
df = pd.Series([1,3,4,5,6], index=['a', 'b', 'c', 'd', 'e'])
```


```python
df[df > 3].fillna(0)
```




    c    4
    d    5
    e    6
    dtype: int64




```python

```


```python

```


```python

```


```python
import pandas as pd

df_with_duplicates = pd.DataFrame({
    'Id': [302, 504, 708, 103, 303, 302],
    'Name': ['Watch', 'Camera', 'Phone', 'Shoes', 'Watch', 'Watch'],
    'Cost': ["300", "400", "350", "100", "300", "300"]
})

df_without_duplicates = df_with_duplicates.groupby(level=0).last()

print("DataFrame with duplicates:")
print(df_with_duplicates, "\n")

print("DataFrame without duplicates:")
print(df_without_duplicates, "\n")
```

    DataFrame with duplicates:
        Id    Name Cost
    0  302   Watch  300
    1  504  Camera  400
    2  708   Phone  350
    3  103   Shoes  100
    4  303   Watch  300
    5  302   Watch  300 
    
    DataFrame without duplicates:
        Id    Name Cost
    0  302   Watch  300
    1  504  Camera  400
    2  708   Phone  350
    3  103   Shoes  100
    4  303   Watch  300
    5  302   Watch  300 
    



```python

o
```


```python

```
