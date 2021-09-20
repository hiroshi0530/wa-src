## HHLアルゴリズム

qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。

qiskitのウェブサイト通りに勉強を進めています。

- https://qiskit.org/textbook/ja/ch-applications/hhl_tutorial.html

私の拙いブログでqiskitがRec（推薦システム）のカテゴライズしいたのは、すべてHHLを理解するためでした。現在、推薦システムに興味があり、開発などを行っていますが、そこで重要なのが連立一次方程式を解く事です。連立一次方程式は、数理モデルをコンピュータを利用してとく場合に高い確率で利用されますが、推薦システムもUser-Item行列から如何にしてユーザーエンゲージメントの高い特徴量を抽出出来るかという事が重要になってきます。

よって、量子コンピュータを利用して高速に連立一次方程式を解く事を目標に量子アルゴリズムの復習を開始したわけですが、ようやく目的までたどり着きました。


### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/rec/qiskit/base9/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/rec/qiskit/base9/base_nb.ipynb)

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

import qiskit
import json

import matplotlib.pyplot as plt
import numpy as np
import math

from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from qiskit.visualization import plot_histogram

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


```

## 理論部分の理解

HHLの仮定
- ローディングを実施する効果的なオラクルが存在
- ハミルトニアンシミュレーションと解の関数の計算が可能
- $A$がエルミート行列
- $\mathcal{O}\left(\log (N) s^{2} \kappa^{2} / \epsilon\right)$で計算可能
- 古典アルゴリズムが完全解を返すが、HHLは解となるベクトルを与える関数を近似するだけ


```python

```

$\tilde{\theta}$ は $2^{n} \theta$ に対するバイナリー近似を示しており、 $n$ の添え文字は $n$ デ ィジットに切り捨てられていることを示しています。

$$
\operatorname{QPE}\left(U,|0\rangle_{n}|\psi\rangle_{m}\right)=|\tilde{\theta}\rangle_{n}|\psi\rangle_{m}
$$


```python

```


```python

```
