## [線型代数] 特異値分解と主成分分析

主に推薦システムの理解に必要な線型代数の知識をまとめていこうと思います。


### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/ml/linalg/base/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/ml/linalg/base/base_nb.ipynb)

### 筆者の環境

筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


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


基本的なライブラリとkerasをインポートしそのバージョンを確認しておきます。


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy  version :', scipy.__version__)
print('numpy  version :', np.__version__)
```

    matplotlib version : 3.3.2
    scipy  version : 1.5.2
    numpy  version : 1.19.2


$$
\begin{aligned}
\mathbf{A} \mathbf{v} &=\sigma \mathbf{u} \\
\mathbf{A}^{T} \mathbf{u} &=\sigma \mathbf{v}
\end{aligned}
$$

$$
\begin{aligned}
&\mathbf{A}^{T} \mathbf{A} \mathbf{v}=\sigma \mathbf{A}^{T} \mathbf{u}=\sigma^{2} \mathbf{v} \\
&\mathbf{A} \mathbf{A}^{T} \mathbf{u}=\sigma \mathbf{A} \mathbf{v}=\sigma^{2} \mathbf{u}
\end{aligned}
$$

$u$と$v$は左特異ベクトル、右特異ベクトルと呼ばれ、$u$と$v$は$AA^{T}, A^{T}A$の固有ベクトル。




```python

```
