
## cirq 入門

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/ml/data100/05/05_nb.ipynb)

### 筆者の環境


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
import pandas as pd

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
print('pandas version :', pd.__version__)
```

    matplotlib version : 3.0.3
    scipy version : 1.4.1
    numpy version : 1.19.4
    pandas version : 1.0.3



```python
from cirq import LineQubit, Circuit, Simulator, measure
import cirq
```


```python
# 量子回路初期化
qr = [LineQubit(i) for i in range(2)]
qc = Circuit()

# 量子回路
qc = qc.from_ops(
    # オラクル(|11>を反転)　
    cirq.H(qr[0]),
    cirq.H(qr[1]),
    cirq.CZ(qr[0],qr[1]),
    cirq.H(qr[0]),
    cirq.H(qr[1]),
    
    # 振幅増幅
    cirq.X(qr[0]),
    cirq.X(qr[1]),
    cirq.CZ(qr[0],qr[1]),
    cirq.X(qr[0]),
    cirq.X(qr[1]),
    cirq.H(qr[0]),
    cirq.H(qr[1]),   

    # 測定 
    cirq.measure(qr[0], key='m0'),
    cirq.measure(qr[1], key='m1'),
)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-8-3f05771ff965> in <module>
          4 
          5 # 量子回路
    ----> 6 qc = qc.from_ops(
          7     # オラクル(|11>を反転)
          8     cirq.H(qr[0]),


    AttributeError: 'Circuit' object has no attribute 'from_ops'



```python

```
