## [線型代数]

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

import pystan

print('matplotlib version :', matplotlib.__version__)
print('scipy  version :', scipy.__version__)
print('numpy  version :', np.__version__)
print('pystan version :', pystan.__version__)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-3-debf317cfac8> in <module>
          8 import pandas as pd
          9 
    ---> 10 import pystan
         11 
         12 print('matplotlib version :', matplotlib.__version__)


    ModuleNotFoundError: No module named 'pystan'



```python

```
