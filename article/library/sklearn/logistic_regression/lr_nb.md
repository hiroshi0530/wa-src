
## scikit-learn 公式データセット

scikit-learnは機械学習に必要なデータセットを用意してくれています。ここでは公式サイトにそってサンプルデータの概要を説明します。

### sickit-learn 解説目次

1. 公式データセット
2. データの作成
3. 線形回帰
4. ロジスティック回帰(/article/library/sklearn/logistic_regression/) <= 本節

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sklearn/logistic_regression/lr_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sklearn/logistic_regression/lr_nb.ipynb)

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



```python
import sklearn

sklearn.__version__
```




    '0.20.3'




```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

boston = load_boston()
```


```python
import numpy as np

from
```
