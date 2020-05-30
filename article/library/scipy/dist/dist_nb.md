
## scipy tips
aaaaa-

### scipy 目次

1. [公式データセット](/article/library/sklearn/datasets/) <= 本節
2. [データの作成](/article/library/sklearn/makedatas/)

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa/blob/master/src/sklearn/datasets/ds_nb.ipynb)


### 筆者の環境


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
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib
import matplotlib.pyplot as plt
import scipy

matplotlib.__version__
scipy.__version__
```




    '1.4.1'



## 正規分布


```python
from scipy.stats import norm

x = norm.rvs(size=1000)
```


```python
plt.grid()
plt.hist(x, bins=20)
```




    (array([  6.,   8.,  16.,  25.,  46.,  52.,  86.,  87., 116., 131., 108.,
            103.,  77.,  62.,  32.,  24.,   6.,  11.,   2.,   2.]),
     array([-2.89146351, -2.5827596 , -2.2740557 , -1.96535179, -1.65664788,
            -1.34794397, -1.03924006, -0.73053615, -0.42183225, -0.11312834,
             0.19557557,  0.50427948,  0.81298339,  1.1216873 ,  1.4303912 ,
             1.73909511,  2.04779902,  2.35650293,  2.66520684,  2.97391075,
             3.28261465]),
     <a list of 20 Patch objects>)




![svg](dist_nb_files/dist_nb_6_1.svg)



```python

```
