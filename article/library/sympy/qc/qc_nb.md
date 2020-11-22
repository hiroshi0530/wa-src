## sympy で量子演算のシミュレーション



### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sympy/base/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sympy/base/base_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


```python
!sw_vers
```


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

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
```

    matplotlib version : 3.3.2
    scipy version : 1.5.2
    numpy version : 1.18.5


## 基本的な確率分布

sympyは3つの数値型



```python
from sympy import *
from sympy.physics.quantum import *
from sympy.physics.quantum.qubit import Qubit,QubitBra
from sympy.physics.quantum.gate import X,Y,Z,H,S,T,CNOT,SWAP, CPHASE
init_printing()

```


```python
import sympy
print('sympy version : ', sympy.__version__)
```

    sympy version :  1.5



```python
_0 = Qubit('0')
_1 = qapply(X(0)*_0)
print(represent(_0))
```

    Matrix([[1], [0]])
    Matrix([[0], [1]])



```python
represent(_1)
```




$\displaystyle \left[\begin{matrix}0\\1\end{matrix}\right]$




```python
Qubit('0')
```


```python
qapply(H(0)*Qubit('00'))
```




$\displaystyle \frac{\sqrt{2} {\left|00\right\rangle }}{2} + \frac{\sqrt{2} {\left|01\right\rangle }}{2}$




```python
represent?
```


```python
Qubit('1')
```




$\displaystyle {\left|1\right\rangle }$




```python
Qubit('00')
```




$\displaystyle {\left|00\right\rangle }$




```python
QubitBra('00')
```




$\displaystyle {\left\langle 00\right|}$




```python
a1 = QubitBra(0) * Qubit('0')
print(a1)
```

    <0|0>



```python
represent(psi)
```




$\displaystyle \left[\begin{matrix}1\\0\end{matrix}\right]$




```python
from sympy import *
from sympy.physics.quantum import *
from sympy.physics.quantum.qubit import Qubit,QubitBra
init_printing() # ベクトルや行列を綺麗に表示するため
psi = Qubit('0')
psi
represent(psi)
```




$\displaystyle \left[\begin{matrix}1\\0\end{matrix}\right]$




```python
psi = Qubit('0')
psi
```




$\displaystyle {\left|0\right\rangle }$




```python
represent(psi)
```




$\displaystyle \left[\begin{matrix}1\\0\end{matrix}\right]$




```python
a, b = symbols('alpha, beta')  #a, bをシンボルとして、alpha, betaとして表示
ket0 = Qubit('0')
ket1 = Qubit('1')
psi = a * ket0 + b* ket1
psi # 状態をそのまま書くとケットで表示してくれる
```




$\displaystyle \alpha {\left|0\right\rangle } + \beta {\left|1\right\rangle }$




```python

```


```python
X(0)
```




$\displaystyle X_{0}$



<div>


```python
represent(X(0), nqubits=1)
```




$\displaystyle \left[\begin{matrix}0 & 1\\1 & 0\end{matrix}\right]$



</div>


```python
represent(X(0), nqubits=2)
```




$\displaystyle \left[\begin{matrix}0 & 1 & 0 & 0\\1 & 0 & 0 & 0\\0 & 0 & 0 & 1\\0 & 0 & 1 & 0\end{matrix}\right]$




```python
represent(H(0),nqubits=1)
simplify(represent(H(0),nqubits=1))
```




$\displaystyle \left[\begin{matrix}\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\\frac{1}{\sqrt{2}} & - \frac{\sqrt{2}}{2}\end{matrix}\right]$




```python
represent(S(0),nqubits=1)
```




$\displaystyle \left[\begin{matrix}1 & 0\\0 & i\end{matrix}\right]$




```python
represent(T(0),nqubits=1)
```




$\displaystyle \left[\begin{matrix}1 & 0\\0 & e^{\frac{i \pi}{4}}\end{matrix}\right]$




```python
ket0 = Qubit('0')
S(0)*Y(0)*X(0)*H(0)*ket0
```




$\displaystyle S_{0} Y_{0} X_{0} H_{0} {\left|0\right\rangle }$




```python
qapply(S(0)*Y(0)*X(0)*H(0)*ket0)

```




$\displaystyle - \frac{\sqrt{2} i {\left|0\right\rangle }}{2} - \frac{\sqrt{2} {\left|1\right\rangle }}{2}$




```python
a,b,c,d = symbols('alpha,beta,gamma,delta')
psi = a*Qubit('0')+b*Qubit('1')
phi = c*Qubit('0')+d*Qubit('1')
```


```python
TensorProduct(psi, phi) #テンソル積

```




$\displaystyle \left({\alpha {\left|0\right\rangle } + \beta {\left|1\right\rangle }}\right)\otimes \left({\delta {\left|1\right\rangle } + \gamma {\left|0\right\rangle }}\right)$




```python
represent(TensorProduct(psi, phi))
```




$\displaystyle \left[\begin{matrix}\alpha \gamma\\\alpha \delta\\\beta \gamma\\\beta \delta\end{matrix}\right]$




```python
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.gate import H, CNOT
from sympy.physics.quantum.qapply import qapply
bell = {}
# to Bell
for yx in ['00', '10', '01', '11']:
  result = qapply(CNOT(0,1)*H(0)*Qubit(yx))
  bell[yx] = result
  print (f'{yx} -> ', result)
# from Bell
for i, state in bell.items():
  result = qapply(H(0)*CNOT(0,1)*state)
  print(f'beta{i} -> ', result)
```

    00 ->  sqrt(2)*|00>/2 + sqrt(2)*|11>/2
    10 ->  sqrt(2)*|01>/2 + sqrt(2)*|10>/2
    01 ->  sqrt(2)*|00>/2 - sqrt(2)*|11>/2
    11 ->  -sqrt(2)*|01>/2 + sqrt(2)*|10>/2
    beta00 ->  |00>
    beta10 ->  |10>
    beta01 ->  |01>
    beta11 ->  |11>



```python

```
