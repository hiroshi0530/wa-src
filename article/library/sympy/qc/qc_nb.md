## sympy で量子演算のシミュレーション

量子計算のシミュレータツールとしてはIBM社のqiskitやGoogleのCirqなどありますが、代表的な数値計算ライブラリであるsympyでも出来るようなので、簡単ですがやってみます。

以下のサイトを参照しました。
- https://docs.sympy.org/latest/index.html
- https://dojo.qulacs.org/ja/latest/notebooks/1.2_qubit_operations.html
- https://qiita.com/openql/items/e5b98bcd13fb4f0b6d59


### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sympy/qc/qc_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sympy/qc/qc_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G95



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


## 量子ビットの表記

sympyそのものと、表記を簡単にするためのrepresent、ブラケット記号で量子ビットを指定することが出来るQubitとQubitBraをimportしておきます。


```python
import sympy
from sympy.physics.quantum import represent
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.qubit import QubitBra
from sympy.physics.quantum.dagger import Dagger
sympy.init_printing()
```


```python
print('sympy version : ', sympy.__version__)
```

    sympy version :  1.5


1量子ビットをブラケット記号を用いて指定します。


```python
# 1量子ビット
q0 = Qubit('0')
q1 = Qubit('1')
p0 = QubitBra('1')
p1 = QubitBra('1')
```


```python
q0
```




$\displaystyle {\left|0\right\rangle }$




```python
q1
```




$\displaystyle {\left|1\right\rangle }$




```python
p0
```




$\displaystyle {\left\langle 1\right|}$




```python
p1
```




$\displaystyle {\left\langle 1\right|}$



representを用いてベクトルで表記できます。


```python
represent(q0)
```




$\displaystyle \left[\begin{matrix}1\\0\end{matrix}\right]$




```python
represent(q0)
```




$\displaystyle \left[\begin{matrix}1\\0\end{matrix}\right]$




```python
represent(p0)
```




$\displaystyle \left[\begin{matrix}0 & 1\end{matrix}\right]$




```python
represent(p0)
```




$\displaystyle \left[\begin{matrix}0 & 1\end{matrix}\right]$



2量子系も同様に可能です。


```python
# 2量子ビット
q00 = Qubit('00')
q01 = Qubit('01')
q10 = Qubit('10')
q11 = Qubit('11')
```


```python
represent(q00)
```




$\displaystyle \left[\begin{matrix}1\\0\\0\\0\end{matrix}\right]$




```python
represent(q01)
```




$\displaystyle \left[\begin{matrix}0\\1\\0\\0\end{matrix}\right]$




```python
represent(q10)
```




$\displaystyle \left[\begin{matrix}0\\0\\1\\0\end{matrix}\right]$




```python
represent(q11)
```




$\displaystyle \left[\begin{matrix}0\\0\\0\\1\end{matrix}\right]$



### 任意の状態


```python
a, b = sympy.symbols('alpha, beta')
psi = a * q0 + b* q1
psi
```




$\displaystyle \alpha {\left|0\right\rangle } + \beta {\left|1\right\rangle }$



エルミート共役を取って内積を計算してみます。


```python
from sympy.physics.quantum.qapply import qapply
qapply(Dagger(psi) * psi)
```




$\displaystyle \alpha \alpha^{\dagger} + \beta \beta^{\dagger}$



## 量子ゲート
まずは1量子ビットに対する演算子からです。
基本的には恒等演算子($I$)、パウリ演算子($X$,$Y$,$Z$)、重ね合わせ状態を作成するアダマール演算子($H$)、位相演算子($ST$,$T$)になります。実際にsympy上でどう定義されているのか見た方がわかりやすいです。


```python
from sympy.physics.quantum.gate import I, X, Y, Z, H, S, T
```


```python
print(type(I))
print(X)
print(Y)
print(Z)
print(H)
print(S)
print(T)
```

    <class 'sympy.core.numbers.ImaginaryUnit'>
    <class 'sympy.physics.quantum.gate.XGate'>
    <class 'sympy.physics.quantum.gate.YGate'>
    <class 'sympy.physics.quantum.gate.ZGate'>
    <class 'sympy.physics.quantum.gate.HadamardGate'>
    <class 'sympy.physics.quantum.gate.PhaseGate'>
    <class 'sympy.physics.quantum.gate.TGate'>



```python
represent(X(0), nqubits=1)
```




$\displaystyle \left[\begin{matrix}0 & 1\\1 & 0\end{matrix}\right]$




```python
represent(X(1), nqubits=2)
```




$\displaystyle \left[\begin{matrix}0 & 0 & 1 & 0\\0 & 0 & 0 & 1\\1 & 0 & 0 & 0\\0 & 1 & 0 & 0\end{matrix}\right]$




```python
represent(Y(0), nqubits=1)
```




$\displaystyle \left[\begin{matrix}0 & - i\\i & 0\end{matrix}\right]$




```python
represent(Z(0), nqubits=1)
```




$\displaystyle \left[\begin{matrix}1 & 0\\0 & -1\end{matrix}\right]$




```python
represent(H(0),nqubits=1)
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



## 1量子ゲートの演算

実際にゲートを作用させてみます。そのためにはqapplyというメソッドを利用します。式を定義してから実際に関数を作用させる形を取ります。$\left| 0\right>$に対してXゲートを作用させます。


```python
from sympy.physics.quantum.qapply import qapply
```


```python
X(0) * q0
```




$\displaystyle X_{0} {\left|0\right\rangle }$




```python
qapply(X(0) * q0)
```




$\displaystyle {\left|1\right\rangle }$



アダマールゲートを利用し、重ね合わせ状態のビットに対して演算を行います。


```python
qapply(H(0)*q0)
```




$\displaystyle \frac{\sqrt{2} {\left|0\right\rangle }}{2} + \frac{\sqrt{2} {\left|1\right\rangle }}{2}$




```python
qapply(Z(0)*H(0)*q0)
```




$\displaystyle \frac{\sqrt{2} {\left|0\right\rangle }}{2} - \frac{\sqrt{2} {\left|1\right\rangle }}{2}$



### 測定
量子コンピュータの最終的な出力結果は測定という行為を行わないといけません。measure_allで全方向（全直交基底）に対する測定を行い、measure_partialで部分的な基底に対する測定を行います。


```python
from sympy.physics.quantum.qubit import measure_all, measure_partial

_ = qapply(Z(0)*H(0)*q0)
```


```python
represent(_)
```




$\displaystyle \left[\begin{matrix}\frac{\sqrt{2}}{2}\\- \frac{\sqrt{2}}{2}\end{matrix}\right]$




```python
measure_all(_)
```




$\displaystyle \left[ \left( {\left|0\right\rangle }, \  \frac{1}{2}\right), \  \left( {\left|1\right\rangle }, \  \frac{1}{2}\right)\right]$




```python
measure_all(q0)
```




$\displaystyle \left[ \left( {\left|0\right\rangle }, \  1\right)\right]$



1量子ビットにmeasure_allすると、２量子ビットが出てきますね。（これは現在不明です）


```python
measure_all(q00)
```




$\displaystyle \left[ \left( {\left|00\right\rangle }, \  1\right)\right]$




```python
measure_partial(q00, (0,))
```




$\displaystyle \left[ \left( {\left|00\right\rangle }, \  1\right)\right]$




```python
measure_partial(q11, (1))
```




$\displaystyle \left[ \left( {\left|11\right\rangle }, \  1\right)\right]$



sympyのdescriptionにある例題を実行して、measure_partialがどうなるか見てみます。おそらく２量子系で意味のある測定が出来るという事でしょうか･･･１量子だとpartialは一つだけですし･･･


```python
qapply(H(0)*H(1)*Qubit('00'))
```




$\displaystyle \frac{{\left|00\right\rangle }}{2} + \frac{{\left|01\right\rangle }}{2} + \frac{{\left|10\right\rangle }}{2} + \frac{{\left|11\right\rangle }}{2}$




```python
measure_partial(qapply(H(0)*H(1)*Qubit('00')), (0,))
```




$\displaystyle \left[ \left( \frac{\sqrt{2} {\left|00\right\rangle }}{2} + \frac{\sqrt{2} {\left|10\right\rangle }}{2}, \  \frac{1}{2}\right), \  \left( \frac{\sqrt{2} {\left|01\right\rangle }}{2} + \frac{\sqrt{2} {\left|11\right\rangle }}{2}, \  \frac{1}{2}\right)\right]$




```python
measure_partial(qapply(H(0)*H(1)*Qubit('00')), (1,))
```




$\displaystyle \left[ \left( \frac{\sqrt{2} {\left|00\right\rangle }}{2} + \frac{\sqrt{2} {\left|01\right\rangle }}{2}, \  \frac{1}{2}\right), \  \left( \frac{\sqrt{2} {\left|10\right\rangle }}{2} + \frac{\sqrt{2} {\left|11\right\rangle }}{2}, \  \frac{1}{2}\right)\right]$



## 2量子系の演算

### CNOT、SWAPゲート

CNOTゲートのsympy上の定義は以下の通り。第一引数が制御ビット、第二引数がターゲットビットです。

```text
This gate performs the NOT or X gate on the target qubit if the control
qubits all have the value 1.

Parameters
----------
label : tuple
    A tuple of the form (control, target).

```

CNOTとSWAPを読み込みます。


```python
from sympy.physics.quantum.gate import CNOT, SWAP
```


```python
represent(CNOT(1,0),nqubits=2)
```




$\displaystyle \left[\begin{matrix}1 & 0 & 0 & 0\\0 & 1 & 0 & 0\\0 & 0 & 0 & 1\\0 & 0 & 1 & 0\end{matrix}\right]$



CNOTをそれぞれの2量子ビットに作用させてみます。


```python
qapply(CNOT(1,0) * q00)
```




$\displaystyle {\left|00\right\rangle }$




```python
qapply(CNOT(1,0) * q01)
```




$\displaystyle {\left|01\right\rangle }$




```python
qapply(CNOT(1,0) * q10)
```




$\displaystyle {\left|11\right\rangle }$




```python
qapply(CNOT(1,0) * q11)
```




$\displaystyle {\left|10\right\rangle }$



SWAPゲートは以下の通りです。引数に交換した量子ビットを指定します。


```python
represent(SWAP(0,1),nqubits=2)
```




$\displaystyle \left[\begin{matrix}1 & 0 & 0 & 0\\0 & 0 & 1 & 0\\0 & 1 & 0 & 0\\0 & 0 & 0 & 1\end{matrix}\right]$



SWAPゲートをそれぞれの2量子ビットに作用させてみます。


```python
qapply(SWAP(0,1) * q00)
```




$\displaystyle {\left|00\right\rangle }$




```python
qapply(SWAP(0,1) * q01)
```




$\displaystyle {\left|10\right\rangle }$




```python
qapply(SWAP(0,1) * q10)
```




$\displaystyle {\left|01\right\rangle }$




```python
qapply(SWAP(0,1) * q11)
```




$\displaystyle {\left|11\right\rangle }$



### テンソル積


```python
a, b, c, d = sympy.symbols('alpha,beta,gamma,delta')
psi = a * q0 + b * q1
phi = c * q0 + d * q1
```


```python
psi
```




$\displaystyle \alpha {\left|0\right\rangle } + \beta {\left|1\right\rangle }$




```python
phi
```




$\displaystyle \delta {\left|1\right\rangle } + \gamma {\left|0\right\rangle }$



テンソル積の計算をするには、TensorProductを利用します。


```python
from sympy.physics.quantum import TensorProduct
TensorProduct(psi, phi)
```




$\displaystyle \left({\alpha {\left|0\right\rangle } + \beta {\left|1\right\rangle }}\right)\otimes \left({\delta {\left|1\right\rangle } + \gamma {\left|0\right\rangle }}\right)$




```python
represent(TensorProduct(psi, phi))
```




$\displaystyle \left[\begin{matrix}\alpha \gamma\\\alpha \delta\\\beta \gamma\\\beta \delta\end{matrix}\right]$



### 測定


```python
measure_all(TensorProduct(psi, phi))
```




$\displaystyle \left[ \left( {\left|00\right\rangle }, \  \frac{\alpha \gamma \overline{\alpha} \overline{\gamma} \overline{\frac{1}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}}}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}\right), \  \left( {\left|01\right\rangle }, \  \frac{\alpha \delta \overline{\alpha} \overline{\delta} \overline{\frac{1}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}}}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}\right), \  \left( {\left|10\right\rangle }, \  \frac{\beta \gamma \overline{\beta} \overline{\gamma} \overline{\frac{1}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}}}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}\right), \  \left( {\left|11\right\rangle }, \  \frac{\beta \delta \overline{\beta} \overline{\delta} \overline{\frac{1}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}}}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}\right)\right]$




```python
measure_partial(TensorProduct(psi, phi), (0,))
```




$\displaystyle \left[ \left( \frac{\alpha \gamma {\left|00\right\rangle }}{\sqrt{\frac{\left|{\alpha \gamma}\right|^{2}}{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}} + \frac{\left|{\beta \gamma}\right|^{2}}{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}} \sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}} + \frac{\beta \gamma {\left|10\right\rangle }}{\sqrt{\frac{\left|{\alpha \gamma}\right|^{2}}{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}} + \frac{\left|{\beta \gamma}\right|^{2}}{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}} \sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}, \  \frac{\alpha \gamma \overline{\alpha} \overline{\gamma} \overline{\frac{1}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}}}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}} + \frac{\beta \gamma \overline{\beta} \overline{\gamma} \overline{\frac{1}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}}}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}\right), \  \left( \frac{\alpha \delta {\left|01\right\rangle }}{\sqrt{\frac{\left|{\alpha \delta}\right|^{2}}{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}} + \frac{\left|{\beta \delta}\right|^{2}}{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}} \sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}} + \frac{\beta \delta {\left|11\right\rangle }}{\sqrt{\frac{\left|{\alpha \delta}\right|^{2}}{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}} + \frac{\left|{\beta \delta}\right|^{2}}{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}} \sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}, \  \frac{\alpha \delta \overline{\alpha} \overline{\delta} \overline{\frac{1}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}}}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}} + \frac{\beta \delta \overline{\beta} \overline{\delta} \overline{\frac{1}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}}}{\sqrt{\left|{\alpha \delta}\right|^{2} + \left|{\alpha \gamma}\right|^{2} + \left|{\beta \delta}\right|^{2} + \left|{\beta \gamma}\right|^{2}}}\right)\right]$



## ベル基底

ベル基底を作ってみます。アダマールゲートとCNOTゲートを組み合わせることで可能です。


```python
qapply(CNOT(0,1) * H(0) * q00)
```




$\displaystyle \frac{\sqrt{2} {\left|00\right\rangle }}{2} + \frac{\sqrt{2} {\left|11\right\rangle }}{2}$




```python
qapply(CNOT(0,1) * H(0) * q01)
```




$\displaystyle \frac{\sqrt{2} {\left|00\right\rangle }}{2} - \frac{\sqrt{2} {\left|11\right\rangle }}{2}$




```python
qapply(CNOT(0,1) * H(0) * q10)
```




$\displaystyle \frac{\sqrt{2} {\left|01\right\rangle }}{2} + \frac{\sqrt{2} {\left|10\right\rangle }}{2}$




```python
qapply(CNOT(0,1) * H(0) * q11)
```




$\displaystyle - \frac{\sqrt{2} {\left|01\right\rangle }}{2} + \frac{\sqrt{2} {\left|10\right\rangle }}{2}$



最後に逆の仮定をたどり、元の状態に戻してみます。


```python
qapply(H(0) * CNOT(0,1) * CNOT(0,1) * H(0) * q00)
```




$\displaystyle {\left|00\right\rangle }$




```python
qapply(H(0) * CNOT(0,1) * CNOT(0,1) * H(0) * q01)
```




$\displaystyle {\left|01\right\rangle }$




```python
qapply(H(0) * CNOT(0,1) * CNOT(0,1) * H(0) * q01)
```




$\displaystyle {\left|01\right\rangle }$




```python
qapply(H(0) * CNOT(0,1) * CNOT(0,1) * H(0) * q11)
```




$\displaystyle {\left|11\right\rangle }$



量子コンピュータのシミュレーションはqiskitなどを使うことが多いですが、sympyでもかなりの事ができることが分かりました。すごいです！
