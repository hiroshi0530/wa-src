## HHLアルゴリズム

qiskitを利用して、量子アルゴリズムについて自分なりに勉強していこうと思います。
個人的な勉強の記録なので、説明などを大幅に省いている可能性があります。

qiskitのウェブサイト通りに勉強を進めています。

- https://qiskit.org/textbook/ja/ch-applications/hhl_tutorial.html

私の拙いブログでqiskitがRec（推薦システム）のカテゴライズしいたのは、すべてHHLを理解するためでした。現在、推薦システムに興味があり、開発などを行っていますが、そこで重要なのが連立一次方程式を解く事です。連立一次方程式は、数理モデルをコンピュータを利用して解く場合に高い確率で利用されますが、推薦システムもUser-Item行列から如何にしてユーザーエンゲージメントの高い特徴量を抽出出来るかという事が重要になってきます。

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



## 共役勾配法

復習の意味を込めて、古典アルゴリズムである共役勾配法の復習をします。
正定値行列である$A$を係数とする連立一次方程式、

$$
A \boldsymbol{x}=\boldsymbol{b}
$$

の解$x$を反復法を用いて数値計算的に解く方法になります。反復法ですので、計算の終了を判定する誤差$(\epsilon)$が必要になります。

$A$,$x$,$b$は以下の様な行列になります。

$$
A = \left(\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 n} \\
a_{21} & a_{22} & \cdots & a_{2 n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n 1} & a_{n 2} & \cdots & a_{n n}
\end{array}\right),\quad x=\left(\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right), \quad b=\left(\begin{array}{c}
b_{1} \\
b_{2} \\
\vdots \\
b_{n}
\end{array}\right)
$$

行列の表式で書くと以下の通りです。

$$
\left(\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 n} \\
a_{21} & a_{22} & \cdots & a_{2 n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n 1} & a_{n 2} & \cdots & a_{n n}
\end{array}\right)\left(\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right)=\left(\begin{array}{c}
b_{1} \\
b_{2} \\
\vdots \\
b_{n}
\end{array}\right)
$$

次に次のように定義される関数$f(x)$を考えます。

$$
f(\boldsymbol{x})=\frac{1}{2}(\boldsymbol{x}, A \boldsymbol{x})-(\boldsymbol{b}, \boldsymbol{x})
$$

$(-,-)$は、ベクトルの内積を計算する演算子です。

$$
(\boldsymbol{x}, \boldsymbol{y})=\boldsymbol{x}^{T} \boldsymbol{y}=\sum_{i=1}^{n} \boldsymbol{x}_{i} \boldsymbol{y}_{i}
$$

成分で表示すると以下の様になります。

$$
f(x)=\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} a_{i j} x_{i} x_{j}-\sum_{i=1}^{n} b_{i} x_{i}
$$

ここで、　$x_k$で微分すると、

$$
\frac{\partial f(x)}{\partial x_{k}}=\frac{1}{2} \sum_{i=1}^{n} a_{i k} x_{i}+\frac{1}{2} \sum_{j=1}^{n} a_{k j} x_{j}-b_{k}
$$

となります。

$A$はエルミート行列なので、

$$
\frac{\partial f(x)}{\partial x_{i}}=\sum_{j=1}^{n} a_{i j} x_{j}-b_{i}=0
$$

となります。

これを一般化すると、

$$
\nabla f(x)=\left(\begin{array}{c}
\frac{\partial f}{\partial x_{1}} \\
\vdots \\
\frac{\partial f}{\partial x_{n}}
\end{array}\right)=A\boldsymbol{x}-b = 0
$$

となり、関数$f(x)$の最小値となる$x$を求める事が、$A\boldsymbol{x}-b = 0$を解く事と同じである事が分かります。

### アルゴリズム

上記の通り、共役勾配法(CG法)は、関数$f(x)$を最小化することに帰着されます。
そのために、ある$x^{(0)}$を出発点に、以下の漸化式に従って最小値とする$x$を求めます。

$$
x^{(k+1)}=x^{(k)}+\alpha_{k} p^{(k)}
$$

ここで、$p^{k}$は解を探索する方向ベクトルです。


$$
f\left(x^{(k)}+\alpha_{k} p^{(k)}\right)=\frac{1}{2} \alpha_{k}{ }^{2}\left(p^{(k)}, A p^{(k)}\right)-\alpha_{k}\left(p^{(k)}, b-A x^{(k)}\right)+f\left(x^{(k)}\right)
$$

$$
\frac{\partial f\left(x^{(k)}+\alpha_{k} p^{(k)}\right)}{\partial \alpha_{k}}=0 \Rightarrow \alpha_{k}=\frac{\left(p^{(k)}, b-A x^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}=\frac{\left(p^{(k)}, r^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}
$$

$$
\begin{aligned}
&r^{(k+1)}=b-A x^{(k+1)} \\
&r^{(k)}=b-A x^{(k)} \\
&r^{(k+1)}-r^{(k)}=A x^{(k+1)}-A x^{(k)}=\alpha_{k} A p^{(k)}
\end{aligned}
$$

$$
\alpha_{k}=\frac{\left(p^{(k)}, r^{(k)}\right)}{\left(p^{(k)}, A p^{(k)}\right)}
$$

$$
\left\|\boldsymbol{r}^{(k+1)}\right\|<\varepsilon
$$

$$
\boldsymbol{p}^{(k+1)}=\boldsymbol{r}^{(k+1)}+\beta^{(k)} \boldsymbol{p}^{(k)}
$$


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

$$
p^{k}=-\nabla f\left(x^{k}\right)=\left(\frac{\partial f\left(x^{k}\right)}{\partial x^{1}}, \ldots, \frac{\partial f\left(x^{k}\right)}{\partial x^{n}}\right)^{T}
$$



### 計算量
$s$は行列$A$の非0要素の割合、$\kappa$は行列$A$の最大固有値と最小固有値の比$\displaystyle \left| \frac{\lambda_{max}}{\lambda_{min}}\right|$、$\epsilon$  は精度です．

$$
O(N s \kappa \log (1 / \varepsilon))
$$

これは$N$に比例する形になっているため、$s\sim \log N$の場合、$N\log N$に比例する事になります。

## HHLアルゴリズム

### HHLの仮定

HHLはある程度の仮定の下に成立するアルゴリズムです。

- ローディングを実施する効果的なオラクルが存在
- ハミルトニアンシミュレーションと解の関数の計算が可能
- $A$がエルミート行列
- $\mathcal{O}\left(\log (N) s^{2} \kappa^{2} / \epsilon\right)$で計算可能
- 古典アルゴリズムが完全解を返すが、HHLは解となるベクトルを与える関数を近似するだけ

### 量子回路へのマッピング

連立一次方程式を量子アルゴリズムで解くには、$Ax=b$を量子回路にマッピングする必要があります。それは、$b$の$i$番目の成分は量子状態$|b\rangle$の$i$番目の基底状態の振幅に対応させるという方法です。また、当然ですが、その際は$\displaystyle \sum_i |b_i|^2 = 1$という規格化が必要です。

$$
Ax=b \rightarrow A|x\rangle = |b\rangle
$$

となります。

### スペクトル分解

$A$はエルミート行列なので、スペクトル分解が可能です。$A$のエルミート性を暗幕的に仮定していましたが、
$$
A'=\left(\begin{array}{ll}
0 & A \\
A & 0
\end{array}\right)
$$

とすれば、$A'$はエルミート行列となるため、問題ありません。よって、$A$は固有ベクトル$|u_i\rangle$とその固有値$\lambda_i$を利用して、以下の様に展開できます。

$$
A=\sum_{j=0}^{N-1} \lambda_{j}\left|u_{j}\right\rangle\left\langle u_{j}\right|
$$

よって、逆行列は以下の様になります。

$$
A^{-1}=\sum_{j=0}^{N-1} \lambda_{j}^{-1}\left|u_{j}\right\rangle\left\langle u_{j}\right|
$$

$u_i$は$A$の固有ベクトルなので、$|b\rangle$はその重ね合わせで表現できます。おそらくこれが量子コンピュータを利用する強い動機になっていると思います。

$$
|b\rangle=\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle
$$

本来であれば、$A$の固有ベクトルを計算できなければこの形で$|b\rangle$を用意することが出来ませんが、量子コンピュータでは$|b\rangle$を読み込むことで、自動的にこの状態を用意することが出来ます。

最終的には以下の式の形を量子コンピュータを利用して求める事になります。

$$
|x\rangle=A^{-1}|b\rangle=\sum_{j=0}^{N-1} \lambda_{j}^{-1} b_{j}\left|u_{j}\right\rangle
$$

### 1. データのロード

対象となる$|b\rangle$の各データの振幅を量子ビット$n_b$にロードします。

$$
|0\rangle_{n_{b}} \mapsto|b\rangle_{n_{b}}
$$

### 2. QPEの適用

量子位相推定を利用して、ユニタリ演算子$U=e^{i A t}$ の$|b\rangle$の位相を推定します。

$|b\rangle$は上述の様に以下になります。

$$
|b\rangle=\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle
$$

$U$は展開して整理すると、以下の様になります。

$$
\begin{aligned}
&U=e^{i A t}=\sum_{k=0}^{\infty} \frac{\left(i A t\right)^{k}}{k !}\\
&=\sum_{k=0}^{\infty} \frac{\left(i t\right)^{k}}{k !}\left(\sum_{j=0}^{N-1} \lambda_{j}|u_{j}\rangle\langle u_{j} |\right)^{k}\\
&=\sum_{k=0}^{\infty} \frac{(i t)^{k}}{k !} \sum_{j=0}^{N-1} \lambda_{j}^{k}\left|u_{j}\right\rangle \langle u_{j} |\\
&=\sum_{j=0}^{N-1}\left(\sum_{k=0}^{\infty} \frac{\left(i t\right)^{k}}{k_{i}} \lambda_{j}^{k}\right)|u_{j}\rangle \langle u_{j}| \\
&=\sum_{r=0}^{N-1} e^{i \lambda_{j} t}\left|u_{j}\right\rangle\left\langle u_{j}\right|
\end{aligned}
$$

$U$を$|b\rangle$に作用させると、

$$
\begin{aligned}
U|b\rangle &=U\left(\sum_{j=0}^{N-1} b_{j}\left|u_{j}\right\rangle\right) \\
&=\sum_{j^{\prime}=0}^{N-1} e^{i \lambda j^{\prime} t}\left|h_{j^{\prime}}\right\rangle\left\langle h_{j^{\prime}}\right| \cdot\left(\sum_{j=0}^{N-1} b_{j}\left|h_{j}\right\rangle\right) \\
&=\sum_{j=0}^{N-1} b_{j} e^{i \lambda_{j} t}\left|u_{j}\right\rangle
\end{aligned}
$$

となり、量子位相推定を利用して、$\lambda_j$の量子状態$|\tilde{\lambda}_j\rangle_{n_l}$を求める事が出来ます。

$\tilde{\lambda_{j}}$は、$\displaystyle 2^{n_{l}} \frac{\lambda_{j} t}{2 \pi}$に対する$n_l$-bitバイナリ近似となります。

$t=2\pi$とし、$\lambda_l$が、$n_l$ビットで正確に表現できるとすると、量子位相推定は以下の様に表現できます。

$$
\operatorname{QPE}\left(e^{i A 2 \pi}, \sum_{j=0}^{N-1} b_{j}|0\rangle_{n_{l}}\left|u_{j}\right\rangle_{n_{b}}\right)=\sum_{j=0}^{N-1} b_{j}\left|\lambda_{j}\right\rangle_{n_{l}}\left|u_{j}\right\rangle_{n_{b}}
$$


```python

```

$b(\cdots)$は二進数表記である事を示し、$b_k(\cdots)$は、二進数表記の$k$ビット目の値を表します。まとめると、以下の様になります。

$$
\begin{aligned}
&b\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)=d_{0} d_{1} d_2 \cdot \cdots d_{m-1} \\
&b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)=d_k \\
&\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)=\frac{d_{0}}{2}+\frac{d_{1}}{4}+\frac{d_{2}}{8} \cdots \frac{d_{m-1}}{2^{m}} \quad \left(0 \leqq \frac{2}{\pi} \cos ^{-1}\left(v_{i}\right) \leqq 1\right)
\end{aligned}
$$

これから、


$$
\begin{aligned}
\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)&=\frac{1}{2} b_{0}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{1}\right)\right)+\frac{1}{4} b_{1}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right)+ \cdots \\
&=\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{n}\right)\right) 2^{-k-1} \cdots (1)
\end{aligned}
$$

$\displaystyle b_{k}\left(\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)\right)$をターゲットビットとして、$R_{y}\left(2^{-k-1}\pi\right)$である制御回転ゲートをかけることを考えます。

$$
\begin{aligned}
&\prod_{k=0}^{m-1} R_{y}\left(b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle \\
&=R_{y}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle \\
\end{aligned}
$$

回転ゲートは、以下の様にになります。

$$
R_{y}(\theta)|0\rangle=\cos \frac{\theta}{2}|0\rangle+\sin \frac{\theta}{2}|1\rangle
$$

この式と(1)を利用して、

$$
\begin{aligned}
&\cos\frac{1}{2}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right) \\
&=\cos\left(\frac{1}{2}\times\frac{2}{\pi} \cos ^{-1}\left(v_{i}\right)\times \pi\right) = v_i
\end{aligned}
$$

となり、こちらを利用して、

$$
R_{y}\left(\sum_{k=0}^{m-1} b_{k}\left(\frac{1}{\pi} \cos ^{-1}\left(v_{i}\right)\right) 2^{-k-1}\pi\right)|0\rangle=v_{i}|0\rangle+\sqrt{1-v_{i}^{2}}|1\rangle
$$

を得る。ここで、$\displaystyle v_i = \frac{1}{\lambda_i}$とすることで、補助ビットを、

$$
\sqrt{1-\frac{C^{2}}{\lambda_{j}^{2}}}|0\rangle+\frac{C}{\lambda_{j}}|1\rangle
$$

と計算することが出来る。




```python

```

$$
\operatorname{QPE}\left(U,|0\rangle_{n}|\psi\rangle_{m}\right)=|\tilde{\theta}\rangle_{n}|\psi\rangle_{m}
$$


```python

```

## 計算量の比較

## 量子


$$
O(s \kappa \operatorname{poly} \log (s \kappa / \varepsilon)))
$$

行列 $A$ がスパース $(s \sim O(\operatorname{poly} \log N))$なら、

$$
O(s \kappa \operatorname{poly} \log (s \kappa / \varepsilon))) \sim O(s \kappa \operatorname{poly} \log N \operatorname{poly} \log (s \kappa / \varepsilon))
$$


### 古典
$$
O(N s \kappa \log (1 / \varepsilon))
$$

## qiskitで実装



```python
from qiskit import Aer
from qiskit.circuit.library import QFT
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms import HHL, NumPyLSsolver
from qiskit.aqua.components.eigs import EigsQPE
from qiskit.aqua.components.reciprocals import LookupRotation
from qiskit.aqua.operators import MatrixOperator
from qiskit.aqua.components.initial_states import Custom
import numpy as np
```


```python
def create_eigs(matrix, num_ancillae, num_time_slices, negative_evals):
    ne_qfts = [None, None]
    if negative_evals:
        num_ancillae += 1
        ne_qfts = [QFT(num_ancillae - 1), QFT(num_ancillae - 1).inverse()]

    return EigsQPE(MatrixOperator(matrix=matrix),
                   QFT(num_ancillae).inverse(),
                   num_time_slices=num_time_slices,
                   num_ancillae=num_ancillae,
                   expansion_mode='suzuki',
                   expansion_order=2,
                   evo_time=None, # np.pi*3/4, #None,  # This is t, can set to: np.pi*3/4
                   negative_evals=negative_evals,
                   ne_qfts=ne_qfts)
```


```python
def fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    fidelity = state_fidelity(solution_hhl_normed, solution_ref_normed)
    print("Fidelity:\t\t %f" % fidelity)
```


```python
matrix = [[1, -1/3], [-1/3, 1]]
vector = [1, 0]
```


```python
orig_size = len(vector)
matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

# Initialize eigenvalue finding module
eigs = create_eigs(matrix, 3, 100, False)
num_q, num_a = eigs.get_register_sizes()

# Initialize initial state module
init_state = Custom(num_q, state_vector=vector)
# init_state = QuantumCircuit(num_q).initialize(vector/np.linalg.norm(vector))

# Initialize reciprocal rotation module
reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
           init_state, reci, num_q, num_a, orig_size)
```

    /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages/qiskit/aqua/components/initial_states/custom.py:79: DeprecationWarning: The Custom class is deprecated as of Aqua 0.9 and will be removed no earlier than 3 months after the release date. Instead, all algorithms and circuits accept a plain QuantumCircuit. Custom(state_vector=vector) is the same as a circuit where the ``initialize(vector/np.linalg.norm(vector))`` method has been called.
      super().__init__()



```python
result = algo.run(QuantumInstance(Aer.get_backend('statevector_simulator')))
print("Solution:\t\t", np.round(result['solution'], 5))

result_ref = NumPyLSsolver(matrix, vector).run()
print("Classical Solution:\t", np.round(result_ref['solution'], 5))

print("Probability:\t\t %f" % result['probability_result'])
fidelity(result['solution'], result_ref['solution'])
```

    /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages/qiskit/aqua/components/initial_states/custom.py:151: DeprecationWarning: The StateVectorCircuit class is deprecated as of Qiskit Aqua 0.9.0 and will be removed no earlier than 3 months after the release. If you need to initialize a circuit, use the QuantumCircuit.initialize or QuantumCircuit.isometry methods. For a parameterized initialization, try the qiskit.ml.circuit.library.RawFeatureVector class.
      svc = StateVectorCircuit(self._state_vector)


    Solution:		 [ 0.66576-0.j -0.38561+0.j]
    Classical Solution:	 [1.125 0.375]
    Probability:		 0.211527
    Fidelity:		 0.438807



```python
print("circuit_width:\t", result['circuit_info']['width'])
print("circuit_depth:\t", result['circuit_info']['depth'])
print("CNOT gates:\t", result['circuit_info']['operations']['cx'])
```

    circuit_width:	 7
    circuit_depth:	 101
    CNOT gates:	 54



```python

```


```python

```


```python

```


```python
from qiskit import aqua
aqua.__version__
```




    '0.9.1'




```python

```


```python
!pip install --upgrade qiskit

```

    Requirement already satisfied: qiskit in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (0.26.2)
    Collecting qiskit
      Downloading qiskit-0.30.0.tar.gz (12 kB)
    Collecting qiskit-terra==0.18.2
      Downloading qiskit_terra-0.18.2-cp38-cp38-macosx_10_9_x86_64.whl (5.3 MB)
    [K     |████████████████████████████████| 5.3 MB 4.8 MB/s eta 0:00:01
    [?25hCollecting qiskit-aer==0.9.0
      Downloading qiskit_aer-0.9.0-cp38-cp38-macosx_10_9_x86_64.whl (8.6 MB)
    [K     |████████████████████████████████| 8.6 MB 5.1 MB/s eta 0:00:01
    [?25hCollecting qiskit-ibmq-provider==0.16.0
      Downloading qiskit_ibmq_provider-0.16.0-py3-none-any.whl (235 kB)
    [K     |████████████████████████████████| 235 kB 39.1 MB/s eta 0:00:01
    [?25hRequirement already satisfied: qiskit-ignis==0.6.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit) (0.6.0)
    Collecting qiskit-aqua==0.9.5
      Downloading qiskit_aqua-0.9.5-py3-none-any.whl (2.1 MB)
    [K     |████████████████████████████████| 2.1 MB 30.0 MB/s eta 0:00:01
    [?25hRequirement already satisfied: pybind11>=2.6 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aer==0.9.0->qiskit) (2.6.2)
    Requirement already satisfied: numpy>=1.16.3 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aer==0.9.0->qiskit) (1.19.2)
    Requirement already satisfied: scipy>=1.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aer==0.9.0->qiskit) (1.5.2)
    Requirement already satisfied: quandl in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (3.6.0)
    Requirement already satisfied: sympy>=1.3 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (1.6.2)
    Requirement already satisfied: fastdtw<=0.3.4 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (0.3.4)
    Collecting docplex>=2.21.207
      Downloading docplex-2.22.213.tar.gz (634 kB)
    [K     |████████████████████████████████| 634 kB 37.7 MB/s eta 0:00:01
    [?25hRequirement already satisfied: pandas in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (1.1.3)
    Collecting yfinance>=0.1.62
      Downloading yfinance-0.1.63.tar.gz (26 kB)
    Requirement already satisfied: psutil>=5 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (5.7.2)
    Requirement already satisfied: scikit-learn>=0.20.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (0.23.2)
    Requirement already satisfied: setuptools>=40.1.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (54.2.0)
    Requirement already satisfied: retworkx>=0.8.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (0.8.0)
    Requirement already satisfied: dlx<=1.0.4 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (1.0.4)
    Requirement already satisfied: h5py<3.3.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-aqua==0.9.5->qiskit) (2.10.0)
    Requirement already satisfied: requests>=2.19 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-ibmq-provider==0.16.0->qiskit) (2.24.0)
    Requirement already satisfied: python-dateutil>=2.8.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-ibmq-provider==0.16.0->qiskit) (2.8.1)
    Collecting websocket-client>=1.0.1
      Using cached websocket_client-1.2.1-py2.py3-none-any.whl (52 kB)
    Requirement already satisfied: requests-ntlm>=1.1.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-ibmq-provider==0.16.0->qiskit) (1.1.0)
    Requirement already satisfied: urllib3>=1.21.1 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-ibmq-provider==0.16.0->qiskit) (1.25.11)
    Requirement already satisfied: ply>=3.10 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra==0.18.2->qiskit) (3.11)
    Collecting tweedledum<2.0,>=1.1
      Downloading tweedledum-1.1.1-cp38-cp38-macosx_10_9_x86_64.whl (1.8 MB)
    [K     |████████████████████████████████| 1.8 MB 45.5 MB/s eta 0:00:01
    [?25hCollecting retworkx>=0.8.0
      Downloading retworkx-0.10.2-cp38-cp38-macosx_10_9_x86_64.whl (1.1 MB)
    [K     |████████████████████████████████| 1.1 MB 17.1 MB/s eta 0:00:01
    [?25hCollecting symengine>0.7
      Downloading symengine-0.8.1-cp38-cp38-macosx_10_9_x86_64.whl (17.7 MB)
    [K     |████████████████████████████████| 17.7 MB 31.2 MB/s eta 0:00:01
    [?25hRequirement already satisfied: fastjsonschema>=2.10 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra==0.18.2->qiskit) (2.15.1)
    Requirement already satisfied: python-constraint>=1.4 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra==0.18.2->qiskit) (1.4.0)
    Requirement already satisfied: jsonschema>=2.6 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra==0.18.2->qiskit) (3.2.0)
    Requirement already satisfied: dill>=0.3 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra==0.18.2->qiskit) (0.3.3)
    Requirement already satisfied: six in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from docplex>=2.21.207->qiskit-aqua==0.9.5->qiskit) (1.15.0)
    Requirement already satisfied: pyrsistent>=0.14.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from jsonschema>=2.6->qiskit-terra==0.18.2->qiskit) (0.17.3)
    Requirement already satisfied: attrs>=17.4.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from jsonschema>=2.6->qiskit-terra==0.18.2->qiskit) (20.3.0)
    Requirement already satisfied: idna<3,>=2.5 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from requests>=2.19->qiskit-ibmq-provider==0.16.0->qiskit) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from requests>=2.19->qiskit-ibmq-provider==0.16.0->qiskit) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from requests>=2.19->qiskit-ibmq-provider==0.16.0->qiskit) (2020.6.20)
    Requirement already satisfied: ntlm-auth>=1.0.2 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from requests-ntlm>=1.1.0->qiskit-ibmq-provider==0.16.0->qiskit) (1.5.0)
    Requirement already satisfied: cryptography>=1.3 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from requests-ntlm>=1.1.0->qiskit-ibmq-provider==0.16.0->qiskit) (3.1.1)
    Requirement already satisfied: cffi!=1.11.3,>=1.8 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from cryptography>=1.3->requests-ntlm>=1.1.0->qiskit-ibmq-provider==0.16.0->qiskit) (1.14.3)
    Requirement already satisfied: pycparser in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from cffi!=1.11.3,>=1.8->cryptography>=1.3->requests-ntlm>=1.1.0->qiskit-ibmq-provider==0.16.0->qiskit) (2.20)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from scikit-learn>=0.20.0->qiskit-aqua==0.9.5->qiskit) (2.1.0)
    Requirement already satisfied: joblib>=0.11 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from scikit-learn>=0.20.0->qiskit-aqua==0.9.5->qiskit) (0.17.0)
    Requirement already satisfied: mpmath>=0.19 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from sympy>=1.3->qiskit-aqua==0.9.5->qiskit) (1.1.0)
    Requirement already satisfied: multitasking>=0.0.7 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from yfinance>=0.1.62->qiskit-aqua==0.9.5->qiskit) (0.0.9)
    Requirement already satisfied: lxml>=4.5.1 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from yfinance>=0.1.62->qiskit-aqua==0.9.5->qiskit) (4.6.1)
    Requirement already satisfied: pytz>=2017.2 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from pandas->qiskit-aqua==0.9.5->qiskit) (2020.1)
    Requirement already satisfied: inflection>=0.3.1 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from quandl->qiskit-aqua==0.9.5->qiskit) (0.5.1)
    Requirement already satisfied: more-itertools in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from quandl->qiskit-aqua==0.9.5->qiskit) (8.6.0)
    Building wheels for collected packages: qiskit, docplex, yfinance
      Building wheel for qiskit (setup.py) ... [?25ldone
    [?25h  Created wheel for qiskit: filename=qiskit-0.30.0-py3-none-any.whl size=11258 sha256=0f32db3ed88a5e775405ba64e7024a1d45d915212033fa32d74ed3bf41ddf62d
      Stored in directory: /Users/hiroshi.wayama/Library/Caches/pip/wheels/01/f8/3f/4edf7a8f1659f7ca83c463df424e107dbceeffe310b6ecc58d
      Building wheel for docplex (setup.py) ... [?25ldone
    [?25h  Created wheel for docplex: filename=docplex-2.22.213-py3-none-any.whl size=696851 sha256=78aca4ab85708f8d2830f352a32b9f4ca193de86ae8ea34348e623ad1d0ce76e
      Stored in directory: /Users/hiroshi.wayama/Library/Caches/pip/wheels/35/3e/11/e31bf877e1965b75dc2f3de4ec2d5c9d1680c6f803ef76ed9f
      Building wheel for yfinance (setup.py) ... [?25ldone
    [?25h  Created wheel for yfinance: filename=yfinance-0.1.63-py2.py3-none-any.whl size=23907 sha256=efa4e5a74644065ab23d18dd96f97d77730e2661513045c386368f21888b10ad
      Stored in directory: /Users/hiroshi.wayama/Library/Caches/pip/wheels/ec/cc/c1/32da8ee853d742d5d7cbd11ee04421222eb354672020b57297
    Successfully built qiskit docplex yfinance
    Installing collected packages: tweedledum, symengine, retworkx, qiskit-terra, yfinance, websocket-client, docplex, qiskit-ibmq-provider, qiskit-aqua, qiskit-aer, qiskit
      Attempting uninstall: retworkx
        Found existing installation: retworkx 0.8.0
        Uninstalling retworkx-0.8.0:
          Successfully uninstalled retworkx-0.8.0
      Attempting uninstall: qiskit-terra
        Found existing installation: qiskit-terra 0.17.4
        Uninstalling qiskit-terra-0.17.4:
          Successfully uninstalled qiskit-terra-0.17.4
      Attempting uninstall: yfinance
        Found existing installation: yfinance 0.1.55
        Uninstalling yfinance-0.1.55:
          Successfully uninstalled yfinance-0.1.55
      Attempting uninstall: docplex
        Found existing installation: docplex 2.15.194
        Uninstalling docplex-2.15.194:
          Successfully uninstalled docplex-2.15.194
      Attempting uninstall: qiskit-ibmq-provider
        Found existing installation: qiskit-ibmq-provider 0.13.1
        Uninstalling qiskit-ibmq-provider-0.13.1:
          Successfully uninstalled qiskit-ibmq-provider-0.13.1
      Attempting uninstall: qiskit-aqua
        Found existing installation: qiskit-aqua 0.9.1
        Uninstalling qiskit-aqua-0.9.1:
          Successfully uninstalled qiskit-aqua-0.9.1
      Attempting uninstall: qiskit-aer
        Found existing installation: qiskit-aer 0.8.2
        Uninstalling qiskit-aer-0.8.2:
          Successfully uninstalled qiskit-aer-0.8.2
      Attempting uninstall: qiskit
        Found existing installation: qiskit 0.26.2
        Uninstalling qiskit-0.26.2:
          Successfully uninstalled qiskit-0.26.2
    Successfully installed docplex-2.22.213 qiskit-0.30.0 qiskit-aer-0.9.0 qiskit-aqua-0.9.5 qiskit-ibmq-provider-0.16.0 qiskit-terra-0.18.2 retworkx-0.10.2 symengine-0.8.1 tweedledum-1.1.1 websocket-client-1.2.1 yfinance-0.1.63



```python
!pip install qiskit-terra
```

    Requirement already satisfied: qiskit-terra in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (0.17.4)
    Requirement already satisfied: scipy>=1.4 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (1.5.2)
    Requirement already satisfied: sympy>=1.3 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (1.6.2)
    Requirement already satisfied: fastjsonschema>=2.10 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (2.15.1)
    Requirement already satisfied: ply>=3.10 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (3.11)
    Requirement already satisfied: dill>=0.3 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (0.3.3)
    Requirement already satisfied: python-dateutil>=2.8.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (2.8.1)
    Requirement already satisfied: psutil>=5 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (5.7.2)
    Requirement already satisfied: numpy>=1.17 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (1.19.2)
    Requirement already satisfied: jsonschema>=2.6 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (3.2.0)
    Requirement already satisfied: python-constraint>=1.4 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (1.4.0)
    Requirement already satisfied: retworkx>=0.8.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from qiskit-terra) (0.8.0)
    Requirement already satisfied: six>=1.11.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from jsonschema>=2.6->qiskit-terra) (1.15.0)
    Requirement already satisfied: setuptools in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from jsonschema>=2.6->qiskit-terra) (54.2.0)
    Requirement already satisfied: attrs>=17.4.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from jsonschema>=2.6->qiskit-terra) (20.3.0)
    Requirement already satisfied: pyrsistent>=0.14.0 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from jsonschema>=2.6->qiskit-terra) (0.17.3)
    Requirement already satisfied: mpmath>=0.19 in /Users/hiroshi.wayama/anaconda3/lib/python3.8/site-packages (from sympy>=1.3->qiskit-terra) (1.1.0)



```python

```


```python
Custom?

init signature:
Custom(
    num_qubits: int,
    state: str = 'zero',
    state_vector: Union[numpy.ndarray, qiskit.aqua.operators.state_fns.state_fn.StateFn, NoneType] = None,
    circuit: Union[qiskit.circuit.quantumcircuit.QuantumCircuit, NoneType] = None,
) -> None
```


```python
QuantumCircuit.initialize?

Args:
    params (str or list or int):
        * str: labels of basis states of the Pauli eigenstates Z, X, Y. See
            :meth:`~qiskit.quantum_info.states.statevector.Statevector.from_label`.
            Notice the order of the labels is reversed with respect to the qubit index to
            be applied to. Example label '01' initializes the qubit zero to `|1>` and the
            qubit one to `|0>`.
        * list: vector of complex amplitudes to initialize to.
        * int: an integer that is used as a bitmap indicating which qubits to initialize
           to `|1>`. Example: setting params to 5 would initialize qubit 0 and qubit 2
           to `|1>` and qubit 1 to `|0>`.
    qubits (QuantumRegister or int):
        * QuantumRegister: A list of qubits to be initialized [Default: None].
        * int: Index of qubit to initialized [Default: None].

Returns:
    qiskit.circuit.Instruction: a handle to the instruction that was just initialized
```


```python

```


```python

```


```python
import math

qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
# Add Controlled-T
qc.cp(math.pi/4, 0, 1)
display(qc.draw())

```


<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌───┐         
q_0: ┤ H ├─■───────
     ├───┤ │P(π/4) 
q_1: ┤ X ├─■───────
     └───┘         </pre>



```python
from qiskit import QuantumCircuit, Aer, assemble
from math import pi
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex

qc.save_unitary()
usim = Aer.get_backend('aer_simulator')
qobj = assemble(qc)
unitary = usim.run(qobj).result().get_unitary()
array_to_latex(unitary, prefix="\\text{Circuit = }\n")
```




$$
\text{Circuit = }

\begin{bmatrix}
0 & 0 & \tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}}  \\
 0 & 0 & \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}}  \\
 \tfrac{1}{\sqrt{2}} & \tfrac{1}{\sqrt{2}} & 0 & 0  \\
 \tfrac{1}{2}(1 + i) & \tfrac{1}{2}(-1 - i) & 0 & 0  \\
 \end{bmatrix}
$$




```python

```


```python

```


```python

```

## 参考文献
- https://www2.yukawa.kyoto-u.ac.  jp/~qischool2019/mitaraiCTO.pdf
