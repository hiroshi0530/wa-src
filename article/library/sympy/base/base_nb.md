## sympy

これまで数値計算はすべてmathematicaを用いてやってきましたが、webシステム開発者としてpythonに移行しようと思います。

まずは最初の基本として、

- http://www.turbare.net/transl/scipy-lecture-notes/packages/sympy.html

に記載された問題を実際に手を動かしてやって行こうと思います。

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/sympy/base/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/sympy/base/base_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


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

print('matplotlib version :', matplotlib.__version__)
print('scipy version :', scipy.__version__)
print('numpy version :', np.__version__)
```

    matplotlib version : 3.0.3
    scipy version : 1.4.1
    numpy version : 1.16.2


## sympyの3つの数値型

sympyは

- Real
- Rational
- Integer

の三つの型を持つようです。


### 分数の扱い


```python
from sympy import *

a = Rational(1,2)
a
```




    1/2



### 円周率の扱い


```python
pi.evalf()
```




    3.14159265358979




```python
pi * 2
```




    2*pi




```python
pi ** 2
```




    pi**2



### 自然対数の扱い


```python
exp(1) ** 2
```




    exp(2)




```python
(exp(1) ** 2).evalf()
```




    7.38905609893065



### 無限大


```python
oo > 999
```




    True




```python
oo + 1
```




    oo



### evalfで桁数の指定

引数に桁数を入れるようです。


```python
pi.evalf(1000)
```




    3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901224953430146549585371050792279689258923542019956112129021960864034418159813629774771309960518707211349999998372978049951059731732816096318595024459455346908302642522308253344685035261931188171010003137838752886587533208381420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909216420199



### 対数


```python
log(10)
```




    log(10)




```python
log(2,2**4).evalf()
```




    0.250000000000000




```python
E.evalf()
```




    2.71828182845905



## シンボルの定義

代数的な操作を可能にするために、$xや$yなどの変数を宣言します。


```python
x = Symbol('x')
y = Symbol('y')
```


```python
x + y ** 2 + x
```




    2*x + y**2




```python
(x + y) ** 3
```




    (x + y)**3



## 代数操作

expandやsimplifyはmathematicaと同じのようで助かります。

### 展開


```python
expand((x + y)**3)
```




    x**3 + 3*x**2*y + 3*x*y**2 + y**3



### 簡易化


```python
simplify((x + x*y) / x)
```




    y + 1



### 因数分解


```python
factor(x**3 + 3*x**2*y + 3*x*y**2 + y**3)
```




    (x + y)**3



## 微積分

### 極限


```python
limit(sin(x)/ x, x, 0)
```




    1




```python
limit(x, x, oo)
```




    oo




```python
limit(1 / x, x, oo)
```




    0




```python
limit(x**x, x, 0)
```




    1



### 微分


```python
diff(sin(x), x)
```




    cos(x)




```python
diff(sin(x) ** 2, x)
```




    2*sin(x)*cos(x)




```python
diff(sin(2 * x) , x)
```




    2*cos(2*x)




```python
diff(tan(x), x)
```




    tan(x)**2 + 1



### 微分が正しいかの確認
極限をとり、微分が正しいかどうかチェックできます。微分の定義に従って計算させるだけです。


```python
limit((tan(x+y) - tan(x))/y, y, 0)
```




    tan(x)**2 + 1



高階微分も可能です。


```python
diff(sin(x), x, 1)
```




    cos(x)



2階微分で元の値にマイナスをかけた値になっている事がわかります。


```python
diff(sin(x), x, 2)
```




    -sin(x)




```python
diff(sin(x), x, 3)
```




    -cos(x)




```python
diff(sin(x), x, 4)
```




    sin(x)



### 級数展開
Taylor展開も可能です。


```python
series(exp(x), x)
```




    1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)




```python
series(cos(x), x)
```




    1 - x**2/2 + x**4/24 + O(x**6)




```python
series(sin(x), x)
```




    x - x**3/6 + x**5/120 + O(x**6)



第三引数で展開する次数を指定出来るかと思いやってみました。


```python
series(exp(x), x, 6)
```




    exp(6) + (x - 6)*exp(6) + (x - 6)**2*exp(6)/2 + (x - 6)**3*exp(6)/6 + (x - 6)**4*exp(6)/24 + (x - 6)**5*exp(6)/120 + O((x - 6)**6, (x, 6))



違うみたいで、第三引数に数値計算する際の中心の数で、第四引数で展開させる次数のようです。


```python
series(exp(x), x, 0, 6)
```




    1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)



### 積分
微分が出来れば、積分ももちろん対応しています。


```python
integrate(x**3,x)
```




    x**4/4




```python
integrate(-sin(x),x)
```




    cos(x)




```python
integrate(exp(x),x)
```




    exp(x)




```python
integrate(log(x),x)
```




    x*log(x) - x




```python
integrate(exp(-x**2),x)
```




    sqrt(pi)*erf(x)/2



積分区間も指定することが出来ます


```python
integrate(x**2, (x, -1, 1))
```




    2/3




```python
integrate(sin(x), (x, 0, pi/2))
```




    1



範囲が無限大の広義積分も可能です。


```python
integrate(exp(-x), (x, 0, oo))
```




    1



ガウス積分をやってみます。


```python
integrate(exp(-x**2), (x, -oo, oo))
```




    sqrt(pi)




```python
integrate(exp(-x**2 / 3), (x, -oo, oo))
```




    sqrt(3)*sqrt(pi)



### 方程式を解く
代数方程式を解くことも可能です。ほぼmathematicaと同じインターフェースです。


```python
solve(x**3 - 1, x)
```




    [1, -1/2 - sqrt(3)*I/2, -1/2 + sqrt(3)*I/2]




```python
solve(x**4 - 1, x)
```




    [-1, 1, -I, I]



多変数の連立方程式も対応しています。


```python
solve([x + 5*y - 2, -3*x + 6*y - 15], [x, y])
```




    {x: -3, y: 1}



オイラーの式も計算してくれるようです。


```python
solve(exp(x) + 1, x)
```




    [I*pi]



##  行列演算

行列演算はnumpyでずっとやっているのでおそらく行列に関しては利用していませんが、一応ここでは手を動かして実行してみます。


```python
from sympy import Matrix

a = Matrix([[1,0], [0,1]])
b = Matrix([[1,1], [1,1]])
```


```python
a
```




    1/2




```python
b
```




    Matrix([
    [1, 1],
    [1, 1]])




```python
a + b
```




    Matrix([
    [2, 1],
    [1, 2]])




```python
a * b
```




    Matrix([
    [1, 1],
    [1, 1]])



## 微分方程式

常微分方程式も解く事が可能です。dsolveを利用します。


```python
f, g = symbols('f g', cls=Function)
f(x)
```




    f(x)




```python
f(x).diff(x,x) + f(x)
```




    f(x) + Derivative(f(x), (x, 2))




```python
dsolve(f(x).diff(x,x) + f(x), f(x))
```




    Eq(f(x), C1*sin(x) + C2*cos(x))



実際に数値計算に利用されているのがよくわかる高機能ぶりでした。

もちろん有料であるmathematicaよりも劣る部分もあると思いますが、積極的にmathematicaからの移行を進めていきます。すごいなぁ〜

## 参考ページ

こちらのページを参考にしました。

- http://www.turbare.net/transl/scipy-lecture-notes/packages/sympy.html
