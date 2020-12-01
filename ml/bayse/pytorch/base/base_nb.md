## top template

### 参考URL
- https://deepblue-ts.co.jp/%E7%B5%B1%E8%A8%88%E5%AD%A6/%E3%83%99%E3%82%A4%E3%82%BA%E7%B5%B1%E8%A8%88/pyro-try/

### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/blob/master/article/library/scipy/template/template_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/blob/master/article/library/scipy/template/template_nb.ipynb)

### 筆者の環境
筆者のOSはmacOSです。LinuxやUnixのコマンドとはオプションが異なります。


```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.14.6
    BuildVersion:	18G6032



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



```python
import torch
import torch.distributions.constraints as constraints
import pyro 
import pyro.distributions as dist
```


```python
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc import NUTS
from pyro.optim import Adam
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.infer import Predictive
```


```python
print('torch version :', torch.__version__)
print('pyro version :', pyro.__version__)
```

    torch version : 1.7.0
    pyro version : 1.5.1



```python
x = np.random.uniform(-2,2,20)
y = 2 * x + 3 + np.random.normal(0,1,20)

plt.grid()
plt.scatter(x,y)
plt.show()
```


    
![svg](base_nb_files/base_nb_8_0.svg)
    



```python
x = torch.tensor(x)
y = torch.tensor(y)
```


```python
print('x type :', type(x))
print('y type :', type(y))
print('x shape :', x.shape)
print('y shape :', y.shape)
```

    x type : <class 'torch.Tensor'>
    y type : <class 'torch.Tensor'>
    x shape : torch.Size([20])
    y shape : torch.Size([20])


## MCMCサンプリング(NUTS)

### モデルの構築


```python
def model(x,y):
  a = pyro.sample('a', dist.Normal(0., 5.))
  b = pyro.sample('b', dist.Normal(0.,5.))
  y = pyro.sample('y', dist.Normal(a*x + b, 1.), obs=y)
  return y
```


```python
nuts_kernel = NUTS(model, adapt_step_size=True)
mcmc_run = MCMC(nuts_kernel, num_samples=1000, warmup_steps=1000)
mcmc_run.run(x, y)
```

    Sample: 100%|█████████████████████████████| 2000/2000 [00:10, 198.96it/s, step size=9.98e-01, acc. prob=0.877]


MCMCで得られたサンプルの取得


```python
posterior_a = mcmc_run.get_samples()['a']
posterior_b = mcmc_run.get_samples()['b']
```


```python
print(posterior_a[:20])
```

    tensor([2.8355, 2.7370, 2.7370, 2.0700, 2.1688, 2.4700, 1.9804, 2.4388, 2.4116,
            2.4116, 2.2302, 2.2365, 2.3438, 2.2297, 2.2256, 2.1555, 2.2773, 2.2274,
            2.2664, 2.2103], dtype=torch.float64)



```python
print(posterior_b[:20])
```

    tensor([3.5353, 3.6787, 3.6787, 3.6408, 4.0378, 3.7112, 3.5571, 3.2672, 3.5452,
            3.5452, 3.5789, 3.5862, 3.7441, 3.5808, 3.6269, 3.3852, 3.9679, 3.3639,
            3.8180, 3.5705], dtype=torch.float64)


次にこのサンプルを用いた予測分布の計算を行います。
こちらも関数一つで予測分布の計算が行えるので簡単です。


```python
pred = Predictive(model,{'a':posterior_a,'b':posterior_b},return_sites=["y"])
```


```python
x_ = np.linspace(-2,2,100)
y_ = pred.get_samples(torch.tensor(x_),None)['y']
```


```python
y_mean = y_.mean(0)
y_std = y_.std(0)
plt.figure(figsize=(10,5))
plt.plot(x_,y_mean)
plt.fill_between(x_,y_mean-y_std*2,y_mean+y_std*2,alpha=0.3)
plt.grid()
plt.scatter(x,y)
plt.show()
```


    
![svg](base_nb_files/base_nb_21_0.svg)
    


## 変分推論


```python
def guide(x,y):
  a_loc = pyro.param('a_loc',torch.tensor(0.))
  b_loc = pyro.param('b_loc',torch.tensor(0.))
  a_scale = pyro.param('a_scale',torch.tensor(1.),constraints.positive)
  b_scale = pyro.param('b_scale',torch.tensor(1.),constraints.positive)
  pyro.sample('a',dist.Normal(a_loc,a_scale))
  pyro.sample('b',dist.Normal(b_loc,b_scale))
```


```python
adam_params = {"lr": 0.001, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 1000
# do gradient steps
for step in range(n_steps):
  svi.step(x, y)
```


```python
for name in pyro.get_param_store():
  print(name + ':{}'.format(pyro.param(name)))
```

    a_loc:0.7868573069572449
    b_loc:0.906243085861206
    a_scale:0.7018969058990479
    b_scale:0.8006583452224731



```python
y_pred = Predictive(model=model,guide=guide,num_samples=1000,return_sites=["y"])
```


```python
x_ = torch.tensor(np.linspace(-2,2,100))
y_ = y_pred.get_samples(x_,None)
```


```python
y_mean = y_['y'].mean(0).detach()
y_std = y_['y'].std(0).detach()
plt.figure(figsize=(10,5))
plt.plot(x_,y_mean)
plt.fill_between(x_,y_mean-y_std*2,y_mean+y_std*2,alpha=0.3)
plt.grid()
plt.scatter(x,y)
plt.show()
```


```python
# aについて
a = np.random.normal(pyro.param('a_loc').detach().numpy(),
                     pyro.param('a_scale').detach().numpy(),1000)
plt.hist(a,density=True,bins=50)
plt.grid()
plt.hist(posterior_a,density=True,alpha=0.5,bins=50)
plt.show()
```


    
![svg](base_nb_files/base_nb_29_0.svg)
    



```python
# bについて
b = np.random.normal(pyro.param('b_loc').detach().numpy(),
                     pyro.param('b_scale').detach().numpy(),1000)
plt.hist(b,density=True,bins=50)
plt.grid()
plt.hist(posterior_b,density=True,alpha=0.5,bins=50)
plt.show()
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-26-66b56ce6f808> in <module>
          3                      pyro.param('b_scale').detach().numpy(),1000)
          4 plt.hist(b,density=True,bins=50)
    ----> 5 plt.hist(posterior_b,density=True,alpha=0.5,bins=50)
          6 plt.show()


    ~/anaconda3/lib/python3.8/site-packages/matplotlib/pyplot.py in hist(x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, data, **kwargs)
       2683         orientation='vertical', rwidth=None, log=False, color=None,
       2684         label=None, stacked=False, *, data=None, **kwargs):
    -> 2685     return gca().hist(
       2686         x, bins=bins, range=range, density=density, weights=weights,
       2687         cumulative=cumulative, bottom=bottom, histtype=histtype,


    ~/anaconda3/lib/python3.8/site-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
       1436     def inner(ax, *args, data=None, **kwargs):
       1437         if data is None:
    -> 1438             return func(ax, *map(sanitize_sequence, args), **kwargs)
       1439 
       1440         bound = new_sig.bind(ax, *args, **kwargs)


    ~/anaconda3/lib/python3.8/site-packages/matplotlib/axes/_axes.py in hist(self, x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, **kwargs)
       6721                 else:
       6722                     height = m
    -> 6723                 bars = _barfunc(bins[:-1]+boffset, height, width,
       6724                                 align='center', log=log,
       6725                                 color=c, **{bottom_kwarg: bottom})


    ~/anaconda3/lib/python3.8/site-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
       1436     def inner(ax, *args, data=None, **kwargs):
       1437         if data is None:
    -> 1438             return func(ax, *map(sanitize_sequence, args), **kwargs)
       1439 
       1440         bound = new_sig.bind(ax, *args, **kwargs)


    ~/anaconda3/lib/python3.8/site-packages/matplotlib/axes/_axes.py in bar(self, x, height, width, bottom, align, **kwargs)
       2479         args = zip(left, bottom, width, height, color, edgecolor, linewidth)
       2480         for l, b, w, h, c, e, lw in args:
    -> 2481             r = mpatches.Rectangle(
       2482                 xy=(l, b), width=w, height=h,
       2483                 facecolor=c,


    ~/anaconda3/lib/python3.8/site-packages/matplotlib/patches.py in __init__(self, xy, width, height, angle, **kwargs)
        740         """
        741 
    --> 742         Patch.__init__(self, **kwargs)
        743 
        744         self._x0 = xy[0]


    ~/anaconda3/lib/python3.8/site-packages/matplotlib/patches.py in __init__(self, edgecolor, facecolor, color, linewidth, linestyle, antialiased, hatch, fill, capstyle, joinstyle, **kwargs)
         56         %(Patch)s
         57         """
    ---> 58         artist.Artist.__init__(self)
         59 
         60         if linewidth is None:


    ~/anaconda3/lib/python3.8/site-packages/matplotlib/artist.py in __init__(self)
        103         self._sketch = mpl.rcParams['path.sketch']
        104         self._path_effects = mpl.rcParams['path.effects']
    --> 105         self._sticky_edges = _XYPair([], [])
        106         self._in_layout = True
        107 


    KeyboardInterrupt: 


## まとめ


```python

```


```python

```


```python

```
