
;sw_vers

;julia --version

VERSION

import Pkg
Pkg.add("Gadfly")
Pkg.add("Flux")
Pkg.add("MLDatasets")

using Gadfly
using LinearAlgebra
using MLDatasets

1 + 2

2 - 1

5 * 3

5 / 3 

5 % 3

x = [1, 2, 3, 4, 5]

y = [10, 20, 30, 40, 50]

x + y

x ⋅ y

dot(x, y)

x .* y

a = [1 2; 3 4]
b = [10 20; 30 40]

a

b

a + b

# * で通常のかけ算
a * b

# .*でアダマール積
a .* b

function sigmoid(x)
  1.0 ./ (1.0 .+ exp.(-x))
end

x = range(-5.0,stop = 5.0,length = 10)
y = sigmoid(x)

plot(x=x, y=y, Geom.line)

function nn3lp(x)
  W1 = [0.1 0.2; 0.3 0.4; 0.5 0.6];
  b1 = [0.1, 0.2, 0.3];
  W2 = [0.1 0.2 0.3; 0.4 0.5 0.6];
  b2 = [0.1, 0.2];
  W3 = [0.1 0.2; 0.3 0.4];
  b3 = [0.1, 0.2];

  a1 = W1 * x .+ b1
  z1 = sigmoid(a1)
  a2 = W2 * z1 .+ b2
  z2 = sigmoid(a2)
  a3 = W3 * z2 .+ b3
  identity(a3)
end

x = [1.0, 0.5]
y = nn3lp(x)



a1 = [1, 2, 3, 4]
c = maximum(a1)
c



x_train, t_train = MNIST.traindata();

x_test, t_test = MNIST.testdata();

size(x_train)

grayim(reshape(collect(UInt8, x_train[:, 1]), 28,28)')




