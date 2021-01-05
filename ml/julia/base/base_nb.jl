;sw_vers

;julia --version

VERSION

import Pkg
Pkg.add("Gadfly")
Pkg.add("Flux")
Pkg.add("MLDatasets")
Pkg.add("Images")
Pkg.add("ImageView")

using Gadfly
using LinearAlgebra
using MLDatasets
using Images
using ImageView

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
x_train
x_train[:,:,1]
size(x_train[:,:,1])

import Images
Images.grayim(reshape(collect(UInt8, x_train[:, 1]), 28,28))
# grayim(x_train[0])
# grayim(x_train[:,:,1])
# imshow(x_train[:,:,1])

1 + 1

img = rand(4, 4)

imshow(img)

imgg = zeros(Gray, 5, 5)

imshow(imgg)

1 + 1

import Pkg;
Pkg.add("ImageIO")
Pkg.add("QuartzImageIO")
Pkg.add("ImageMagick")

array_2d = rand(5, 5)
imgg = colorview(Gray, array_2d)

imgg = colorview(Gray, x_train[:,:,9]')

# x = x_train[:,:,1]
# x = flipdim(x, 1)
# x = flipdim(x, 2)
# 
# imgg = colorview(Gray, x)

grayim(reshape(collect(UInt8, x_train[:, 1]), 28,28))







Pkg.add("JLD")

using JLD

network = load("/path/to/sample_network.jld")



function onehot{T}(::Type{T}, t::AbstractVector, l::AbstractVector)
    r = zeros(T, length(l), length(t))
    for i = 1:length(t)
        r[findfirst(l, t[i]), i] = 1
    end
    r
end
# @inline onehot(t, l) = onehot(Int, t, l)


2 + 3

function onehot(T, t, l)
    r = zeros(T, length(l), length(t))
    for i = 1:length(t)
        r[findfirst(l, t[i]), i] = 1
    end
    r
end

t = [1 2 3]
println(t[1])
l = [4 5 6]
onehot(Int64, t, l)

a = [2, 3, 1, 9, 4, 5]

findall(x->x==minimum(a), a)

minmum(a)


