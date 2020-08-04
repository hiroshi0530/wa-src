
using LinearAlgebra

1+1


x = [1,2,3]
y = [10,20,30]

x + y

x .* y

dot(x,y)

using Gadfly

x = linspace(-5.0, 5.0)
y = sigmoid(x)
plot(x=x, y=y, Geom.line)
