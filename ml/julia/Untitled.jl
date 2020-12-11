using LinearAlgebra

1+1


x = [1,2,3]
y = [10,20,30]

x + y

x .* y

dot(x,y)



q = [1,0]

kron(q,q)

function svq(N)
  q = [1,0]
  sv = q
  for i=2:N
    sv = kron(sv,q)
  end
  return sv
end

svq(3)













using Gadfly

x = linspace(-5.0, 5.0)
y = sigmoid(x)
plot(x=x, y=y, Geom.line)
