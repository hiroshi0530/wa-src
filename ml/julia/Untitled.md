```julia
using LinearAlgebra
```


```julia
1+1

```




    2




```julia
x = [1,2,3]
y = [10,20,30]

x + y
```




    3-element Array{Int64,1}:
     11
     22
     33




```julia
x .* y
```




    3-element Array{Int64,1}:
     10
     40
     90




```julia
dot(x,y)
```




    140




```julia

```


```julia
q = [1,0]
```




    2-element Array{Int64,1}:
     1
     0




```julia
kron(q,q)
```




    4-element Array{Int64,1}:
     1
     0
     0
     0




```julia
function svq(N)
  q = [1,0]
  sv = q
  for i=2:N
    sv = kron(sv,q)
  end
  return sv
end
```




    svq (generic function with 1 method)




```julia
svq(3)
```




    8-element Array{Int64,1}:
     1
     0
     0
     0
     0
     0
     0
     0




```julia

```


```julia

```


```julia

```


```julia

```


```julia

```


```julia

```


```julia
using Gadfly

x = linspace(-5.0, 5.0)
y = sigmoid(x)
plot(x=x, y=y, Geom.line)
```


    ArgumentError: Package Gadfly not found in current path:
    - Run `import Pkg; Pkg.add("Gadfly")` to install the Gadfly package.


    

    Stacktrace:

     [1] require(::Module, ::Symbol) at ./loading.jl:893

     [2] include_string(::Function, ::Module, ::String, ::String) at ./loading.jl:1091

