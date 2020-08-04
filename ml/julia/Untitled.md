

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

