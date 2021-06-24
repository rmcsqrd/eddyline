module eddyline

using ImageView
using LinearAlgebra

include("load_data.jl")
include("layers.jl")
greet() = print("Hello World!")

end # module
