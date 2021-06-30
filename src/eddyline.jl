module eddyline

using ImageView
using LinearAlgebra

include("load_data.jl")
export load_image_data

include("layers.jl")
export conv2d, pool2d, dense, flatten, model

include("nonlinearities.jl")
export relu, softmax


end # module
