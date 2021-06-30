"""
CONVOLUTIONAL LAYER
"""
# struct to contain values of conv layer
mutable struct conv2d{M<:AbstractMatrix, V, F}
    kernel::M
    bias::V
    σ::F
end

# constructor function for conv layer
function conv2d(kernel_size::Tuple{Int64, Int64}, σ)
    kernel = rand(kernel_size...)
    bias = rand()
    return conv2d(kernel, bias, σ)
end

# functor for evaluating the layer
function (layer::conv2d)(input; padding=false)
        kernel, b, σ = layer.kernel, layer.bias, layer.σ
        xdim, ydim = size(input) 
        kx, ky = size(kernel)
      
        # check input stuff
        @assert size(kernel) == reverse(size(kernel)) "kernel dimensions need to equal"
        @assert isodd(size(kernel)[1]) "kernel dimensions need to be odd"

        # determine padding and setup
        offset = Int((kx-1)/2)
        pad = padding ? offset : 0 

        pad_input = zeros(xdim+pad*2, ydim+pad*2)
        pad_input[1+pad:pad+xdim, 1+pad:pad+ydim] = input

        output = zeros(size(input) .- (2*offset) .+ (2*pad))
        out_xdim, out_ydim = size(output)

        # do convolution
        for i in offset+1:out_xdim+1
            for j in offset+1:out_ydim+1
                input_kernel = pad_input[i-offset:i+offset, j-offset:j+offset]
                output[i-offset,j-offset] = σ.(dot(input_kernel, kernel) + b)
            end
        end
        return output
end

"""
POOLING LAYER
"""
mutable struct pool2d{V}
    n::V
end

function (layer::pool2d)(input; type=:max)

        # interpret type and define operation function
        # n is dimension for pooling kernel
        n = layer.n
        @assert type ∈ [:max, :min, :average] "check type argument"
        @assert size(input) .% n == (0,0) "pooling dimensions not compatible with image dimensions (make sure input dims are divisible by pool_size)"

        # define pooling function based on type
        if type == :max
            pool_op = (kernel) -> maximum(kernel)
        elseif type == :min
            pool_op = (kernel) -> minimum(kernel)
        elseif type == :average
            pool_op = (kernel) -> sum(kernel)/length(kernel)
        end

        # do pooling operation
        output_dims = Int.(size(input) ./ n)
        output = zeros(output_dims)

        out_x, out_y = size(output)

        for i in 1:out_x
            for j in 1:out_y
                k = input[(i*n)-n+1:i*n, (j*n)-n+1:j*n]
                output[i,j] = pool_op(k)
            end
        end
        return output
end

"""
DENSE LAYER
"""
mutable struct dense{M<:AbstractMatrix, V, F}
    W::M
    b::V
    σ::F
end

function dense(in_dim::Int64, out_dim::Int64, σ)
    W = rand(out_dim, in_dim)
    b = rand(out_dim)
    return dense(W, b, σ)
end

function (layer::dense)(input)
    return layer.σ.(layer.W*input) + layer.b
end

"""
FLATTEN FUNCTION
"""
function flatten(input)
    return reshape(input, prod(size(input)))
end

"""
MODEL WRAPPER FUNCTION
"""
struct model{T<:Tuple}
    layers::T
end

function (model::model)(input::Matrix{Float64})
    for layer in model.layers
        input = layer(input)
    end
    return input
end
