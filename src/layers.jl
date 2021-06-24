"""

"""
function conv2d(img, kernel, padding=false)
        xdim, ydim = size(img) 
        kx, ky = size(kernel)
      
        # check input stuff
        @assert size(kernel) == reverse(size(kernel)) "kernel dimensions need to equal"
        @assert isodd(size(kernel)[1]) "kernel dimensions need to be odd"

        # determine padding and setup
        offset = Int((kx-1)/2)
        pad = padding ? offset : 0 

        pad_img = zeros(xdim+pad*2, ydim+pad*2)
        pad_img[1+pad:pad+xdim, 1+pad:pad+ydim] = img

        outimg = zeros(size(img) .- (2*offset) .+ (2*pad))
        out_xdim, out_ydim = size(outimg)

        # do convolution
        for i in offset+1:out_xdim
                for j in offset+1:out_ydim
                        img_kernel = pad_img[i-offset:i+offset, j-offset:j+offset]
                        outimg[i-offset,j-offset] = dot(img_kernel, kernel)
                end
        end
        return outimg
end

function pool2d(img, pool_size, type=:max)

        # interpret type and define operation function
        @assert type ∈ [:max, :min, :average] "check type argument"
        @assert size(img) .% pool_size == (0,0) "pooling dimensions not compatible with image dimensions"
        if type == :max
                pool_op = (kernel) -> maximum(kernel)
        elseif type == :min
                pool_op = (kernel) -> minimum(kernel)
        elseif type == :average
                pool_op = (kernel) -> sum(kernel)/sum(size(kernel))
        end

        # do pooling operation
        output_dims = Int.(size(img) ./ pool_size)
        output = zeros(output_dims)

        
        
                

end
