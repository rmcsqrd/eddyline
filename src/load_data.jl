"""
the MNIST data is stored in a binary format described here: http://yann.lecun.com/exdb/mnist/

This function takes in a list of integers and returns an array of associated pixel values/labels at the indices in the list
"""
function load_image_data(index_list, type="train")

        # determine which type of data
        if type == "train"
                img_fp = "/Users/rmcmahon/dev/eddyline/data/train-images-idx3-ubyte"
                label_fp = "/Users/rmcmahon/dev/eddyline/data/train-labels-idx1-ubyte"
        else
                img_fp = "/Users/rmcmahon/dev/eddyline/data/t10k-images-idx3-ubyte"
                label_fp = "/Users/rmcmahon/dev/eddyline/data/t10k-labels-idx1-ubyte"
        end

        # look at image data
        io = open(img_fp)
        data = read(io)

        # read header information
        # ntoh changes endianess from source to system
        magic_num, nimg, nrow, ncol = ntoh.(reinterpret(UInt32, data[1:16]))
        @assert magic_num == 2051 "File loading incorrectly (magic number doesn't match)"
        @assert maximum(index_list) <= nimg "Element of index list is larger than number of images"
        pixel_data = data[17:end]

        # read in image information
        image_array = zeros(nrow, ncol, length(index_list))
        for (k, index) in enumerate(index_list)
                start_offset = max((index-1)*nrow*ncol, 1)
                end_offset = start_offset+nrow*ncol-1
                image_array[:, :, k] = reshape(pixel_data[start_offset:end_offset], (nrow, ncol))' # transpose because of how reshape reshapes
        end

        # look at label data
        io = open(label_fp)
        data = read(io)

        # start by reading header info
        magic_num, nitem = ntoh.(reinterpret(UInt32, data[1:8]))
        @assert magic_num == 2049 "File loading incorrectly (magic number doesn't match)"
        label_data = data[9:end]
        label_list = []
        [append!(label_list, Int8(label_data[k])) for k in index_list]

        return image_array, label_list
end
