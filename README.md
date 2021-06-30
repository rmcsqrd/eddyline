# eddyline
Work in progress machine learning testbed using the MNIST numbers dataset

## Example Usage

#### Load Image
```julia
using ImageView
idx_list = [1,2,100] # indices of MNIST dataset to load
img_array, labels = load_image_data(idx_list)
imshow(img_array[:,:,1]) # should show the number 5
println(labels[1]) # should be 5
```

#### Construct/Evaluate a Model
```julia
(@v1.6) pkg> activate .
using eddyline
img_array, label = eddyline.load_image_data([1]);
img = img_array[:,:,1];
l1 = conv2d([1 0 -1; 1 0 -1; 1 0 -1], 1, relu);
l2 = pool2d(2);
l3 = dense(169, 9, relu);
simple_conv = model((l1, l2, flatten, l3, softmax))
simple_conv(img)
```


