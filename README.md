# SPEML3-Model-Inversion
To install Theano use:
`conda install -c conda-forge theano`

Otherwise it will install <1.0.4 and it has issues with NUMPY

For pylearn2 you have to download it, install it (w setup.py), and change every 
`from theano.compat import six`

to just `import six` (also install six). Don't use Conda

Todo: 
  * Understand what TensorFlow is doing
  * Understand why ZCA and GCA are used for preprocess
 
If it works, which I guess it should:
  * Add other models like in the original paper
  * Add some image manipulation operations like it mentions in the paper (but they don't implement??)
