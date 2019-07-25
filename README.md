# SPEML3-Model-Inversion
Implementation of the Model Inversion Attack introduced with [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures](https://dl.acm.org/citation.cfm?id=2813677) (Fredrikson Et al.). Forked from https://github.com/yashkant/Model-Inversion-Attack

To install Theano use:
`conda install -c conda-forge theano`

Otherwise it will install <1.0.4 and it has issues with NUMPY

For pylearn2 you have to download it, install it (w setup.py), install six (w conda), and change every 
`from theano.compat import six`

to just `import six`. Don't use Conda for pylearn2, older version.

Todo: 
  * Understand what TensorFlow is doing
    * Use up-to-date TF commands and not deprecated ones
  * Understand why ZCA and GCA are used for preprocess
    * This is the most expensive / longest part
    * The preprocesed data is only used for inversion phase, training phase uses original
    * If we find other implementations we might be able to get rid of pylearn2
    * since pylearn2 uses old Theano functions it might not use the GPU in the inversion phase (no big impact tho)
  * Add other models like in the original paper
  * Add some image manipulation operations like it mentions in the paper (but they don't implement??)
