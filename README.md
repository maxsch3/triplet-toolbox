# Triplet tools

This package is a set of tool for training triplet networks built with tensorflow and/or keras. 
It supports both tensorflow V1 and V2. It also supports Keras standalone and tf.keras 

It has the following components:

- loss functions:
    - batch semi hard loss from google's paper [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) 
    - batch hard loss from paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
    - batch pre-train loss a loss functions to be used to prime a newly inititalized model


# Installation

```shell script
pip install triplet-tools
```

or 

```shell script
pip install git+https://github.com/maxsch3/triplet-toolbox
```


# Usage

## Use of loss functions

The loss functions can be used with tensorflow or tensorflow/keras models. They all used in the same way:

```python
from triplet_tools.losses import triplet_batch_semihard_loss

loss_function = triplet_batch_semihard_loss(margin=0.5, metric='euclidean_norm')

# Model definition goes here
# ...

model.compile('adam', loss_function)

model.fit(...)
```

all loss functions are interchangeable and can be used in one model without changes to it. The change of loss function 
does not require model re-initialization, so you can stop training and continue with another loss funtion. You just 
need to compile it with new loss:

```python
from triplet_tools.losses import triplet_batch_hard_loss

new_loss_function = triplet_batch_hard_loss(metric='cosine')

model.compile('adam', new_loss_function)

model.fit(...)
```

## Requirements to a model and data structure

All loss functions implement triplet mining approach where triplets are generated inside network automatically,
so you don't have to design a triplet generator outside of your network in data generator 
Instead, you will need to create a single encoder network outputting embeddings similar to encoder part 
 in autoencoders and the loss funstions will do the rest
 
The functions use effective hard triplet mining which looks for all possible pairs in a minibatch and 
only selects the worst performing pairs, which called hard triplets. These hard triplets are then used in loss
function for calculating gradients

All loss functions in tensorflow/keras take two arguments: 
- y_true - "True" labels or values from training data.
- y_pred - predicted labels or values taken from model outputs

For y_true, all loss functions expect labels as a vector of integers with dimension `(batch size, )` or
`(batch_size, 1)`. While y_pred, must be embeddings produced by a model. Embeddings are vectors in latent space,
each of which represents a datapoint. Embeddings is an array (tensor) of floats with dimensions
`(batch_size, latent_dimensions)`. Where `latent_dimensions` is dimensionality of latent space.

The embeddings should allow both positive and negative values, and therefore the activation on the output layer of
your model must allow them. For example, you can use `linear` or `sigmoid` activations:

```python
latent_dimensions = 10

# inputs and core model definition
output = Dense(latent_dimensions, activation='linear')(x)

model = Model(inputs, output)
```

## Metrics supported

All loss functions have parameter `metrics` which defines the way how distance is calculated between vectors. 

Currently 3 types of metrics are supported:

- Euclidean normalized (default). Standard euclidean distance with prior normalization of all vectors 
(making their lengths = 1)
- Euclidean (without normalization)
- Cosine - cosine distance calculated as ![formula](https://render.githubusercontent.com/render/math?math=1-cos\(v_i,v_j\))

# Reference

## priming loss

*function* `triplet_batch_priming_loss` *(metric='euclidean_norm')*

Use this loss function when your model outputs a single solution for all inputs. It makes high gradients 
when samples from different classes are put together, pushing them apart. 


**Parameters:** 

- metric - *string* one of the supported metrics. Default: `euclidean_norm`

**Returns:**

- function that can be plugged into a keras/tensorflow model

Example:

```python
from triplet_tools.losses import triplet_batch_priming_loss

loss_function = triplet_batch_priming_loss(metric='euclidean')

model.compile('adam', loss_function)

model.fit(...)
```

## batch hard loss

This loss function is an implementation of Batch Hard loss described in this [paper](https://arxiv.org/abs/1703.07737)

It neither pulls positive pairs to anchor point nor pushes negative pairs away. Instead, it pushes positive and 
negative vectors apart leaving anchor point alone. 

It is calculated using below formula

![formula](https://render.githubusercontent.com/render/math?math=Loss=\sum_{i}^N%20max(d_i^{%2B}-d_i^{-}%2Bmargin,%200))

Which means it maximizes difference between positive and negative distances mined for each anchor point up to a 
distance = margin. 
No loss and gradients are generated if the difference is beyond the margin. 

**Parameters:**

- metric - *string* one of the supported metrics. Default: `euclidean_norm`
- margin - *float* sets parameter margin from above formula. Default: `1.0`

**Returns**

- function that can be plugged into a keras/tensorflow model

Example:

```python
from triplet_tools.losses import triplet_batch_hard_loss

loss_function = triplet_batch_priming_loss(margin=.4, metric='cosine')

model.compile('adam', loss_function)

model.fit(...)
```

## batch semihard loss

This loss function is an implementation of Batch Semihard loss described in this [FaceNet paper](https://arxiv.org/abs/1503.03832)
from Google. 

*There is an existing implementation of triplet loss with batch semihard online mining in 
[Tensorflow addons](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/triplet_semihard_loss) 
but tensorflow's implementation is missing parameter `semi_margin` available in this function*

This function is an evolution of batch hard loss with slightly changed negative pair selection: instead of picking the
nearest vector not belonging to the same class as an anchor vector (hard negative), this function picks not the hardest 
negative, but so called semi-hard negative. Semi hard negative is a nearest negative which is no closer to an anchor
poing than its hard positive pair plus additional `semihard_margin`, which can be both positive and negative 

You can read more about that in the paper or in this [blog](https://omoindrot.github.io/triplet-loss)

**Parameters**

- metric - *string* one of the supported metrics. Default: `euclidean_norm`
- margin - *float* sets parameter margin from above formula. Default: `1.0`
- semi_margin - *float* moves semi hard cutoff closer (semi_margin < 0) or (semi_margin > 0) 
further than anchor's hard positive. `Default: 0.0`

Example:
```python
from triplet_tools.losses import triplet_batch_semihard_loss

loss_function = triplet_batch_semihard_loss(margin=.4, semi_margin=.05)

model.compile('adam', loss_function)

model.fit(...)
```

