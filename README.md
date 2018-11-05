DiffAI
======

![High Level](https://raw.githubusercontent.com/eth-sri/diffai/master/media/overview.png)

DiffAI is a system for training neural networks to be provably robust and for proving that they are robust.

Background
----------

By now, it is well known that otherwise working networks can be tricked by clever attacks.  
For example [Goodfellow et al.](https://arxiv.org/abs/1412.6572) demonstrated a network with high classification accuracy which classified one image of a panda correctly, and a seemingly identical attack picture 
incorrectly.  Many defenses against this type of attack have been produced, but very few produce networks for which *provably* verifying the safety of a prediction is feasible.

Abstract Interpretation is a technique for verifying properties of programs by soundly overapproximating their behavior.  When applied to neural networks, an infinite set (a ball) of possible inputs is passed to an approximating "abstract" network
to produce a superset of the possible outputs from the actual network.  Provided an appropreate representation for these sets, demonstrating that the network classifies everything in the ball correctly becomes a simple task.
The method used to represent these sets is the abstract domain, and the specific approximations are the abstract transformers.  

In DiffAI, the entire abstract interpretation process is programmed using PyTorch so that it is differentiable and can be run on the GPU, 
and a loss function is crafted so that low values correspond to inputs which can be proved safe (robust).

Requirements 
------------

python 3.5 or higher, pip3, and virtualenv

Recommended Setup 
-----------------

```
$ git clone https://github.com/eth-sri/DiffAI.git
$ cd DiffAI
$ virtualenv pytorch --python python3.6 # or whatever version you are using
$ source pytorch/bin/activate
(pytorch) $ pip install -r requirements.txt
```

Note: you need to activate your virtualenv every time you start a new shell.

Getting Started
---------------

DiffAI can be run as a standalone program.  To see a list of arguments, type 

```
(pytorch) $ python . --help
```

At the minimum, DiffAI expects at least one domain to train with and one domain to test with, and a network with which to test.  For example, to train with the Box domain, baseline training (Point) and test against the FGSM attack and the ZSwitch domain with a simple feed forward network on the MNIST dataset (default, if none provided), you would type:

```
(pytorch) $ python . -d Point -d Box -t FGSM -t ZSwitch -n ffnn
```

Unless otherwise specified by "--out", the output is logged to the folder "out/".  
In the folder corresponding to the experiment that has been run, one can find the saved configuration options in 
"config.txt", and a pickled net which is saved every 10 epochs (provided that testing is set to happen every 10th epoch).

To load a saved model, use "--test"

The default specification type is the L_infinity Ball specified explicitly by "--spec boxSpec", 
which uses an epsilon specified by "--width"

Contents
--------

* components.py: A high level neural network library for composable layers and operations
* domains.py: abstract domains and attacks which can be used as a drop in replacement for pytorch tensors in any model built with components from components.py
* losses.py: high level loss functions for training which can take advantage of abstract domains.
* models.py: A repository of models to train with which are used in the paper.
* \_\_main\_\_.py: The entry point to run the experiments.
* helpers.py: Assorted helper functions.  Does some monkeypatching, so you might want to be careful importing our library into your project.

Notes
-----

Not all of the datasets listed in the help message are supported.  Supported datasets are:

* CIFAR10
* CIFAR100
* MNIST
* SVHN
* FashionMNIST

Unsupported datasets will not necessarily throw errors.

About
-----

* This repository contains the code used for the experiments in the ICML paper, [Differentiable Abstract Interpretation for Provably Robust Neural Networks](https://files.sri.inf.ethz.ch/website/papers/icml18-diffai.pdf)
* Further information and related projects can be found at [the SafeAI Project](http://safeai.ethz.ch/)
* [High level slides](https://files.sri.inf.ethz.ch/website/slides/mirman2018differentiable.pdf)

Citing This Framework
---------------------

```
@inproceedings{
  title={Differentiable Abstract Interpretation for Provably Robust Neural Networks},
  author={Mirman, Matthew and Gehr, Timon and Vechev, Martin},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2018},
  url={https://www.icml.cc/Conferences/2018/Schedule?showEvent=2477},
}
```

Contributors
------------

* [Matthew Mirman](https://www.mirman.com) - matt@mirman.com
* [Timon Gehr](https://www.sri.inf.ethz.ch/tg.php) - timon.gehr@inf.ethz.ch
* [Martin Vechev](https://www.sri.inf.ethz.ch/vechev.php) - martin.vechev@inf.ethz.ch

License and Copyright
---------------------

* Copyright (c) 2018 [Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [MIT License](https://opensource.org/licenses/MIT)
