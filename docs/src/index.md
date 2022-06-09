```@meta
CurrentModule = WarcraftShortestPaths
```

# WarcraftShortestPaths

Documentation for [WarcraftShortestPaths.jl](https://github.com/LouisBouvier/WarcraftShortestPaths.jl).

This package implements techniques of machine learning for operations research to compute shortest paths 
on Warcraft terrain images. 

This application was introduced in this [paper](https://arxiv.org/abs/1912.02175),
with the corresponding [dataset](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S)
and [code](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers).
It was also considered in a [Learning with Differentiable Perturbed Optimizers](https://arxiv.org/pdf/2002.08676.pdf)
setting, with corresponding [code](https://github.com/google-research/google-research/tree/master/perturbations/experiments).   

We focus on two main frameworks: learning by imitation, involving Fenchel-Young losses and perturbed maximizers, and learning 
from experience. Both are based on the [InferOpt.jl](https://github.com/axelparmentier/InferOpt.jl) package. We also leverage 
the [GridGraphs.jl](https://github.com/gdalle/GridGraphs.jl) package to compute shortest paths on grid graphs using 
Dijkstra algorithm.

## Dataset overview

Each point of the dataset is linked to a (12x12) Warcraft terrain grid. It is composed of:
- A color image of the Warcraft terrain of size (96x96).
- The cost labels for the corresponding grid of size (12x12).
- A 0-1 shortest path mask of size (12x12).

## Two frameworks

The two distinct frameworks we consider are:

1) **Learning by imitation**: given the images and labels, learn the cost ``c_\theta`` such that the labelled shortest 
paths are close to the shortest path computed with ``c_\theta`` using Dijkstra on the Warcraft grids.
More details on this framework can be seen [here](https://axelparmentier.github.io/InferOpt.jl/dev/math/#Learning-by-imitation).

2) **Learning by experience**: given the images and a black-box function that computes the cost of a path on any grid, learn the cost such that the true cost of the paths computed as shortest paths with respect to the learned costs are low.
More details on this framework can be seen [here](https://axelparmentier.github.io/InferOpt.jl/dev/math/#Learning-by-experience).

