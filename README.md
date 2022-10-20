# WarcraftShortestPaths

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://LouisBouvier.github.io/WarcraftShortestPaths.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://LouisBouvier.github.io/WarcraftShortestPaths.jl/dev)
[![Build Status](https://github.com/LouisBouvier/WarcraftShortestPaths.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LouisBouvier/WarcraftShortestPaths.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/LouisBouvier/WarcraftShortestPaths.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LouisBouvier/WarcraftShortestPaths.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

## Overview

This package implements techniques of machine learning for operations research to compute shortest paths 
on Warcraft terrain images. It is one of the applications of our paper [Learning with Combinatorial Optimization Layers: a Probabilistic Approach](https://arxiv.org/abs/2207.13513).

This application was introduced in this [paper](https://arxiv.org/abs/1912.02175),
with the corresponding [dataset](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S)
and [code](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers).
It was also considered in a [Learning with Differentiable Perturbed Optimizers](https://arxiv.org/pdf/2002.08676.pdf)
setting, with corresponding [code](https://github.com/google-research/google-research/tree/master/perturbations/experiments).   

We focus on two main frameworks: learning by imitation, involving Fenchel-Young losses and perturbed maximizers, and learning 
from experience. Both are based on the [InferOpt.jl](https://github.com/axelparmentier/InferOpt.jl) package. We also leverage 
the [GridGraphs.jl](https://github.com/gdalle/GridGraphs.jl) package to compute shortest paths on grid graphs using 
Dijkstra algorithm.

## Get started


1) Please download this [dataset](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S) and place it in the `data` folder of the repo. You can unzip it manually or using `decompress_dataset` function.

2) You can then activate the environment locally with this command:

```julia
using Pkg
Pkg.activate(".")
```

## Dataset overview

Each point of the dataset is linked to a (12x12) Warcraft terrain grid. It is composed of:
- A color image of the Warcraft terrain of size (96x96).
- The cost labels for the corresponding grid of size (12x12).
- A 0-1 shortest path mask of size (12x12).

## Two frameworks

The two distinct frameworks we consider are:

1) **Learning by imitation**: given the images and labels, learn the cost such that the labelled shortest 
paths are close to the shortest path computed with the given cost using Dijkstra on the Warcraft grids.

2) **Learning by experience**: given the images and a black-box function that computes the cost of a path on any grid, learn the cost such that the true cost of the paths computed as shortest paths with respect to the learned costs are low.