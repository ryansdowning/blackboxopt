# Black Box Optimization

![Travis-CI](https://img.shields.io/travis/com/ryansdowning/blackboxopt/main)
![Pytest Coverage](https://img.shields.io/codecov/c/github/ryansdowning/blackboxopt)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ryansdowning/blackboxopt.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ryansdowning/blackboxopt/context:python)

A package for applying hyperparameter tuning algorithms on a black box function which defines an optimization problem

My attempt at making an all-in-one hyperparameter optimization package. High-level abstraction allows algorithms to be implemented independently of search space constraints. Inspiration pulled from `ray[tune]` and `hyperopt`.

![alt text](https://imgs.xkcd.com/comics/standards.png)