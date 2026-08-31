# ml-from-scratch-cpp

Fundamental machine learning algorithms implemented from scratch, written in C++.

## Algorithms

- [x] K-nearest neighbours (KNN)
- [ ] Linear regression
- [ ] Decision tree
- [ ] Multi-layer perceptron (MLP)

## Project layout

```text
ml-from-scratch-cpp/
├── CMakeLists.txt                         # CMake build configuration
├── include/
│   └── ml_from_scratch/
│       ├── core/                          # Shared types, views, distances, and metrics
│       │   ├── criteria.hpp
│       │   ├── distance.hpp
│       │   ├── metrics.hpp
│       │   ├── types.hpp
│       │   └── view.hpp
│       └── supervised/
│           ├── linear_model/
│           │   └── linear_regression.hpp
│           ├── neighbours/
│           │   ├── knn.hpp
│           │   └── knn_optimised.hpp
│           └── tree/
│               └── decision_tree.hpp
├── src/
│   ├── core/                              # Implementations for shared utilities
│   └── supervised/
│       ├── linear_model/
│       ├── neighbours/
│       └── tree/
├── examples/
│   └── knn_examples.cpp                   # K-nearest-neighbours example program
├── tests/
│   ├── core/
│   │   └── test_distance.cpp
│   └── supervised/
│       └── neighbours/
│           ├── test_knn.cpp
│           └── test_knn_optimised.cpp
├── external/
│   └── googletest/                        # GoogleTest dependency (git submodule)
├── TODO.md                                # Planned work
├── LICENSE
└── README.md
```
