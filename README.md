# ml-from-scratch-cpp
Fundamental machine learning algorithms implemented from scratch, written in C++.




Repo structure:
ml-from-scratch-cpp/
├─ CMakeLists.txt
├─ include/
│  └─ ml_from_scratch/
│     ├─ core/
│     │  ├─ types.hpp           # using Real=double; using index_t=size_t; constants
│     │  ├─ view.hpp            # MatrixView / VectorView (non-owning buffer views)
│     │  ├─ distance.hpp        # l2, l1, cosine (used by KNN; K-means later)
│     │  ├─ metrics.hpp         # mse, mae, accuracy; (entropy utils if needed)
│     │  ├─ criteria.hpp        # gini, entropy, squared_error (for trees)
│     │  ├─ rng.hpp             # seedable RNG helpers (for tree splits / tie-breaks)
│     │  ├─ io.hpp              # minimal CSV loader → flat std::vector<double>
│     │  └─ errors.hpp          # input validation helpers & exceptions
│     ├─ la/                    # your own linear algebra mini-lib (files later)
│     ├─ supervised/
│     │  ├─ neighbors/
│     │  │  └─ knn.hpp          # KNNClassifier (manual distances; simple vote)
│     │  ├─ linear_model/
│     │  │  └─ linear_regression.hpp  # OLS via your QR (later lives in la/)
│     │  └─ tree/
│     │     ├─ decision_tree.hpp       # Core tree impl (splitting, nodes, stopping)
│     │     ├─ decision_tree_class.hpp # using alias: core + {gini/entropy} policy
│     │     └─ decision_tree_regr.hpp  # using alias: core + {squared_error} policy
│     └─ unsupervised/                 # (empty for now; kmeans later)
├─ src/                                 # optional .cpps if you don’t go header-only
│     ├─ core/
│     │  ├─ types.cpp           
│     │  ├─ view.cpp            
│     │  ├─ distance.cpp        
│     │  ├─ metrics.cpp         
│     │  ├─ criteria.cpp        
│     │  ├─ rng.cpp             
│     │  ├─ io.cpp              
│     │  └─ errors.cpp          
│     ├─ la/                    
│     ├─ supervised/
│     │  ├─ neighbors/
│     │  │  └─ knn.cpp          
│     │  ├─ linear_model/
│     │  │  └─ linear_regression.cpp  
│     │  └─ tree/
│     │     ├─ decision_tree.cpp      
│     │     ├─ decision_tree_class.cpp 
│     │     └─ decision_tree_regr.cpp  
│     └─ unsupervised/
├─ examples/
│  ├─ knn_iris.cpp
│  ├─ linreg_toy.cpp
│  └─ dtree_toy.cpp
├─ tests/
│  ├─ test_knn.cpp
│  ├─ test_linear_regression.cpp
│  └─ test_decision_tree.cpp
├─ datasets/
│  ├─ toy_linreg.csv
│  ├─ toy_knn.csv
│  └─ toy_dtree.csv
├─ .clang-format
├─ .clang-tidy
├─ .gitignore
└─ README.md

