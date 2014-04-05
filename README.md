# Orchestra

[![Build Status](https://travis-ci.org/sjenkz/Orchestra.jl.png)](https://travis-ci.org/sjenkz/Orchestra.jl)

Orchestra is a heterogeneous ensemble learning package for the Julia programming
language. It is driven by a uniform machine learner API designed for learner
composition.

## Tutorial

## Available Learners

| Learner               | Library         | Constraints      | Metrics  | Description                       |
|-----------------------|-----------------|------------------|----------|-----------------------------------|
| SVM                   | LIBSVM.jl       | Numeric features | accuracy | Support Vector Machines.          |
| PrunedTree            | DecisionTree.jl |                  | accuracy | C4.5 Decision Tree.               |
| RandomForest          | DecisionTree.jl |                  | accuracy | C4.5 Random Forest.               |
| DecisionStumpAdaboost | DecisionTree.jl |                  | accuracy | C4.5 Adaboosted Decision Stumps.  |
| VoteEnsemble          | Orchestra.jl    |                  | accuracy | Majority Vote Ensemble.           |
| StackEnsemble         | Orchestra.jl    |                  | accuracy | Stack Ensemble.                   |
| BestLearnerSelection  | Orchestra.jl    |                  | accuracy | Selects best learner out of pool. |

## Changes

See [CHANGELOG.yml](CHANGELOG.yml).

## Future Work

See [FUTUREWORK.yml](CHANGELOG.yml).

## Contributing 

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT "Expat" License. See [LICENSE.md](LICENSE.md).
