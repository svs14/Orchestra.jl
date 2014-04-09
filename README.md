# Orchestra

[![Build Status](https://travis-ci.org/svs14/Orchestra.jl.svg?branch=master)](https://travis-ci.org/svs14/Orchestra.jl)

Orchestra is a heterogeneous ensemble learning package for the Julia programming
language. It is driven by a uniform machine learner API designed for learner
composition.

## Tutorial

### Get the Data

```julia
import RDatasets
using Orchestra.Util
using Orchestra.Learners

dataset = RDatasets.dataset("datasets", "iris")
instances = array(dataset[:, 1:(end-1)])
labels = array(dataset[:, end])
(train_ind, test_ind) = holdout(size(instances, 1), 0.3)
train_instances = instances[train_ind, :]
test_instances = instances[test_ind, :]
train_labels = labels[train_ind]
test_labels = labels[test_ind]
```

### Try a Learner

```julia
learner = PrunedTree()
train!(learner, train_instances, train_labels)
predictions = predict!(learner, test_instances)
result = score(learner, test_instances, test_labels, predictions)
```

### Try another Learner

```julia
learner = DecisionStumpAdaboost()
```

### ... More

```julia
learner = RandomForest()
```

### Which is best? Machine decides

```julia
learner = BestLearnerSelection({
  :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()]
})
```

### Why even choose? Majority rules

```julia
learner = VoteEnsemble({
  :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()]
})
```

### A Learner on a Learner? We have to go Deeper

```julia
learner = StackEnsemble({
    :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()], 
    :stacker => DecisionStumpAdaboost()
})
```

### Ensemble of Ensembles of Ensembles

```julia
ensemble_1 = RandomForest()
ensemble_2 = StackEnsemble({
  :learners => [PrunedTree(), DecisionStumpAdaboost()], 
  :stacker => DecisionStumpAdaboost()
})
ensemble_3 = VoteEnsemble({:learners => [ensemble_1, ensemble_2]})
ensemble_4 = VoteEnsemble()
learner = VoteEnsemble({:learners => [ensemble_3, ensemble_4]})
```

### Woah!

## Available Learners

### Julia

### Orchestra

### Python


### R


| Learner               | Library         | Constraints | Metrics  | Description                                      |
|-----------------------|-----------------|-------------|----------|--------------------------------------------------|
| PrunedTree            | DecisionTree.jl |             | accuracy | C4.5 Decision Tree.                              |
| RandomForest          | DecisionTree.jl |             | accuracy | C4.5 Random Forest.                              |
| DecisionStumpAdaboost | DecisionTree.jl |             | accuracy | C4.5 Adaboosted Decision Stumps.                 |
| VoteEnsemble          | Orchestra.jl    |             | accuracy | Majority Vote Ensemble.                          |
| StackEnsemble         | Orchestra.jl    |             | accuracy | Stack Ensemble.                                  |
| BestLearnerSelection  | Orchestra.jl    |             | accuracy | Selects best learner out of pool.                |
| SKLRandomForest       | scikit-learn    |             | accuracy | Random Forest.                                   |
| SKLExtraTrees         | scikit-learn    |             | accuracy | Extra-trees.                                     |
| SKLGradientBoosting   | scikit-learn    |             | accuracy | Gradient Boosting Machine.                       |
| SKLLogisticRegression | scikit-learn    |             | accuracy | Logistic Regression.                             |
| SKLPassiveAggressive  | scikit-learn    |             | accuracy | Passive Aggressive.                              |
| SKLRidge              | scikit-learn    |             | accuracy | Ridge classifier.                                |
| SKLRidgeCV            | scikit-learn    |             | accuracy | Ridge classifier with in-built Cross Validation. |
| SKLSGD                | scikit-learn    |             | accuracy | Linear classifiers with SGD training.            |
| SKLKNeighbors         | scikit-learn    |             | accuracy | K Nearest Neighbors                              |
| SKLRadiusNeighbors    | scikit-learn    |             | accuracy | Within Radius Neighbors Vote.                    |
| SKLNearestCentroid    | scikit-learn    |             | accuracy | Nearest Centroid.                                |
| SKLSVC                | scikit-learn    |             | accuracy | C-Support Vector Classifier.                     |
| SKLLinearSVC          | scikit-learn    |             | accuracy | Linear Support Vector Classifier.                |
| SKLNuSVC              | scikit-learn    |             | accuracy | Nu-Support Vector Classifier.                    |
| SKLDecisionTree       | scikit-learn    |             | accuracy | Decision Tree.                                   |
## Changes

See [CHANGELOG.yml](CHANGELOG.yml).

## Future Work

See [FUTUREWORK.md](FUTUREWORK.md).

## Contributing 

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT "Expat" License. See [LICENSE.md](LICENSE.md).
