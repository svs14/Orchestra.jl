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

| Learner               | Library           | Metrics  | Description                                      |
|-----------------------|-------------------|----------|--------------------------------------------------|
| PrunedTree            | DecisionTree.jl   | accuracy | C4.5 Decision Tree.                              |
| RandomForest          | DecisionTree.jl   | accuracy | C4.5 Random Forest.                              |
| DecisionStumpAdaboost | DecisionTree.jl   | accuracy | C4.5 Adaboosted Decision Stumps.                 |


### Orchestra

| Learner               | Library           | Metrics  | Description                                      |
|-----------------------|-------------------|----------|--------------------------------------------------|
| VoteEnsemble          | Orchestra.jl      | accuracy | Majority Vote Ensemble.                          |
| StackEnsemble         | Orchestra.jl      | accuracy | Stack Ensemble.                                  |
| BestLearnerSelection  | Orchestra.jl      | accuracy | Selects best learner out of pool.                |


### Python

| Learner               | Library           | Metrics  | Description                                      |
|-----------------------|-------------------|----------|--------------------------------------------------|
| SKLRandomForest       | scikit-learn 0.14 | accuracy | Random Forest.                                   |
| SKLExtraTrees         | scikit-learn 0.14 | accuracy | Extra-trees.                                     |
| SKLGradientBoosting   | scikit-learn 0.14 | accuracy | Gradient Boosting Machine.                       |
| SKLLogisticRegression | scikit-learn 0.14 | accuracy | Logistic Regression.                             |
| SKLPassiveAggressive  | scikit-learn 0.14 | accuracy | Passive Aggressive.                              |
| SKLRidge              | scikit-learn 0.14 | accuracy | Ridge classifier.                                |
| SKLRidgeCV            | scikit-learn 0.14 | accuracy | Ridge classifier with in-built Cross Validation. |
| SKLSGD                | scikit-learn 0.14 | accuracy | Linear classifiers with SGD training.            |
| SKLKNeighbors         | scikit-learn 0.14 | accuracy | K Nearest Neighbors                              |
| SKLRadiusNeighbors    | scikit-learn 0.14 | accuracy | Within Radius Neighbors Vote.                    |
| SKLNearestCentroid    | scikit-learn 0.14 | accuracy | Nearest Centroid.                                |
| SKLSVC                | scikit-learn 0.14 | accuracy | C-Support Vector Classifier.                     |
| SKLLinearSVC          | scikit-learn 0.14 | accuracy | Linear Support Vector Classifier.                |
| SKLNuSVC              | scikit-learn 0.14 | accuracy | Nu-Support Vector Classifier.                    |
| SKLDecisionTree       | scikit-learn 0.14 | accuracy | Decision Tree.                                   |


### R

Python library 'rpy2' is required to interface with R.

R library 'caret' offers more than 100 learners. 
See [here](http://caret.r-forge.r-project.org/modelList.html) for more details.

```julia
# Example usage for using CARET.
learner = CRTWrapper({
  :learner => "svmLinear", 
  :impl_options => {:C => 5.0}
})
```

| Learner               | Library           | Metrics  | Description                                      |
|-----------------------|-------------------|----------|--------------------------------------------------|
| CRTWrapper            | caret 6.0         | accuracy | Wrapper to all CARET machine learners.           |

## Known Limitations

Learners have only been tested on instances with numeric features. 

Inconsistencies may result in using nominal features directly without a numeric transformation (i.e. one-hot coding).

## Changes

See [CHANGELOG.yml](CHANGELOG.yml).

## Future Work

See [FUTUREWORK.md](FUTUREWORK.md).

## Contributing 

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT "Expat" License. See [LICENSE.md](LICENSE.md).
