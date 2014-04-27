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
using Orchestra.Transformers

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
fit!(learner, train_instances, train_labels)
predictions = transform!(learner, test_instances)
result = score(:accuracy, test_labels, predictions)
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
learner = BestLearner({
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

| Learner               | Library           | Outputs  | Description                                      |
|-----------------------|-------------------|----------|--------------------------------------------------|
| PrunedTree            | DecisionTree.jl   | class    | C4.5 Decision Tree.                              |
| RandomForest          | DecisionTree.jl   | class    | C4.5 Random Forest.                              |
| DecisionStumpAdaboost | DecisionTree.jl   | class    | C4.5 Adaboosted Decision Stumps.                 |


### Orchestra

| Learner               | Library           | Outputs  | Description                                      |
|-----------------------|-------------------|----------|--------------------------------------------------|
| VoteEnsemble          | Orchestra.jl      | class    | Majority Vote Ensemble.                          |
| StackEnsemble         | Orchestra.jl      | class    | Stack Ensemble.                                  |
| BestLearner           | Orchestra.jl      | class    | Selects best learner out of pool.                |


### Python

Most classifiers are available from scikit-learn.

Orchestra accessible learners are listed [here](src/learners/python/scikit_learn.jl).
See the scikit-learn [API](http://scikit-learn.org/stable/modules/classes.html) for what options are available per learner.

```julia
# Example usage for using scikit-learn.
learner = SKLLearner({
  :learner => "RandomForestClassifier", 
  :impl_options => {:max_depth => 3}
})
```

| Learner               | Library           | Outputs  | Description                                      |
|-----------------------|-------------------|----------|--------------------------------------------------|
| SKLLearner            | scikit-learn 0.14 | class    | Wrapper to most scikit-learn machine learners.   |


### R

Python library 'rpy2' is required to interface with R.

R library 'caret' offers more than 100 learners. 
See [here](http://caret.r-forge.r-project.org/modelList.html) for more details.

```julia
# Example usage for using CARET.
learner = CRTLearner({
  :learner => "svmLinear", 
  :impl_options => {:C => 5.0}
})
```

| Learner               | Library           | Outputs  | Description                                      |
|-----------------------|-------------------|----------|--------------------------------------------------|
| CRTLearner            | caret 6.0         | class    | Wrapper to all CARET machine learners.           |

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
