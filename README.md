# Orchestra

[![Build Status](https://travis-ci.org/svs14/Orchestra.jl.svg?branch=master)](https://travis-ci.org/svs14/Orchestra.jl)

Orchestra is a heterogeneous ensemble learning package for the Julia programming
language. It is driven by a uniform machine learner API designed for learner
composition.

## Getting Started

We will cover how to predict on a dataset using Orchestra.

### Obtain Data

A tabular dataset will be used to obtain our instances and labels. 

This will be split it into a training and test set using holdout method.

```julia
import RDatasets
using Orchestra.Util
using Orchestra.Transformers

# Obtain instances and labels
dataset = RDatasets.dataset("datasets", "iris")
instances = array(dataset[:, 1:(end-1)])
labels = array(dataset[:, end])

# Split into training and test sets
(train_ind, test_ind) = holdout(size(instances, 1), 0.3)
train_instances = instances[train_ind, :]
test_instances = instances[test_ind, :]
train_labels = labels[train_ind]
test_labels = labels[test_ind]
```

### Create a Learner

A transformer processes instances in some form. Coincidentally, a learner is a subtype of transformer.

A transformer can be created by instantiating it, taking an options dictionary as an optional argument. 

All transformers, including learners are called in the same way.

```julia
# Learner with default settings
learner = PrunedTree()

# Learner with some of the default settings overriden
learner = PrunedTree({
  :impl_options => {
    :purity_threshold => 0.5
  }
})

# All learners are called in the same way.
learner = StackEnsemble({
  :learners => [
    PrunedTree(), 
    RandomForest(),
    DecisionStumpAdaboost()
  ], 
  :stacker => RandomForest()
})
```

### Create a Pipeline

Normally we may require the use of data pre-processing before the instances are passed to the learner.

We shall use a pipeline transformer to chain many transformers in sequence.

In this case we shall one hot encode categorical features, impute NA values and numerically standardize before we call the learner.

```julia
# Create pipeline
pipeline = Pipeline({
  :transformers => [
    OneHotEncoder(), # Encodes nominal features into numeric
    Imputer(), # Imputes NA values
    StandardScaler(), # Standardizes features 
    learner # Predicts labels on instances
  ]
})
```

### Train and Predict

Training is done via the `fit!` function, predicton via `transform!`. 

All transformers, provide these two functions. They are always called the same way.

```julia
# Train
fit!(pipeline, train_instances, train_labels)

# Predict
predictions = transform!(pipeline, test_instances)
```

### Assess

Finally we assess how well our learner performed.

```julia
# Assess predictions
result = score(:accuracy, test_labels, predictions)
```

## Available Transformers

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
