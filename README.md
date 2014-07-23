# Orchestra

[![Build Status](https://travis-ci.org/svs14/Orchestra.jl.svg?branch=master)](https://travis-ci.org/svs14/Orchestra.jl)
[![Coverage Status](https://coveralls.io/repos/svs14/Orchestra.jl/badge.png?branch=master)](https://coveralls.io/r/svs14/Orchestra.jl?branch=master)

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
fit!(pipeline, instances[train_ind, :], labels[train_ind])

# Predict
predictions = transform!(pipeline, instances[test_ind, :])
```

### Assess

Finally we assess how well our learner performed.

```julia
# Assess predictions
result = score(:accuracy, labels[test_ind], predictions)
```

## Available Transformers

Outlined are all the transformers currently available via Orchestra.

### Orchestra

#### Baseline (Orchestra.jl Learner)

Baseline learner that by default assigns the most frequent label.
```julia
learner = Baseline({
  # Output to train against
  # (:class).
  :output => :class,
  # Label assignment strategy.
  # Function that takes a label vector and returns the required output.
  :strategy => mode
})
```

#### Identity (Orchestra.jl Transformer)

Identity transformer passes the instances as is.
```julia
transformer = Identity()
```

#### VoteEnsemble (Orchestra.jl Learner)

Set of machine learners that majority vote to decide prediction.
```julia
learner = VoteEnsemble({
  # Output to train against
  # (:class).
  :output => :class,
  # Learners in voting committee.
  :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()]
})
```

#### StackEnsemble (Orchestra.jl Learner)

Ensemble where a 'stack' learner learns on a set of learners' predictions.
```julia
learner = StackEnsemble({
  # Output to train against
  # (:class).
  :output => :class,
  # Set of learners that produce feature space for stacker.
  :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()],
  # Machine learner that trains on set of learners' outputs.
  :stacker => RandomForest(),
  # Proportion of training set left to train stacker itself.
  :stacker_training_proportion => 0.3,
  # Provide original features on top of learner outputs to stacker.
  :keep_original_features => false
})
```

#### BestLearner (Orchestra.jl Learner)

Selects best learner out of set. 
Will perform a grid search on learners if options grid is provided.
```julia
learner = BestLearner({
  # Output to train against
  # (:class).
  :output => :class,
  # Function to return partitions of instance indices.
  :partition_generator => (instances, labels) -> kfold(size(instances, 1), 5),
  # Function that selects the best learner by index.
  # Arg learner_partition_scores is a (learner, partition) score matrix.
  :selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, 2))[2],      
  # Score type returned by score() using respective output.
  :score_type => Real,
  # Candidate learners.
  :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()],
  # Options grid for learners, to search through by BestLearner.
  # Format is [learner_1_options, learner_2_options, ...]
  # where learner_options is same as a learner's options but
  # with a list of values instead of scalar.
  :learner_options_grid => nothing
})
```

#### OneHotEncoder (Orchestra.jl Transformer)

Transforms instances with nominal features into one-hot form 
and coerces the instance matrix to be of element type Float64.
```julia
transformer = OneHotEncoder({
  # Nominal columns
  :nominal_columns => nothing,
  # Nominal column values map. Key is column index, value is list of
  # possible values for that column.
  :nominal_column_values_map => nothing
})
```

#### Imputer (Orchestra.jl Transformer)

Imputes NaN values from Float64 features.
```julia
transformer = Imputer({
  # Imputation strategy.
  # Statistic that takes a vector such as mean or median.
  :strategy => mean
})
```

#### Pipeline (Orchestra.jl Transformer)

Chains multiple transformers in sequence.
```julia
transformer = Pipeline({
  # Transformers as list to chain in sequence.
  :transformers => [OneHotEncoder(), Imputer()],
  # Transformer options as list applied to same index transformer.
  :transformer_options => nothing
})
```

#### Wrapper (Orchestra.jl Transformer)

Wraps around an Orchestra transformer.
```julia
transformer = Wrapper({
  # Transformer to call.
  :transformer => OneHotEncoder(),
  # Transformer options.
  :transformer_options => nothing
})
```


### Julia

#### PrunedTree (DecisionTree.jl Learner)

Pruned ID3 decision tree.
```julia
learner = PrunedTree({
  # Output to train against
  # (:class).
  :output => :class,
  # Options specific to this implementation.
  :impl_options => {
    # Merge leaves having >= purity_threshold combined purity.
    :purity_threshold => 1.0
  }
})
```

#### RandomForest (DecisionTree.jl Learner)

Random forest (C4.5).
```julia
learner = RandomForest({
  # Output to train against
  # (:class).
  :output => :class,
  # Options specific to this implementation.
  :impl_options => {
    # Number of features to train on with trees.
    :num_subfeatures => nothing,
    # Number of trees in forest.
    :num_trees => 10,
    # Proportion of trainingset to be used for trees.
    :partial_sampling => 0.7
  }
})
```

#### DecisionStumpAdaboost (DecisionTree.jl Learner)

Adaboosted C4.5 decision stumps.
```julia
learner = DecisionStumpAdaboost({
  # Output to train against
  # (:class).
  :output => :class,
  # Options specific to this implementation.
  :impl_options => {
    # Number of boosting iterations.
    :num_iterations => 7
  }
})
```

#### PCA (DimensionalityReduction.jl Transformer)

Principal Component Analysis rotation
on features.
Features ordered by maximal variance descending.

Fails if zero-variance feature exists.
```julia
transformer = PCA({
  # Center features
  :center => true,
  # Scale features
  :scale => true
})
```

#### StandardScaler (MLBase.jl Transformer)

Standardizes each feature using (X - mean) / stddev.
Will produce NaN if standard deviation is zero.
```julia
transformer = StandardScaler({
  # Center features
  :center => true,
  # Scale features
  :scale => true
})
```

### Python

See the scikit-learn [API](http://scikit-learn.org/stable/modules/classes.html) for what options are available per learner.

#### SKLLearner (scikit-learn 0.15 Learner)

Wrapper for scikit-learn that provides access to most learners.

Options for the specific scikit-learn learner is to be passed
in `options[:impl_options]` dictionary.

Available learners:

  - "AdaBoostClassifier"
  - "BaggingClassifier"
  - "ExtraTreesClassifier"
  - "GradientBoostingClassifier"
  - "RandomForestClassifier"
  - "LDA"
  - "LogisticRegression"
  - "PassiveAggressiveClassifier"
  - "RidgeClassifier"
  - "RidgeClassifierCV"
  - "SGDClassifier"
  - "KNeighborsClassifier"
  - "RadiusNeighborsClassifier"
  - "NearestCentroid"
  - "QDA"
  - "SVC"
  - "LinearSVC"
  - "NuSVC"
  - "DecisionTreeClassifier"

```julia
learner = SKLLearner({
  # Output to train against
  # (:class).
  :output => :class,
  :learner => "LinearSVC",
  # Options specific to this implementation.
  :impl_options => Dict()
})
```


### R

Python library 'rpy2' is required to interface with R.

R library 'caret' offers more than 100 learners. 
See [here](http://caret.r-forge.r-project.org/modelList.html) for more details.

#### CRTLearner (caret 6.0 Learner)

CARET wrapper that provides access to all learners.

Options for the specific CARET learner is to be passed
in `options[:impl_options]` dictionary.
```julia
learner = CRTLearner({
  # Output to train against
  # (:class).
  :output => :class,
  :learner => "svmLinear",
  :impl_options => Dict()
})
```


## Known Limitations

Learners have only been tested on instances with numeric features. 

Inconsistencies may result in using nominal features directly without a numeric transformation (i.e. OneHotEncoder).

## Misc

The links provided below will only work if you are viewing this in the GitHub repository.

### Changes

See [CHANGELOG.yml](CHANGELOG.yml).

### Future Work

See [FUTUREWORK.md](FUTUREWORK.md).

### Contributing 

See [CONTRIBUTING.md](CONTRIBUTING.md).

### License

MIT "Expat" License. See [LICENSE.md](LICENSE.md).
