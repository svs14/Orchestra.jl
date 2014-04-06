# Orchestra

[![Build Status](https://travis-ci.org/svs14/Orchestra.jl.png)](https://travis-ci.org/svs14/Orchestra.jl)

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
learner = SVM()
```

### ... More

```julia
learner = RandomForest()
```

### Which is best? Machine decides

```julia
learner = BestLearnerSelection({:learners => [PrunedTree(), SVM(), RandomForest()]})
```

### Why even choose? Majority rules

```julia
learner = VoteEnsemble({:learners => [PrunedTree(), SVM(), RandomForest()]})
```

### A Learner on a Learner? We have to go Deeper

```julia
learner = StackEnsemble({:learners => [PrunedTree(), SVM(), RandomForest()], :stacker => SVM()})
```

### Ensemble of Ensembles of Ensembles

```julia
ensemble_1 = RandomForest()
ensemble_2 = StackEnsemble({:learners => [PrunedTree(), SVM()], :stacker => SVM()})
ensemble_3 = VoteEnsemble({:learners => [ensemble_1, ensemble_2]})
ensemble_4 = VoteEnsemble()
learner = VoteEnsemble({:learners => [ensemble_3, ensemble_4]})
```

### Woah!

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

See [FUTUREWORK.md](FUTUREWORK.md).

## Contributing 

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT "Expat" License. See [LICENSE.md](LICENSE.md).
