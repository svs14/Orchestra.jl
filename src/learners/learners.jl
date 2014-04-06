# Learner definitions and implementations.
module Learners

export Learner,
       PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       VoteEnsemble, 
       StackEnsemble,
       SVM,
       BestLearnerSelection,
       train!,
       predict!,
       score

# Include abstract learner as convenience
importall Orchestra.AbstractLearner

# Include atomic Julia learners
include(joinpath("julia", "decisiontree.jl"))
importall .DecisionTreeWrapper
include(joinpath("julia", "libsvm.jl"))
importall .LIBSVMWrapper

# Include aggregate learners last, dependent on atomic learners
include(joinpath("julia", "ensemble.jl"))
importall .EnsembleMethods
include(joinpath("julia", "selection.jl"))
importall .SelectionMethods

end # module
