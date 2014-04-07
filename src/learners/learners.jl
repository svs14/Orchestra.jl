# Learner definitions and implementations.
module Learners

export Learner,
       PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       VoteEnsemble, 
       StackEnsemble,
       BestLearnerSelection,
       SKLLearner,
       SKLRandomForest,
       SKLExtraTrees,
       SKLGradientBoosting,
       SKLLogisticRegression,
       SKLPassiveAggressive,
       SKLRidge,
       SKLRidgeCV,
       SKLSGD,
       SKLKNeighbors,
       SKLRadiusNeighbors,
       SKLNearestCentroid,
       SKLSVC,
       SKLLinearSVC,
       SKLNuSVC,
       SKLDecisionTree,
       train!,
       predict!,
       score

# Include abstract learner as convenience
importall Orchestra.AbstractLearner

# Include atomic Julia learners
include(joinpath("julia", "decisiontree.jl"))
importall .DecisionTreeWrapper

# Include atomic Python learners
include(joinpath("python", "scikit_learn.jl"))
importall .ScikitLearnWrapper

# Include aggregate learners last, dependent on atomic learners
include(joinpath("julia", "ensemble.jl"))
importall .EnsembleMethods
include(joinpath("julia", "selection.jl"))
importall .SelectionMethods

end # module
