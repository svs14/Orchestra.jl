# Transformer definitions and implementations.
module Transformers

export Transformer,
       Learner,
       OneHotEncoder,
       Imputer,
       Pipeline,
       Wrapper,
       Identity,
       Baseline,
       PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       StandardScaler,
       PCA,
       VoteEnsemble, 
       StackEnsemble,
       BestLearnerEnsemble,
       SKLLearner,
       CRTLearner,
       fit!,
       transform!

# Obtain system details
import Orchestra.System: HAS_SKL, HAS_CRT

# Include abstract types as convenience
importall Orchestra.Types

# Include atomic Orchestra transformers
include(joinpath("orchestra", "baseline.jl"))
importall .BaselineMethods
include(joinpath("orchestra", "transformers.jl"))
importall .OrchestraTransformers

# Include Julia transformers
include(joinpath("julia", "decisiontree.jl"))
importall .DecisionTreeWrapper
include(joinpath("julia", "mlbase.jl"))
importall .MLBaseWrapper
include(joinpath("julia", "dimensionalityreduction.jl"))
importall .DimensionalityReductionWrapper

# Include Python transformers
if HAS_SKL
  include(joinpath("python", "scikit_learn.jl"))
  importall .ScikitLearnWrapper
end

# Include R transformers
if HAS_CRT
  include(joinpath("r", "caret.jl"))
  importall .CaretWrapper
end

# Include aggregate transformers last, dependent on atomic transformers
include(joinpath("orchestra", "ensemble.jl"))
importall .EnsembleMethods

end # module
