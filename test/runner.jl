# Run all tests.
module TestRunner
  using FactCheck

  include("test_abstractlearner.jl")
  include("test_util.jl")
  include("learners/julia/test_decisiontree.jl")
  include("learners/julia/test_ensemble.jl")
  include("learners/julia/test_selection.jl")
  # include("learners/python/test_scikit_learn.jl")

  exitstatus()
end # module
