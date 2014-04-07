# Run all tests.
module TestRunner
  using FactCheck

  include("test_abstractlearner.jl")
  include("test_util.jl")
  include(joinpath("learners", "julia", "test_decisiontree.jl"))
  include(joinpath("learners", "orchestra", "test_ensemble.jl"))
  include(joinpath("learners", "orchestra", "test_selection.jl"))
  include(joinpath("learners", "python", "test_scikit_learn.jl"))

  exitstatus()
end # module
