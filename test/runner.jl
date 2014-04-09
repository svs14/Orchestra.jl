# Run all tests.
module TestRunner
  using FactCheck
  using Orchestra.System

  include("test_abstractlearner.jl")
  include("test_util.jl")
  include(joinpath("learners", "julia", "test_decisiontree.jl"))
  include(joinpath("learners", "orchestra", "test_ensemble.jl"))
  include(joinpath("learners", "orchestra", "test_selection.jl"))
  if HAS_SKL
    include(joinpath("learners", "python", "test_scikit_learn.jl"))
  else
    info("Skipping scikit-learn tests.")
  end
  if HAS_CRT
    include(joinpath("learners", "r", "test_caret.jl"))
  else
    info("Skipping CARET tests.")
  end
  include("test_system.jl")

  exitstatus()
end # module
