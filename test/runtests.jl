# Run all tests.
module TestRunner
  using FactCheck
  using Orchestra.System

  include("test_util.jl")
  include(joinpath("orchestra", "test_transformers.jl"))
  include(joinpath("julia", "test_decisiontree.jl"))
  include(joinpath("julia", "test_mlbase.jl"))
  include(joinpath("julia", "test_dimensionalityreduction.jl"))
  include(joinpath("orchestra", "test_ensemble.jl"))
  if LIB_SKL_AVAILABLE
    include(joinpath("python", "test_scikit_learn.jl"))
  else
    info("Skipping scikit-learn tests.")
  end
  if LIB_CRT_AVAILABLE
    include(joinpath("r", "test_caret.jl"))
  else
    info("Skipping CARET tests.")
  end
  include("test_system.jl")

  exitstatus()
end # module
