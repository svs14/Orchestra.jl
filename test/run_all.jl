# Run all tests.
module RunAllTests
    using FactCheck

    include("test_abstractlearner.jl")
    include("test_util.jl")
    include("learners/julia/test_decisiontree.jl")
    include("learners/julia/test_ensemble.jl")
    include("learners/julia/test_libsvm.jl")
    include("learners/julia/test_selection.jl")

    exitstatus()
end # module
