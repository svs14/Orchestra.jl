# Orchestra module.
module Orchestra

# Load source files
include("system.jl")
include("abstractlearner.jl")
include("util.jl")
include(joinpath("learners", "learners.jl"))

end # module
