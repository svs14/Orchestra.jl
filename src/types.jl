# Orchestra types.
module Types

import Orchestra.Structures: OCDM

export Transformer,
       Learner,
       TestLearner,
       OCDM,
       fit!,
       transform!

# All transformer types must have implementations 
# of function `fit!` and `transform!`.
abstract Transformer

# Learner abstract type which all machine learners implement.
abstract Learner <: Transformer

# Test learner. 
# Used to separate production learners from test.
abstract TestLearner <: Learner

# Trains transformer on provided instances and labels.
#
# @param transformer Target transformer.
# @param instances Training instances.
# @param labels Training labels.
# @return Mutated target transformer.
function fit!(transformer::Transformer,
  instances::Matrix{Float64}, labels::Vector{Float64})

  error(typeof(transformer), " does not implement fit!")
end
function fit!(transformer::Transformer,
  instances::OCDM, labels::OCDM)

  error(typeof(transformer), " does not implement fit!")
end

# Trains transformer on provided instances and labels.
#
# @param transformer Target transformer.
# @param instances Original instances.
# @return Transformed instances.
function transform!(transformer::Transformer,
  instances::Matrix{Float64})

  error(typeof(transformer), " does not implement transform!")
end
function transform!(transformer::Transformer,
  instances::OCDM)

  error(typeof(transformer), " does not implement transform!")
end

end # module
