# Baseline methods.
module BaselineMethods

importall Orchestra.Types
importall Orchestra.Util
import StatsBase: mode

export Baseline,
       Identity,
       fit!,
       transform!

# Baseline learner that by default assigns the most frequent label.
type Baseline <: Learner
  model::Dict
  options::Dict

  function Baseline(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Label assignment strategy.
      # Function that takes a label vector and returns the required output.
      :strategy => mode
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(bl::Baseline, instances::Matrix{Float64}, labels::Vector{Float64})
  bl.model[:impl] = bl.options[:strategy](labels)

  return bl
end

function transform!(bl::Baseline, instances::Matrix{Float64})
  return fill(bl.model[:impl], size(instances, 1))
end


# Identity transformer passes the instances as is.
type Identity <: Transformer
  model::Dict
  options::Dict

  function Identity(options=Dict())
    default_options = Dict()
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(id::Identity, instances::Matrix{Float64}, labels::Vector{Float64})
  return id
end

function transform!(id::Identity, instances::Matrix{Float64})
  return instances
end

end # module
