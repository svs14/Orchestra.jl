# Baseline methods.
module BaselineMethods

importall Orchestra.Types
import Stats: mode

export Baseline,
       Identity,
       fit!,
       transform!

# Baseline learner.
#
# <pre>
# default_options = {
#   # Label assignment strategy.
#   # Function that takes a label vector and returns the required output.
#   :strategy => mode
# }
# </pre>
type Baseline <: Learner
  model
  options

  function Baseline(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      # Label assignment strategy.
      # Function that takes a label vector and returns the required output.
      :strategy => mode
    }
    new(nothing, merge(default_options, options))
  end
end

function fit!(bl::Baseline, instances::Matrix, labels::Vector)
  bl.model = bl.options[:strategy](labels)
end

function transform!(bl::Baseline, instances::Matrix)
  return fill(bl.model, size(instances, 1))
end


# Identity transformer passes the instances as is.
type Identity <: Transformer
  model
  options

  function Identity(options=Dict())
    default_options = Dict{Symbol, Any}()
    new(nothing, merge(default_options, options))
  end
end

function fit!(id::Identity, instances::Matrix, labels::Vector)
  nothing
end

function transform!(id::Identity, instances::Matrix)
  return instances
end

end # module
