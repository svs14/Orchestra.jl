# Dimensionality Reduction transformers.
module DimensionalityReductionWrapper

importall Orchestra.Types
importall Orchestra.Util
import DimensionalityReduction: pca

export PCA,
       fit!,
       transform!

# Principal Component Analysis rotation
# on features.
# Features ordered by maximal variance descending.
#
# Fails if zero-variance feature exists.
type PCA <: Transformer
  model::Dict
  options::Dict

  function PCA(options=Dict())
    default_options = {
      # Center features
      :center => true,
      # Scale features
      :scale => true
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(p::PCA,instances::Matrix{Float64}, labels::Vector{Float64})
  pca_model = pca(instances; p.options...)
  p.model[:impl] = pca_model

  return p
end

function transform!(p::PCA, instances::Matrix{Float64})
  return instances * p.model[:impl].rotation
end

end # module
