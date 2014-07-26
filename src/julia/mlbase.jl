# MLBase transformers.
module MLBaseWrapper

importall Orchestra.Types
importall Orchestra.Util

import MLBase: Standardize, estimate, transform

export StandardScaler,
       fit!,
       transform!

# Standardizes each feature using (X - mean) / stddev.
# Will produce NaN if standard deviation is zero.
type StandardScaler <: Transformer
  model::Dict
  options::Dict

  function StandardScaler(options=Dict())
    default_options = {
      # Center features
      :center => true,
      # Scale features
      :scale => true
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(st::StandardScaler,
  instances::Matrix{Float64}, labels::Vector{Float64})

  st_transform = estimate(Standardize, instances'; st.options...)
  st.model[:impl] = {
    :standardize_transform => st_transform
  }

  return st
end

function transform!(st::StandardScaler, instances::Matrix{Float64})
  st_transform = st.model[:impl][:standardize_transform]
  transposed_instances = instances'
  return transform(st_transform, transposed_instances)'
end

end # module
