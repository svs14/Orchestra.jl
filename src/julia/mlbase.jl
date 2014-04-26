# MLBase Transformers.
module MLBaseWrapper

importall Orchestra.Types
import MLBase: Standardize, estimate, transform

export Standardizer,
       fit!,
       transform!

# Standardizes each feature using (X - mean) / stddev.
# Will produce NaN if standard deviation is zero.
#
# <pre>
# default_options = {
#   :center => true,
#   :scale => true
# }
# </pre>
type Standardizer <: Transformer
  model
  options

  function Standardizer(options=Dict())
    default_options = {
      :center => true,
      :scale => true
    }
    new(nothing, merge(default_options, options))
  end
end

function fit!(st::Standardizer, instances::Matrix, labels::Vector)
  st_transform = estimate(Standardize, instances'; st.options...)
  st.model = {
    :standardize_transform => st_transform
  }
end

function transform!(st::Standardizer, instances::Matrix)
  st_transform = st.model[:standardize_transform]
  transposed_instances = instances'
  return transform(st_transform, transposed_instances)'
end

end # module
