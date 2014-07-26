# Wrapper to CARET library.
module CaretWrapper

importall Orchestra.Types
importall Orchestra.Util

# Setup R
using PyCall
@pyimport rpy2.robjects as RO
@pyimport rpy2.robjects.packages as RP
@pyimport rpy2.robjects.numpy2ri as N2R
N2R.activate()
RP.importr("caret")

export CRTLearner,
       fit!,
       transform!


# Builds R dataframe out of dataset.
# Returns (dataframe, label_factor_levels).
function dataset_to_r_dataframe(
  instances::Matrix{Float64}; labels=nothing)

  # Build dataframe
  df_dict = Dict()
  for col in 1:size(instances, 2)
    df_dict["X$col"] = RO.FloatVector(instances[:, col])
  end

  if labels == nothing
    return (RO.DataFrame(df_dict), nothing)
  else
    r_labels = RO.FactorVector(labels)
    df_dict["Y"] = r_labels
    return (RO.DataFrame(df_dict), r_labels[:levels])
  end
end


# CARET wrapper that provides access to all learners.
# 
# Options for the specific CARET learner is to be passed
# in `options[:impl_options]` dictionary.
type CRTLearner <: Learner
  model::Dict
  options::Dict
  
  function CRTLearner(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      :learner => "svmLinear",
      :impl_options => Dict()
    }
    new(Dict(), nested_dict_merge(default_options, options)) 
  end
end

function fit!(crtw::CRTLearner,
  instances::Matrix{Float64}, labels::Vector{Float64})

  impl_options = crtw.options[:impl_options]
  crtw.model[:impl] = Dict()
  crtw.model[:impl][:learner] = crtw.options[:learner]

  # Build R dataframe out of dataset
  r_dataset_df, r_label_factors = dataset_to_r_dataframe(
    instances; labels=labels
  )

  # Assign label factors
  crtw.model[:impl][:label_factors] = Float64[
    parsefloat(lf) for lf in collect(r_label_factors)
  ]

  caret_formula = RO.Formula("Y ~ .")
  r_fit_control = pycall(RO.r[:trainControl], PyObject,
    method = "none"
  )
  if isempty(impl_options)
    # Train with default (no grid search)
    r_model = pycall(RO.r[:train], PyObject,
      caret_formula,
      method = crtw.model[:impl][:learner],
      data = r_dataset_df,
      trControl = r_fit_control,
      tuneLength = 1
    )
  else
    # Train with specified learner parameters
    r_model = pycall(RO.r[:train], PyObject,
      caret_formula,
      method = crtw.model[:impl][:learner],
      data = r_dataset_df,
      trControl = r_fit_control,
      tuneGrid = RO.DataFrame(impl_options)
    )
  end
  crtw.model[:impl][:r_model] = r_model

  return crtw
end

function transform!(crtw::CRTLearner, instances::Matrix{Float64})
  r_instance_df, _ = dataset_to_r_dataframe(instances)
  predictions = collect(pycall(RO.r[:predict], PyObject,
    crtw.model[:impl][:r_model],
    newdata = r_instance_df
  ))
  label_factors = crtw.model[:impl][:label_factors]
  predictions = [label_factors[x] for x in predictions]

  return predictions
end

end # module
