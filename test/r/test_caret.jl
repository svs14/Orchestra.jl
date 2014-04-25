module TestCaretWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

using FactCheck
using Fixtures

using MLBase
importall Orchestra.Types
importall Orchestra.Transformers.CaretWrapper
CW = CaretWrapper
using PyCall
@pyimport rpy2.robjects as RO
@pyimport rpy2.robjects.packages as RP
@pyimport rpy2.robjects.numpy2ri as N2R
N2R.activate()
RP.importr("caret")

function behavior_check(caret_learner::String, impl_options=Dict())
  # Predict with Orchestra learner
  srand(1)
  pycall(RO.r["set.seed"], PyObject, 1)
  learner = CRTLearner({
    :learner => caret_learner, 
    :impl_options => impl_options
  })
  orchestra_predictions = train_and_transform!(learner, nfcp)

  # Predict with backend learner
  srand(1)
  pycall(RO.r["set.seed"], PyObject, 1)
  (r_dataset_df, label_factors) = CW.dataset_to_r_dataframe(
    nfcp.train_instances, nfcp.train_labels
  )
  label_factors = collect(label_factors)
  caret_formula = RO.Formula("Y ~ .")
  r_fit_control = pycall(RO.r[:trainControl], PyObject,
    method = "none"
  )
  if isempty(impl_options)
    r_model = pycall(RO.r[:train], PyObject,
      caret_formula,
      method = caret_learner,
      data = r_dataset_df,
      trControl = r_fit_control,
      tuneLength = 1
    )
  else
    r_model = pycall(RO.r[:train], PyObject,
      caret_formula,
      method = caret_learner,
      data = r_dataset_df,
      trControl = r_fit_control,
      tuneGrid = RO.DataFrame(impl_options)
    )
  end
  (r_instance_df, _) = CW.dataset_to_r_dataframe(nfcp.test_instances)
  original_predictions = collect(pycall(RO.r[:predict], PyObject,
    r_model,
    newdata = r_instance_df
  ))
  original_predictions = map(x -> label_factors[x], original_predictions)

  # Verify same predictions
  @fact orchestra_predictions => original_predictions
end

facts("CARET learners", using_fixtures) do
  context("CRTLearner gives same results as its backend", using_fixtures) do
    caret_learners = ["svmLinear", "nnet", "earth"]
    for caret_learner in caret_learners
      behavior_check(caret_learner)
    end
  end
  context("CRTLearner with options gives same results as its backend", using_fixtures) do
    behavior_check("svmLinear", {:C => 5.0})
  end
end

end # module
