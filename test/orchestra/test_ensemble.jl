module TestEnsembleMethods

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = MLProblem(;
  output = :class,
  feature_type = Float64,
  label_type = Float64,
  handle_na = false,
  dataset_type = Matrix{Float64}
)

using FactCheck


importall Orchestra.Transformers.EnsembleMethods
import Orchestra.Transformers.DecisionTreeWrapper: fit!, transform!
import Orchestra.Transformers.DecisionTreeWrapper: PrunedTree
import Orchestra.Transformers.DecisionTreeWrapper: RandomForest
import Orchestra.Transformers.DecisionTreeWrapper: DecisionStumpAdaboost

facts("Ensemble learners") do
  context("VoteEnsemble predicts according to majority") do
    always_1_options = { :label => 1.0 }
    always_2_options = { :label => 2.0 }
    learner = VoteEnsemble({
      :learners => [
        AlwaysSameLabelLearner(always_1_options),
        AlwaysSameLabelLearner(always_1_options),
        AlwaysSameLabelLearner(always_2_options)
      ]
    })
    fit!(learner, nfcp.train_instances, nfcp.train_labels)
    predictions = transform!(learner, nfcp.test_instances)
    expected_predictions = fill(1.0, size(nfcp.test_instances, 1))

    @fact predictions => expected_predictions
  end

  context("StackEnsemble predicts with combined learners") do
    # Fix random seed, due to stochasticity in stacker.
    srand(2)

    always_1_options = { :label => 1.0 }
    learner = StackEnsemble({
      :learners => [
        AlwaysSameLabelLearner(always_1_options),
        AlwaysSameLabelLearner(always_1_options),
        PerfectScoreLearner()
      ],
      :keep_original_features => true
    })
    fit!(learner, nfcp.train_instances, nfcp.train_labels)
    predictions = transform!(learner, nfcp.test_instances)
    unexpected_predictions = fill(1.0, size(nfcp.test_instances, 1))

    @fact predictions => not(unexpected_predictions)
  end

  context("BestLearner picks the best learner") do
    always_1_options = { :label => 1.0 }
    always_2_options = { :label => 2.0 }
    learner = BestLearner({
      :learners => [
        AlwaysSameLabelLearner(always_1_options),
        PerfectScoreLearner(),
        AlwaysSameLabelLearner(always_2_options)
      ]
    })
    fit!(learner, nfcp.train_instances, nfcp.train_labels)

    @fact learner.model[:impl][:best_learner_index] => 2
  end

  context("BestLearner conducts grid search") do
    learner = BestLearner({
      :learners => [PrunedTree(), DecisionStumpAdaboost(), RandomForest()],
      :learner_options_grid => [
        {
          :impl_options => {
            :purity_threshold => [0.5, 1.0]
          }
        },
        Dict(),
        {
          :impl_options => {
            :num_trees => [5, 10, 20], 
            :partial_sampling => [0.5, 0.7]
          }
        }
      ]
    })
    fit!(learner, nfcp.train_instances, nfcp.train_labels)

    @fact length(learner.model[:impl][:learners]) => 8
  end
end

end # module
