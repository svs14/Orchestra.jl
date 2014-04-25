module TestDecisionTreeWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

using FactCheck
using Fixtures

importall Orchestra.Learners.DecisionTreeWrapper
using DecisionTree

facts("DecisionTree learners", using_fixtures) do
  context("PrunedTree gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    learner = PrunedTree()
    orchestra_predictions = train_and_transform!(learner, nfcp)

    # Predict with original backend learner
    srand(1)
    model = build_tree(nfcp.train_labels, nfcp.train_instances)
    model = prune_tree(model, 1.0)
    original_predictions = apply_tree(model, nfcp.test_instances)

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("RandomForest gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    learner = RandomForest()
    orchestra_predictions = train_and_transform!(learner, nfcp)

    # Predict with original backend learner
    srand(1)
    model = build_forest(
      nfcp.train_labels,
      nfcp.train_instances,
      size(nfcp.train_instances, 2),
      10,
      0.7
    )
    original_predictions = apply_forest(model, nfcp.test_instances)

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("DecisionStumpAdaboost gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    learner = DecisionStumpAdaboost()
    orchestra_predictions = train_and_transform!(learner, nfcp)

    # Predict with original backend learner
    srand(1)
    model, coeffs = build_adaboost_stumps(
      nfcp.train_labels,
      nfcp.train_instances,
      7
    )
    original_predictions = apply_adaboost_stumps(
      model, coeffs, nfcp.test_instances
    )

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end
end

end # module
