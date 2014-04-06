module TestScikitLearnWrapper

include("../fixture_learners.jl")
importall .FixtureLearners

using FactCheck
using Fixtures

importall Orchestra.Learners.ScikitLearnWrapper
using PyCall
@pyimport sklearn.ensemble as ENS
@pyimport sklearn.linear_model as LM

facts("scikit-learn learners", using_fixtures) do
  context("SKLRandomForest gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    learner = SKLRandomForest()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    model = ENS.RandomForestClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLExtraTrees gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    learner = SKLExtraTrees()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    model = ENS.ExtraTreesClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLGradientBoosting gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    learner = SKLGradientBoosting()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    model = ENS.GradientBoostingClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLLogisticRegression gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    learner = SKLLogisticRegression()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    model = LM.LogisticRegression()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

end

end # module
