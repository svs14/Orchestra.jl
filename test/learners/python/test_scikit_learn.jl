module TestScikitLearnWrapper

include("../fixture_learners.jl")
importall .FixtureLearners

using FactCheck
using Fixtures

importall Orchestra.Learners.ScikitLearnWrapper
using PyCall
@pyimport sklearn.ensemble as ENS

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
end

end # module
