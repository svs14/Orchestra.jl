module TestScikitLearnWrapper

include("../fixture_learners.jl")
importall .FixtureLearners

using FactCheck
using Fixtures

importall Orchestra.Learners.ScikitLearnWrapper
using PyCall
@pyimport sklearn.ensemble as ENS
@pyimport sklearn.linear_model as LM
@pyimport random as RAN
@pyimport sklearn.neighbors as NN

facts("scikit-learn learners", using_fixtures) do
  context("SKLRandomForest gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLRandomForest()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = ENS.RandomForestClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLExtraTrees gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLExtraTrees({:impl_options => {:random_state => 1}})
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = ENS.ExtraTreesClassifier(random_state = 1)
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLGradientBoosting gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLGradientBoosting()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = ENS.GradientBoostingClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLLogisticRegression gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
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

  context("SKLPassiveAggressive gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLPassiveAggressive()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = LM.PassiveAggressiveClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLRidge gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLRidge()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = LM.RidgeClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLRidgeCV gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLRidgeCV()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = LM.RidgeClassifierCV()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLSGD gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLSGD()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = LM.SGDClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLKNeighbors gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLKNeighbors()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = NN.KNeighborsClassifier()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLRadiusNeighbors gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLRadiusNeighbors()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    outlier_label = train_labels[rand(1:size(train_labels, 1))]
    model = NN.RadiusNeighborsClassifier(outlier_label = outlier_label)
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

  context("SKLNearestCentroid gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    srand(1)
    RAN.seed(1)
    learner = SKLNearestCentroid()
    train!(learner, train_instances, train_labels)
    orchestra_predictions = predict!(learner, test_instances)

    # Predict with original backend learner
    srand(1)
    RAN.seed(1)
    model = NN.NearestCentroid()
    model[:fit](train_instances, train_labels)
    original_predictions = collect(model[:predict](test_instances))

    # Verify same predictions
    @fact orchestra_predictions => original_predictions
  end

end

end # module
