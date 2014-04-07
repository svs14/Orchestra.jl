module TestScikitLearnWrapper

include("../fixture_learners.jl")
importall .FixtureLearners

using FactCheck
using Fixtures

importall Orchestra.AbstractLearner
importall Orchestra.Learners.ScikitLearnWrapper
using PyCall
@pyimport sklearn.ensemble as ENS
@pyimport sklearn.linear_model as LM
@pyimport random as RAN
@pyimport sklearn.neighbors as NN


function skl_train_and_predict!(learner::Learner)
  RAN.seed(1)
  return train_and_predict!(learner)
end

function backend_train_and_predict!(sk_learner)
  RAN.seed(1)
  srand(1)
  sk_learner[:fit](train_instances, train_labels)
  return collect(sk_learner[:predict](test_instances))
end

function behavior_check(learner::Learner, sk_learner)
  # Predict with Orchestra learner
  orchestra_predictions = skl_train_and_predict!(learner)

  # Predict with original backend learner
  original_predictions = backend_train_and_predict!(sk_learner)

  # Verify same predictions
  @fact orchestra_predictions => original_predictions
end


facts("scikit-learn learners", using_fixtures) do
  context("SKLRandomForest gives same results as its backend", using_fixtures) do
    learner = SKLRandomForest()
    sk_learner = ENS.RandomForestClassifier()
    behavior_check(learner, sk_learner)
  end

  context("SKLExtraTrees gives same results as its backend", using_fixtures) do
    learner = SKLExtraTrees({:impl_options => {:random_state => 1}})
    sk_learner = ENS.ExtraTreesClassifier(random_state = 1)
    behavior_check(learner, sk_learner)
  end

  context("SKLGradientBoosting gives same results as its backend", using_fixtures) do
    learner = SKLGradientBoosting()
    sk_learner = ENS.GradientBoostingClassifier()
    behavior_check(learner, sk_learner)
  end

  context("SKLLogisticRegression gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    learner = SKLLogisticRegression()
    sk_learner = LM.LogisticRegression()
    behavior_check(learner, sk_learner)
  end

  context("SKLPassiveAggressive gives same results as its backend", using_fixtures) do
    learner = SKLPassiveAggressive()
    sk_learner = LM.PassiveAggressiveClassifier()
    behavior_check(learner, sk_learner)
  end

  context("SKLRidge gives same results as its backend", using_fixtures) do
    learner = SKLRidge()
    sk_learner = LM.RidgeClassifier()
    behavior_check(learner, sk_learner)
  end

  context("SKLRidgeCV gives same results as its backend", using_fixtures) do
    learner = SKLRidgeCV()
    sk_learner = LM.RidgeClassifierCV()
    behavior_check(learner, sk_learner)
  end

  context("SKLSGD gives same results as its backend", using_fixtures) do
    learner = SKLSGD()
    sk_learner = LM.SGDClassifier()
    behavior_check(learner, sk_learner)
  end

  context("SKLKNeighbors gives same results as its backend", using_fixtures) do
    learner = SKLKNeighbors()
    sk_learner = NN.KNeighborsClassifier()
    behavior_check(learner, sk_learner)
  end

  context("SKLRadiusNeighbors gives same results as its backend", using_fixtures) do
    learner = SKLRadiusNeighbors()
    outlier_label = train_labels[rand(1:size(train_labels, 1))]
    sk_learner = NN.RadiusNeighborsClassifier(outlier_label = outlier_label)
    behavior_check(learner, sk_learner)
  end

  context("SKLNearestCentroid gives same results as its backend", using_fixtures) do
    # Predict with Orchestra learner
    learner = SKLNearestCentroid()
    sk_learner = NN.NearestCentroid()
    behavior_check(learner, sk_learner)
  end

end

end # module
