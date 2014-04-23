module TestScikitLearnWrapper

include(joinpath("..", "fixture_learners.jl"))
using .FixtureLearners
nfcp = NumericFeatureClassification()

using FactCheck
using Fixtures

importall Orchestra.AbstractLearner
importall Orchestra.Learners.ScikitLearnWrapper
using PyCall
@pyimport sklearn.neighbors as NN
@pyimport random as RAN

function skl_train_and_predict!(learner::Learner, problem::MLProblem, seed=1)
  RAN.seed(seed)
  return train_and_predict!(learner, problem, seed)
end

function backend_train_and_predict!(sk_learner, seed=1)
  RAN.seed(seed)
  srand(seed)
  sk_learner[:fit](nfcp.train_instances, nfcp.train_labels)
  return collect(sk_learner[:predict](nfcp.test_instances))
end

function behavior_check(learner::Learner, sk_learner)
  # Predict with Orchestra learner
  orchestra_predictions = skl_train_and_predict!(learner, nfcp)

  # Predict with original backend learner
  original_predictions = backend_train_and_predict!(sk_learner)

  # Verify same predictions
  @fact orchestra_predictions => original_predictions
end


facts("scikit-learn learners", using_fixtures) do
  context("SKLWrapper gives same results as its backend", using_fixtures) do
    learner_names = collect(keys(ScikitLearnWrapper.learner_dict))
    for learner_name in learner_names
      sk_learner = ScikitLearnWrapper.learner_dict[learner_name]()
      impl_options = Dict()

      if in(learner_name, ["RandomForestClassifier", "ExtraTreesClassifier"])
        impl_options = {:random_state => 1}
        sk_learner = ScikitLearnWrapper.learner_dict[learner_name](
          random_state = 1
        )
      elseif learner_name == "RadiusNeighborsClassifier"
        outlier_label = nfcp.train_labels[rand(1:size(nfcp.train_labels, 1))]
        impl_options = {:outlier_label => outlier_label}
        sk_learner = NN.RadiusNeighborsClassifier(outlier_label = outlier_label)
      end

      learner = SKLWrapper({
        :learner => learner_name,
        :impl_options => impl_options
      })
      behavior_check(learner, sk_learner)
    end
  end
end

end # module
