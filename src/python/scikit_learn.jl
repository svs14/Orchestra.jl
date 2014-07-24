# Wrapper module for scikit-learn machine learners.
module ScikitLearnWrapper

importall Orchestra.Types
importall Orchestra.Util

using PyCall
@pyimport sklearn.ensemble as ENS
@pyimport sklearn.linear_model as LM
@pyimport sklearn.lda as LDA_M
@pyimport sklearn.qda as QDA_M
@pyimport sklearn.neighbors as NN
@pyimport sklearn.svm as SVM
@pyimport sklearn.tree as TREE

export SKLLearner,
       fit!,
       transform!

# Available scikit-learn learners.
learner_dict = {
  "AdaBoostClassifier" => ENS.AdaBoostClassifier,
  "BaggingClassifier" => ENS.BaggingClassifier,
  "ExtraTreesClassifier" => ENS.ExtraTreesClassifier,
  "GradientBoostingClassifier" => ENS.GradientBoostingClassifier,
  "RandomForestClassifier" => ENS.RandomForestClassifier,
  "LDA" => LDA_M.LDA,
  "LogisticRegression" => LM.LogisticRegression,
  "PassiveAggressiveClassifier" => LM.PassiveAggressiveClassifier,
  "RidgeClassifier" => LM.RidgeClassifier,
  "RidgeClassifierCV" => LM.RidgeClassifierCV,
  "SGDClassifier" => LM.SGDClassifier,
  "KNeighborsClassifier" => NN.KNeighborsClassifier,
  "RadiusNeighborsClassifier" => NN.RadiusNeighborsClassifier,
  "NearestCentroid" => NN.NearestCentroid,
  "QDA" => QDA_M.QDA,
  "SVC" => SVM.SVC,
  "LinearSVC" => SVM.LinearSVC,
  "NuSVC" => SVM.NuSVC,
  "DecisionTreeClassifier" => TREE.DecisionTreeClassifier
}

# Wrapper for scikit-learn that provides access to most learners.
# 
# Options for the specific scikit-learn learner is to be passed
# in `options[:impl_options]` dictionary.
# 
# Available learners:
#
#   - "AdaBoostClassifier"
#   - "BaggingClassifier"
#   - "ExtraTreesClassifier"
#   - "GradientBoostingClassifier"
#   - "RandomForestClassifier"
#   - "LDA"
#   - "LogisticRegression"
#   - "PassiveAggressiveClassifier"
#   - "RidgeClassifier"
#   - "RidgeClassifierCV"
#   - "SGDClassifier"
#   - "KNeighborsClassifier"
#   - "RadiusNeighborsClassifier"
#   - "NearestCentroid"
#   - "QDA"
#   - "SVC"
#   - "LinearSVC"
#   - "NuSVC"
#   - "DecisionTreeClassifier"
#
type SKLLearner <: Learner
  model::Dict
  options::Dict
  
  function SKLLearner(options=Dict())
    default_options = {
      # Output to train against
      # (:class).
      :output => :class,
      :learner => "LinearSVC",
      # Options specific to this implementation.
      :impl_options => Dict()
    }
    new(Dict(), nested_dict_merge(default_options, options)) 
  end
end

function fit!(sklw::SKLLearner, instances::Matrix, labels::Vector)
  impl_options = copy(sklw.options[:impl_options])
  learner = sklw.options[:learner]
  py_learner = learner_dict[learner]

  # Assign Orchestra-specific defaults if required
  if learner == "RadiusNeighborsClassifier"
    if get(impl_options, :outlier_label, nothing) == nothing
      impl_options[:outlier_label] = labels[rand(1:size(labels, 1))]
    end
  end

  # Train
  sklw.model[:impl] = py_learner(;impl_options...)
  sklw.model[:impl][:fit](instances, labels)

  return sklw
end

function transform!(sklw::SKLLearner, instances::Matrix)
  return collect(sklw.model[:impl][:predict](instances))
end

end # module
