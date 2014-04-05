module TestLIBSVMWrapper

include("fixture_learners.jl")
importall .FixtureLearners

using FactCheck
using Fixtures

importall Orchestra.Learners.LIBSVMWrapper
using LIBSVM

facts("SVM learners", using_fixtures) do
    context("SVM gives same results as its backend", using_fixtures) do
        # Predict with Orchestra learner
        srand(1)
        learner = SVM()
        train!(learner, train_instances, train_labels)
        orchestra_predictions = predict!(learner, test_instances)

        # Predict with original backend learner
        srand(1)
        model = LIBSVM.svmtrain(
            train_labels, train_instances'
        )
        original_predictions, _ = LIBSVM.svmpredict(model, test_instances')
        if typeof(original_predictions) <: Array{ASCIIString,1}
            original_predictions = convert(Array{Any,1}, original_predictions)
        end

        # Verify same predictions
        @fact orchestra_predictions => original_predictions
    end
end

end # module
