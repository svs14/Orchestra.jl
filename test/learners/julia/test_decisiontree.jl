module TestDecisionTreeWrapper

include("fixture_learners.jl")
importall .FixtureLearners

using FactCheck
using Fixtures

importall Orchestra.Learners.DecisionTreeWrapper
using DecisionTree

facts("DecisionTree learners", using_fixtures) do
    context("PrunedTree gives same results as its backend", using_fixtures) do
        # Predict with Orchestra learner
        srand(1)
        learner = PrunedTree()
        train!(learner, train_instances, train_labels)
        orchestra_predictions = predict!(learner, test_instances)

        # Predict with original backend learner
        srand(1)
        model = build_tree(train_labels, train_instances)
        model = prune_tree(model, 1.0)
        original_predictions = apply_tree(model, test_instances)

        # Verify same predictions
        @fact orchestra_predictions => original_predictions
    end

    context("RandomForest gives same results as its backend", using_fixtures) do
        # Predict with Orchestra learner
        srand(1)
        learner = RandomForest()
        train!(learner, train_instances, train_labels)
        orchestra_predictions = predict!(learner, test_instances)

        # Predict with original backend learner
        srand(1)
        model = build_forest(
            train_labels,
            train_instances,
            size(train_instances, 2),
            10,
            0.7
        )
        original_predictions = apply_forest(model, test_instances)

        # Verify same predictions
        @fact orchestra_predictions => original_predictions
    end

    context("DecisionStumpAdaboost gives same results as its backend", using_fixtures) do
        # Predict with Orchestra learner
        srand(1)
        learner = DecisionStumpAdaboost()
        train!(learner, train_instances, train_labels)
        orchestra_predictions = predict!(learner, test_instances)

        # Predict with original backend learner
        srand(1)
        model, coeffs = build_adaboost_stumps(
            train_labels,
            train_instances,
            7
        )
        original_predictions = apply_adaboost_stumps(
            model, coeffs, test_instances
        )

        # Verify same predictions
        @fact orchestra_predictions => original_predictions
    end
end

end # module
