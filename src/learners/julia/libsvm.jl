# Support vector machines as provided by LIBSVM Julia package.
module LIBSVMWrapper

importall Orchestra.AbstractLearner
import LIBSVM

export SVM,
       train!,
       predict!

# Support vector machine.
#
# <pre>
# default_options = {
#     # Metric to train against
#     # (:accuracy).
#     :metric => :accuracy,
#     # Options specific to this implementation.
#     :impl_options => {
#         # Type of support vector machine to use
#         # (CSVC, NuSVC, OneClassSVM, EpsilonSVR, NuSVR).
#         :svm_type => LIBSVM.CSVC,
#         # Kernel type to use
#         # (Linear, Polynomial, RBF, Sigmoid, Precomputed).
#         :kernel_type => LIBSVM.RBF,
#         :degree => 3,
#         :gamma => nothing,
#         :coef0 => 0.0,
#         :C => 1.0,
#         :nu => 0.5,
#         :p => 0.1,
#         :cache_size => 100.0,
#         :eps => 0.001,
#         :shrinking => true,
#         :probability_estimates => false,
#         :weights => nothing,
#         :verbose => false
#     },
# }
# </pre>
type SVM <: Learner
    model
    options
    
    function SVM(options=Dict())
        # TODO(sjenkz): Comment each option.
        default_options = {
            # Metric to train against
            # (:accuracy).
            :metric => :accuracy,
            # Options specific to this implementation.
            :impl_options => {
                # Type of support vector machine to use
                # (CSVC, NuSVC, OneClassSVM, EpsilonSVR, NuSVR).
                :svm_type => LIBSVM.CSVC,
                # Kernel type to use
                # (Linear, Polynomial, RBF, Sigmoid, Precomputed).
                :kernel_type => LIBSVM.RBF,
                :degree => 3,
                :gamma => nothing,
                :coef0 => 0.0,
                :C => 1.0,
                :nu => 0.5,
                :p => 0.1,
                :cache_size => 100.0,
                :eps => 0.001,
                :shrinking => true,
                :probability_estimates => false,
                :weights => nothing,
                :verbose => false
            },
        }
        new(nothing, merge(default_options, options)) 
    end
end


function train!(svm::SVM, instances::Matrix, labels::Vector)
    # Set training-dependent options
    impl_options = copy(svm.options[:impl_options])
    if impl_options[:gamma] == nothing
        impl_options[:gamma] = 1.0 / size(instances, 2)
    end
    # Train model
    svm.model = LIBSVM.svmtrain(
        labels, instances'; impl_options...
    )
end

function predict!(svm::SVM, instances::Matrix)
    # Predict on instances
    # NOTE(sjenkz): instances must be transposed to fit into LIBSVM API.
    predicted_labels, decision_values = LIBSVM.svmpredict(svm.model, instances')

    # Coerce into standardized category label format if labels are strings
    if typeof(predicted_labels) <: Array{ASCIIString,1}
        predicted_labels = convert(Array{Any,1}, predicted_labels)
    end

    # Return predicted labels
    return predicted_labels
end

end # module
