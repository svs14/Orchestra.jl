# Transformers provided by Orchestra.
module OrchestraTransformers

importall Orchestra.Types
import Orchestra.Util: infer_eltype

export OneHotEncoder,
       Imputer,
       Pipeline,
       Wrapper,
       fit!,
       transform!

# Transforms instances with nominal features into one-hot form 
# and coerces the instance matrix to be of element type FloatingPoint.
#
# <pre>
# default_options = {
#   # Nominal columns
#   :nominal_columns => nothing,
#   # Nominal column values map. Key is column index, value is list of
#   # possible values for that column.
#   :nominal_column_values_map => nothing
# }
# </pre>
type OneHotEncoder <: Transformer
  model
  options

  function OneHotEncoder(options=Dict())
    default_options = {
      # Nominal columns
      :nominal_columns => nothing,
      # Nominal column values map. Key is column index, value is list of
      # possible values for that column.
      :nominal_column_values_map => nothing
    }
    new(nothing, merge(default_options, options))
  end
end

function fit!(ohe::OneHotEncoder, instances::Matrix, labels::Vector)
  # Obtain nominal columns
  nominal_columns = ohe.options[:nominal_columns]
  if nominal_columns == nothing
    nominal_columns = find_nominal_columns(instances)
  end

  # Obtain unique values for each nominal column
  nominal_column_values_map = ohe.options[:nominal_column_values_map]
  if nominal_column_values_map == nothing
    nominal_column_values_map = Dict{Int, Any}()
    for column in nominal_columns
      nominal_column_values_map[column] = unique(instances[:, column])
    end
  end

  # Create model
  ohe.model = {
    :nominal_columns => nominal_columns,
    :nominal_column_values_map => nominal_column_values_map
  }
end

function transform!(ohe::OneHotEncoder, instances::Matrix)
  nominal_columns = ohe.model[:nominal_columns]
  nominal_column_values_map = ohe.model[:nominal_column_values_map]

  # Create new transformed instance matrix of type FloatingPoint
  num_rows = size(instances, 1)
  num_columns = 
    (size(instances, 2) - length(nominal_columns)) + 
    sum(map(x -> length(x), values(nominal_column_values_map)))
  transformed_instances = zeros(FloatingPoint, num_rows, num_columns)

  # Fill transformed instance matrix
  col_start_index = 1
  for column in 1:size(instances, 2)
    if !in(column, nominal_columns)
      transformed_instances[:, col_start_index] = instances[:, column]
      col_start_index += 1
    else
      col_values = nominal_column_values_map[column]
      for row in 1:size(instances, 1)
        entry_value = instances[row, column]
        entry_value_index = findfirst(col_values, entry_value)
        if entry_value_index == 0
          warn("Unseen value found in OneHotEncoder,
                for entry ($row, $column) = $(entry_value). 
                Patching value to $(col_values[1]).")
          entry_value_index = 1
        end
        entry_column = (col_start_index - 1) + entry_value_index
        transformed_instances[row, entry_column] = 1
      end
      col_start_index += length(nominal_column_values_map[column])
    end
  end

  return transformed_instances
end

# Finds all nominal columns.
# 
# Nominal columns are those that do not have Real type nor
# do all their elements correspond to Real.
function find_nominal_columns(instances::Matrix)
  nominal_columns = Int[]
  for column in 1:size(instances, 2)
    col_eltype = infer_eltype(instances[:, column])
    if !issubtype(col_eltype, Real)
      push!(nominal_columns, column)
    end
  end
  return nominal_columns
end


# Imputes NaN values from FloatingPoint features.
#
# <pre>
# default_options = {
#   # Imputation strategy.
#   # Statistic that takes a vector such as mean or median.
#   :strategy => mean
# }
# </pre>
type Imputer <: Transformer
  model
  options

  function Imputer(options=Dict())
    default_options = {
      # Imputation strategy.
      # Statistic that takes a vector such as mean or median.
      :strategy => mean
    }
    new(nothing, merge(default_options, options))
  end
end

function fit!(imp::Imputer, instances::Matrix, labels::Vector)
  imp.model = imp.options
end

function transform!(imp::Imputer, instances::Matrix)
  new_instances = copy(instances)
  strategy = imp.model[:strategy]

  for column in 1:size(instances, 2)
    column_values = instances[:, column]
    col_eltype = infer_eltype(column_values)

    if issubtype(col_eltype, Real)
      na_rows = map(x -> isnan(x), column_values)
      if any(na_rows)
        fill_value = strategy(column_values[!na_rows])
        new_instances[na_rows, column] = fill_value
      end
    end
  end

  return new_instances
end


# Chains multiple transformers in sequence.
#
# <pre>
# default_options = {
#   # Transformers to chain in sequence.
#   :transformers => [OneHotEncoder(), Imputer(), StandardScaler()],
#   # Transformer options applied to same index transformer.
#   :transformer_options => []
# }
# </pre>
type Pipeline <: Transformer
  model
  options

  function Pipeline(options=Dict())
    default_options = {
      # Transformers as list to chain in sequence.
      :transformers => [OneHotEncoder(), Imputer()],
      # Transformer options as list applied to same index transformer.
      :transformer_options => nothing
    }
    new(nothing, merge(default_options, options))
  end
end

# NOTE(svs14): Method is only idempotent if the same transformer_options
#              are overriden at each subsequent fit! call.
function fit!(pipe::Pipeline, instances::Matrix, labels::Vector)
  transformers = pipe.options[:transformers]
  transformer_options = pipe.options[:transformer_options]

  current_instances = instances
  for t_index in 1:length(transformers)
    transformer = transformers[t_index]
    if transformer_options != nothing
      merge!(transformer.options, transformer_options[t_index])
    end
    fit!(transformer, current_instances, labels)
    current_instances = transform!(transformer, current_instances)
  end

  pipe.model = {
      :transformers => transformers,
      :transformer_options => transformer_options
  }
end

function transform!(pipe::Pipeline, instances::Matrix)
  transformers = pipe.model[:transformers]

  current_instances = instances
  for t_index in 1:length(transformers)
    transformer = transformers[t_index]
    current_instances = transform!(transformer, current_instances)
  end

  return current_instances
end


# Wraps around an Orchestra transformer.
#
# <pre>
# default_options = {
#   # Transformer to call.
#   :transformer => OneHotEncoder(),
#   # Transformer options.
#   :transformer_options => nothing
# }
# </pre>
type Wrapper <: Transformer
  model
  options

  function Wrapper(options=Dict())
    default_options = {
      # Transformer to call.
      :transformer => OneHotEncoder(),
      # Transformer options.
      :transformer_options => nothing
    }
    new(nothing, merge(default_options, options))
  end
end

# NOTE(svs14): Method is only idempotent if the same transformer_options
#              are overriden at each subsequent fit! call.
function fit!(wrapper::Wrapper, instances::Matrix, labels::Vector)
  transformer = wrapper.options[:transformer]
  transformer_options = wrapper.options[:transformer_options]

  if transformer_options != nothing
    merge!(transformer.options, transformer_options)
  end
  fit!(transformer, instances, labels)

  wrapper.model = {
      :transformer => transformer,
      :transformer_options => transformer_options
  }
end

function transform!(wrapper::Wrapper, instances::Matrix)
  transformer = wrapper.model[:transformer]
  return transform!(transformer, instances)
end

end # module
