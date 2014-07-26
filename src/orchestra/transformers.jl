# Transformers provided by Orchestra.
module OrchestraTransformers

importall Orchestra.Types
importall Orchestra.Util

export OneHotEncoder,
       Imputer,
       Pipeline,
       Wrapper,
       fit!,
       transform!

# Transforms instances with nominal features into one-hot form.
type OneHotEncoder <: Transformer
  model::Dict
  options::Dict

  function OneHotEncoder(options=Dict())
    default_options = {
      # Nominal columns
      :nominal_columns => nothing,
      # Nominal column values map. Key is column index, value is list of
      # possible values for that column.
      :nominal_column_values_map => nothing
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(ohe::OneHotEncoder,
  instances::Matrix{Float64}, labels::Vector{Float64})

  # Obtain nominal columns
  nominal_columns = ohe.options[:nominal_columns]
  if nominal_columns == nothing
    nominal_columns = Int[]
  end

  # Obtain unique values for each nominal column
  nominal_column_values_map = ohe.options[:nominal_column_values_map]
  if nominal_column_values_map == nothing
    nominal_column_values_map = Dict{Int, Array}()
    for column in nominal_columns
      nominal_column_values_map[column] = unique(instances[:, column])
    end
  end

  # Create model
  ohe.model[:impl] = {
    :nominal_columns => nominal_columns,
    :nominal_column_values_map => nominal_column_values_map
  }

  return ohe
end

function transform!(ohe::OneHotEncoder,
  instances::Matrix{Float64})

  nominal_columns = ohe.model[:impl][:nominal_columns]
  nominal_column_values_map = ohe.model[:impl][:nominal_column_values_map]

  # Create new transformed instance matrix of type Float64
  num_rows = size(instances, 1)
  num_columns = (size(instances, 2) - length(nominal_columns)) 
  if !isempty(nominal_column_values_map)
    num_columns += sum(map(x -> length(x), values(nominal_column_values_map)))
  end
  transformed_instances = zeros(Float64, num_rows, num_columns)

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


# Imputes NaN values from Float64 features.
type Imputer <: Transformer
  model::Dict
  options::Dict

  function Imputer(options=Dict())
    default_options = {
      # Imputation strategy.
      # Statistic that takes a vector such as mean or median.
      :strategy => mean
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(imp::Imputer,
  instances::Matrix{Float64}, labels::Vector{Float64})

  imp.model[:impl] = imp.options

  return imp
end

function transform!(imp::Imputer, instances::Matrix{Float64})
  new_instances = copy(instances)
  strategy = imp.model[:impl][:strategy]

  # Iterate through columns and impute using strategy function
  for column in 1:size(instances, 2)
    column_values = instances[:, column]

    na_rows = [isnan(x) for x in column_values]
    if any(na_rows)
      fill_value = strategy(column_values[!na_rows])
      new_instances[na_rows, column] = fill_value
    end
  end

  return new_instances
end


# Chains multiple transformers in sequence.
type Pipeline <: Transformer
  model::Dict
  options::Dict

  function Pipeline(options=Dict())
    default_options = {
      # Transformers as list to chain in sequence.
      :transformers => [OneHotEncoder(), Imputer()],
      # Transformer options as list applied to same index transformer.
      :transformer_options => nothing
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(pipe::Pipeline,
  instances::Matrix{Float64}, labels::Vector{Float64})

  transformers = pipe.options[:transformers]
  transformer_options = pipe.options[:transformer_options]

  # Iterate through transformers
  # Train on previous tranformer's outputs.
  current_instances = instances
  new_transformers = Transformer[]
  for t_index in 1:length(transformers)
    transformer = create_transformer(transformers[t_index], transformer_options)
    push!(new_transformers, transformer)
    fit!(transformer, current_instances, labels)
    current_instances = transform!(transformer, current_instances)
  end

  pipe.model[:impl] = {
    :transformers => new_transformers,
    :transformer_options => transformer_options
  }

  return pipe
end

function transform!(pipe::Pipeline, instances::Matrix{Float64})
  transformers = pipe.model[:impl][:transformers]

  # Transform on previous tranformer's outputs.
  current_instances = instances
  for t_index in 1:length(transformers)
    transformer = transformers[t_index]
    current_instances = transform!(transformer, current_instances)
  end

  return current_instances
end


# Wraps around an Orchestra transformer.
type Wrapper <: Transformer
  model::Dict
  options::Dict

  function Wrapper(options=Dict())
    default_options = {
      # Transformer to call.
      :transformer => OneHotEncoder(),
      # Transformer options.
      :transformer_options => nothing
    }
    new(Dict(), nested_dict_merge(default_options, options))
  end
end

function fit!(wrapper::Wrapper,
  instances::Matrix{Float64}, labels::Vector{Float64})

  # Create transformer
  transformer_options = wrapper.options[:transformer_options]
  transformer = create_transformer(
    wrapper.options[:transformer],
    transformer_options
  )
  if transformer_options != nothing
    transformer_options = 
      nested_dict_merge(transformer.options, transformer_options)
  end

  # Train transformer
  fit!(transformer, instances, labels)

  wrapper.model[:impl] = {
    :transformer => transformer,
    :transformer_options => transformer_options
  }

  return wrapper
end

function transform!(wrapper::Wrapper, instances::Matrix{Float64})
  transformer = wrapper.model[:impl][:transformer]
  return transform!(transformer, instances)
end

end # module
