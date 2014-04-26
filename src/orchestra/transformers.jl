# Transformers provided by Orchestra.
module OrchestraTransformers

importall Orchestra.Types
import Orchestra.Util: infer_eltype

export OneHotEncoder,
       fit!,
       transform!

# Transforms instances with nominal features into one-hot form 
# and coerces the instance matrix to be of element type Real.
#
# <pre>
# default_options = {
#   # Nominal columns
#   :nominal_columns => nothing,
#   # Nominal column values map
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

  # Create new transformed instance matrix of type Real
  num_rows = size(instances, 1)
  num_columns = 
    (size(instances, 2) - length(nominal_columns)) + 
    sum(map(x -> length(x), values(nominal_column_values_map)))
  transformed_instances = zeros(Real, num_rows, num_columns)

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

end # module
