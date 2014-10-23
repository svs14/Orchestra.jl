# Conversion utilities.
module Conversion

using Orchestra.Util
using Orchestra.Structures
using Orchestra.Types

import DataFrames: DataArray, DataFrame, PooledDataArray, complete_cases
import DataFrames: isna, pool, eltypes, eachcol

export orchestra_convert

# Convert target to given type.
#
# @param _  Target type.
# @param obj Object to convert.
# @return Converted object to target type.
function orchestra_convert{T}(_::Type{T}, obj;) 
  error("Conversion not provided for given arguments.")
end

function orchestra_convert(_::Type{OCDM}, da::DataArray;
  column_vars=nothing, column_names=nothing)

  # Convert to Vector,then OCDM
  da_as_mat = orchestra_convert(Vector, da)
  return orchestra_convert(OCDM, da_as_mat;
    column_vars=column_vars,
    column_names=column_names
  )
end
function orchestra_convert(_::Type{OCDM}, pda::PooledDataArray;
  column_vars=nothing, column_names=nothing)

  # Convert to dataframe,then OCDM
  if column_vars == nothing
    column_vars = [NominalVar(levels(pda))]
  end
  pda_as_da = convert(DataArray, pda)
  return orchestra_convert(OCDM, pda_as_da;
    column_vars=column_vars,
    column_names=column_names
  )
end
function orchestra_convert(_::Type{OCDM}, df::DataFrame;
  column_vars=nothing, column_names=nothing)

  # Build column OCDMs
  col_ocdms = OCDM[]
  for col = 1:size(df, 2)
    # Build name
    col_names = [string(names(df)[col])]
    if column_names != nothing
      col_names = [column_names[col]]
    end

    # Build variable if specified
    col_vars = nothing
    if column_vars != nothing
      col_vars = [column_vars[col]]
    end

    # Build column OCDM
    col_ocdm = orchestra_convert(OCDM, df[:, col];
      column_vars = col_vars,
      column_names = col_names
    )
    push!(col_ocdms, col_ocdm)
  end

  # Build full ocdm
  return hcat(col_ocdms...)
end

function orchestra_convert(_::Type{OCDM}, vec::AbstractVector;
  column_vars=nothing, column_names=nothing)

  # Conver to AbstractMatrix, then OCDM
  vec_as_mat = reshape(vec, length(vec), 1)
  return orchestra_convert(OCDM, vec_as_mat;
    column_vars = column_vars,
    column_names = column_names
  )
end

function orchestra_convert(_::Type{OCDM}, mat::AbstractMatrix;
  column_vars=nothing, column_names=nothing)

  # Create missing options
  if column_vars == nothing
    column_vars = [
      infer_var_type(mat[:, col_ind])
      for col_ind = 1:size(mat, 2)
    ]
  end
  if column_names == nothing
    column_names = gen_column_names(size(mat, 2))
  end

  # Initialize matrix and context
  ocdm_mat = similar(mat, Float64)
  ocdm_ctx = Dict{Symbol, Vector}()

  # Create OCDM
  fill_ocdm_mat!(ocdm_mat, mat, column_vars)
  fill_ocdm_ctx!(ocdm_ctx, column_vars, column_names)
  ocdm = OCDM(ocdm_mat, ocdm_ctx)

  return ocdm
end

# Generate column names.
gen_column_names(len) = String["V$(col_ind)" for col_ind = 1:len]

# Fill OCDM context with required values.
function fill_ocdm_ctx!(ocdm_ctx, column_vars, column_names)
  ocdm_ctx[:column_vars] = column_vars
  ocdm_ctx[:column_names] = column_names
end

# Fill OCDM matrix with values found in original matrix.
function fill_ocdm_mat!(ocdm_mat, mat, column_vars)
  # Fill values element-by-element
  for col = 1:size(mat, 2)
    col_var = column_vars[col]
    col_var_type = typeof(col_var)

    # Fill value according to variable type
    if col_var_type <: NumericVar
      for row = 1:size(mat, 1)
        mat_val = mat[row, col]
        ocdm_mat[row, col] = orchestra_isna(mat_val) ? nan(Float64) : mat_val
      end
    elseif col_var_type <: NominalVar || col_var_type <: OrdinalVar
      levels = col_var.levels
      level_dict = Dict(levels, 0:length(levels)-1)
      for row = 1:size(mat, 1)
        mat_val = mat[row, col]
        ocdm_mat[row, col] = 
          orchestra_isna(mat_val) ? nan(Float64) : level_dict[mat_val]
      end
    end
  end
end


function orchestra_convert(_::Type{DataArray}, vec::Vector)
  # Build NA bitmask
  na_bitmask = orchestra_isna(vec)
  
  # Build NA-free array, 
  # replacing NA elements with first element that is not NA
  na_free_array = copy(vec)
  na_free_array[na_bitmask] = first(vec[!na_bitmask])
  
  # Convert NA-free array to inferred type
  el_type = infer_eltype(na_free_array)
  na_free_array = convert(Vector{el_type}, na_free_array)
  
  # Build and return data array
  data_array = DataArray(na_free_array, na_bitmask)
  return data_array
end

function orchestra_convert(_::Type{DataFrame}, mat::Matrix;
  create_factors=true)

  # Build columns as data arrays
  columns = Array(Any, size(mat, 2))
  for i = 1:size(mat, 2)
    columns[i] = orchestra_convert(DataArray, mat[:, i])
    
    el_type = eltype(columns[i])
    if create_factors && (el_type <: String || el_type <: Symbol)
      columns[i] = pool(columns[i])
    end
  end
  # Build data frame
  df = DataFrame(columns)

  return df
end

function orchestra_convert(_::Type{Vector}, da::DataArray)
  # Build vector, with NA replaced with nan(Float64)
  vec = Array(Any, size(da)...)
  for x = 1:length(da)
    da_val = da[x]
    vec[x] = isna(da_val) ? nan(Float64) : da_val
  end

  return vec
end

function orchestra_convert(_::Type{Matrix}, df::DataFrame)
  # Build matrix, with NA replaced with nan(Float64)
  mat = Array(Any, size(df)...)
  for col = 1:size(df, 2)
    for row = 1:size(df, 1)
      df_val = df[row, col]
      mat[row, col] = isna(df_val) ? nan(Float64) : df_val
    end
  end

  return mat
end

function orchestra_convert(_::Type{Vector{Float64}}, da::DataArray)
  # Fail if column is not subtype of Real
  if !issubtype(eltype(da), Real)
    error("DataArray is not subtype of Real.")
  end

  # Build vector, with NA replaced with nan(Float64)
  vec = Array(Float64, size(da)...)
  for x = 1:length(da)
    da_val = da[x]
    vec[x] = isna(da_val) ? nan(Float64) : da_val
  end

  return vec
end

function orchestra_convert(_::Type{Matrix{Float64}}, df::DataFrame)
  # Fail if any columns are not subtype of Real
  df_eltypes = eltypes(df)
  if any([!issubtype(t, Real) for t in df_eltypes])
    error("Some columns are not subtypes of Real.")
  end

  # Build matrix, with NA replaced with nan(Float64)
  mat = Array(Float64, size(df)...)
  for col = 1:size(df, 2)
    for row = 1:size(df, 1)
      df_val = df[row, col]
      mat[row, col] = isna(df_val) ? nan(Float64) : df_val
    end
  end

  return mat
end

end # module
