# Conversion utilities.
module Conversion

using Orchestra.Util

import DataFrames: DataArray, DataFrame, complete_cases, isna, pool, eltypes

export orchestra_convert

# Convert target to given type.
#
# @param _  Target type.
# @param obj Object to convert.
# @return Converted object to target type.
function orchestra_convert{T}(_::Type{T}, obj;) 
  error("Conversion not provided for given arguments.")
end

# Convert vector to data array.
# 
# @param _ DataArray's type.
# @param vec Vector.
# @param nan_as_na Treat NaN as NA.
# @return Data array.
function orchestra_convert(_::Type{DataArray}, vec::Vector;
  nan_as_na=true)

  # Build NA bitmask
  na_bitmask = Bool[isna(x) for x in vec]
  if nan_as_na
    na_bitmask |= Bool[typeof(x) <: FloatingPoint && isnan(x) for x in vec]
  end
  
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

# Convert matrix to dataframe.
#
# Creates factors (PooledDataArray) 
# from String or Symbol columns if option is set.
# 
# @param _ DataFrame's type.
# @param mat Source matrix.
# @param create_factors Create factors from relevant columns.
# @param nan_as_na Treat NaN as NA element.
# @return DataFrame produced from matrix.
function orchestra_convert(_::Type{DataFrame}, mat::Matrix;
  create_factors=true, nan_as_na=true)

  # Build columns as data arrays
  columns = Array(Any, size(mat, 2))
  for i = 1:size(mat, 2)
    columns[i] = orchestra_convert(DataArray, mat[:, i]; nan_as_na=nan_as_na)
    
    el_type = eltype(columns[i])
    if create_factors && (el_type <: String || el_type <: Symbol)
      columns[i] = pool(columns[i])
    end
  end
  # Build data frame
  df = DataFrame(columns)

  return df
end

# Converts data array to vector.
# Throws error if not possible.
#
# @param _ Vector's type.
# @param da Data array.
# @return Vector.
function orchestra_convert(_::Type{Vector}, da::DataArray)
  # Build vector, with NA replaced with nan(Float64)
  vec = Array(Any, size(da)...)
  for x = 1:length(da)
    da_val = da[x]
    vec[x] = isna(da_val) ? nan(Float64) : da_val
  end

  return vec
end

# Converts dataframe to matrix.
# Throws error if not possible.
#
# @param df Data frame.
# @return Matrix.
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

# Converts data array to float vector.
# Throws error if not possible.
#
# @param _ Vector{Float64}'s type.
# @param da Data array.
# @return Float vector.
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

# Converts dataframe to float matrix.
# Throws error if not possible.
#
# @param _ Matrix{Float64}'s type.
# @param df Data frame.
# @return Float matrix.
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
