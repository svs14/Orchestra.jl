module TestConversion

using FactCheck

import DataFrames: DataArray, DataFrame
import DataFrames: complete_cases, isna, pool, eltypes
import DataFrames: @data, NA, array, PooledDataArray

importall Orchestra.Conversion

facts("Orchestra conversion functions") do
  context("orchestra_convert throws error on undefined conversion") do
    @fact_throws orchestra_convert(Dict, [1,2])
  end
  context("orchestra_convert handles vector to data array") do
    context("in the simple case") do
      # Simple case
      vec = {1.0, 4.0, 3, NA}
      data_array = orchestra_convert(DataArray, vec)
      expected = convert(DataArray{Real}, @data([1.0, 4.0, 3, NA]))

      @fact typeof(data_array) <: DataArray => true
      @fact isequal(data_array, expected) => true
    end
    context("treating NaN as NA") do
      # Treat NaN as NA
      vec = {1.0, 4.0, 3, NA, nan(3.0)}
      data_array = orchestra_convert(DataArray, vec)
      expected = convert(DataArray{Real}, @data([1.0, 4.0, 3, NA, NA]))

      @fact typeof(data_array) <: DataArray => true
      @fact isequal(data_array, expected) => true
    end
    context("not treating NaN as NA") do
      # Don't treat NaN as NA
      vec = {1.0, 4.0, 3, NA, nan(3.0)}
      data_array = orchestra_convert(DataArray, vec; nan_as_na=false)
      expected = convert(DataArray{Real}, @data([1.0, 4.0, 3, NA, nan(3.0)]))

      @fact typeof(data_array) <: DataArray => true
      @fact isequal(data_array, expected) => true
    end
  end

  context("orchestra_convert handles matrix to data frame") do
    mat = {
      0  1.0 "a";
      1  2.0 "b";
      3  4.0 "c";
      NA 6.0 "d";
    }
    df = orchestra_convert(DataFrame, mat)

    @fact typeof(df) <: DataFrame => true
    @fact eltype(df[1]) => Int
    @fact eltype(df[2]) => Float64
    @fact eltype(df[3]) => ASCIIString

    @fact typeof(df[1]) <: DataArray => true
    @fact typeof(df[2]) <: DataArray => true
    @fact typeof(df[3]) <: PooledDataArray => true

    @fact isequal(array(df), mat) => true
  end

  context("orchestra_convert handles data frame to matrix") do
    mat = {
      0  1.0 "a";
      1  2.0 "b";
      3  4.0 "c";
      NA 6.0 "d";
    }
    df = orchestra_convert(DataFrame, mat)

    expected_mat = {
      0.0      1.0 "a";
      1.0      2.0 "b";
      3.0      4.0 "c";
      nan(0.0) 6.0 "d";
    }
    actual_mat = orchestra_convert(Matrix, df)

    @fact typeof(actual_mat) <: Matrix => true
    @fact isequal(actual_mat, expected_mat) => true
  end
  context("orchestra_convert handles data array to vector") do
    da = @data([1, 2, NA, 4, "b"])
    actual_vec = orchestra_convert(Vector, da)
    expected_vec = {1, 2, nan(Float64), 4, "b"}

    @fact typeof(actual_vec) <: Vector => true
    @fact isequal(actual_vec, expected_vec) => true
  end

  context("orchestra_convert handles data frame to float matrix") do
    context("in the simple case") do
      mat = {
        0  1.0 "a";
        1  2.0 "b";
        3  4.0 "c";
        NA 6.0 "d";
      }
      df = orchestra_convert(DataFrame, mat)

      expected_float_mat = [
        0.0      1.0;
        1.0      2.0;
        3.0      4.0;
        nan(0.0) 6.0;
      ]
      actual_float_mat = orchestra_convert(Matrix{Float64}, df[:, 1:end-1])

      @fact typeof(actual_float_mat) <: Matrix{Float64} => true
      @fact isequal(actual_float_mat, expected_float_mat) => true
    end
    context("by throwing an error on incompatible data frame") do
      mat = {
        0  1.0 "a";
        1  2.0 "b";
        3  4.0 "c";
        NA 6.0 "d";
      }
      df = orchestra_convert(DataFrame, mat)

      @fact_throws orchestra_convert(Matrix{Float64}, df)
    end
  end

  context("orchestra_convert handles data array to float vector") do
    context("in the simple case") do
      da = @data([1, 2, NA, 4])
      actual_vec = orchestra_convert(Vector{Float64}, da)
      expected_vec = Float64[1, 2, nan(Float64), 4]

      @fact typeof(actual_vec) <: Vector{Float64} => true
      @fact isequal(actual_vec, expected_vec) => true
    end
    context("by throwing an error on incompatible data array") do
      da = @data([1, 2, NA, "a"])

      @fact_throws orchestra_convert(Vector{Float64}, da)
    end
  end
end

end # module
