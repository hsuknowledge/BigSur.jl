import Arrow
using DataFrames: DataFrame
using SparseArrays: SparseMatrixCSC

"""
Import SparseMatrixCSC-formatted data from DataFrame.

To read from an Arrow file, use `Arrow.Table` and pipe it to `DataFrame`.

Example
=======

```
umi = Arrow.Table("demo.arrow") |> DataFrame
mat = import_CSC_Matrix(umi)
```
"""
function import_CSC_Matrix(df::DataFrame, row::Integer = 1)
    @assert names(df) == ["rowname", "colname", "rowidx", "colptr", "value"]
    m = length(df.rowname[row])
    n = length(df.colname[row])
    rowidx = df.rowidx[row] .+ 1
    colptr = df.colptr[row] .+ 1
    SparseMatrixCSC(m, n, colptr, rowidx, copy(df.value[row]))
end
