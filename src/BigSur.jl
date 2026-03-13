#module BigSur

using IntervalArithmetic
setdisplay(:midpoint; ng_flag = false)

using DataFrames: DataFrame
using OhMyThreads: @tasks
using VectorizedReduction: vmapreducethen
using LazyArrays: LazyArray, @~
using StatsBase: median, mad
using CurveFit: CurveFitProblem, PowerCurveFitAlgorithm, solve, coef
using Roots: find_zero, Order16
using IntervalRootFinding: roots, Bisection, root_region
using Combinatorics: stirlings2
using MultipleTesting: adjust, BenjaminiHochberg
using Distributions: LogNormal, Poisson

include("DataStructure.jl")
include("IntervalPolynomial.jl") ## for Cornish-Fisher expansion
include("StatsCore.jl")
include("MaillardCorrection.jl") ## for Cornish-Fisher expansion
include("Algorithm.jl")

include("IntervalExtra.jl") ## interval-valued normccdf function

function findVariableGenes(mat::AbstractMatrix{<:Real}, names;
                           mean_lowbound = 0.1, mean_highbound = 100,
                           min_fano = 1.5, FDR = 0.05)
    m, n = size(mat)
    model = BigSurModel(mat, names)
    set_cv_fitting_range!(model, mean_lowbound, mean_highbound)
    best_cv = find_cv!(model)
    apply_cv!(model, best_cv; to_all = true)
    df = rowdata(model)
    try
        quant = map(1:m) do i
            f = quantile_null_mcFano(model, i) - df.mcFano[i]
            r = roots(f, interval(-500, 500))
            idx_min = sortperm(@. mid(abs(root_region(r))))[1]
            root_region(r[idx_min])
        end
        df[!, :quantile] = mid.(quant)
        df[!, :p_val] = pval = map(q -> mid(normccdf(q)), quant)
        df[!, :padj_BH] = padj = adjust(pval, BenjaminiHochberg())
        df[!, :highly_variable] = hvg = @. df.mcFano >= min_fano && padj <= FDR
    catch e
        @warn "Some data weren't fully processed. Please check `rowdata(returned_model)`."
        @info e stacktrace()
    end
    model
end

#end # module
