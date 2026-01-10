#module BigSur

using IntervalArithmetic
setdisplay(:midpoint; ng_flag = false)

using DataFrames: DataFrame
using OhMyThreads: @tasks
using LazyArrays: LazyArray, @~
using CurveFit: CurveFitProblem, PowerCurveFitAlgorithm, solve, coef
using Roots: find_zero
using IntervalRootFinding: roots, Bisection, root_region
using Combinatorics: stirlings2
using MultipleTesting: adjust, BenjaminiHochberg
using LinearAlgebra: norm
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
        ## if big cv: no expansion (order 0), assume normal (or maybe we include skew?)
        order = best_cv > 1 ? 0 : 4
        df[!, :quantile] = quant = map(1:m) do i
            f = quantile_null_mcFano(model, i; order = order) - df.mcFano[i]
            r = roots(f, interval(-1000, 1000))
            idx_min = sortperm(@. mid(abs(root_region(r))))[1]
            root_region(r[idx_min])
        end
        df[!, :p_val] = pval = map(q -> mid(normccdf(q)), quant)
        df[!, :padj_BH] = padj = adjust(pval, BenjaminiHochberg())
        df[!, :highly_variable] = hvg = @. df.mcFano >= min_fano && padj <= FDR
    catch
        @warn "Some data weren't fully processed. Please check `rowdata(returned_model)`."
    end
    model
end

#end # module
