#module BigSur

using DataFrames: DataFrame
using OhMyThreads: tmap!, @tasks
using GLM: lm, @formula, coef
using Optim: optimize
using LazyArrays: LazyArray, @~
using PartialFunctions: @$
import Combinatorics # stirlings2
using Polynomials: Polynomial, roots
using Distributions: ccdf, Normal
using MultipleTesting: adjust, BenjaminiHochberg

include("StatsCore.jl")
include("Algorithm.jl")

function findVariableGenes(mat::AbstractMatrix{T}, names;
                           mean_lower = 0.1, mean_upper = 100,
                           min_fano = 1.5, FDR = 0.05) where {T<:Number}
    m, n = size(mat)
    calc_mu = calculator_expected_counts(mat)
    calc_cv = calculator_coefficient_variation(calc_mu)
    rowmask = (x -> mean_lower < x < mean_upper).(calc_cv.model.gene_mean)
    @info "finding fit for c in " * string(sum(rowmask)) * " genes"
    @time @info find_coefficient_variation!(calc_cv, rowmask)
    @info "using this c to calculate modified corrected Fano factors for all"
    mcFano = calc_cv.model.mcFano
    nulldistribution_fano = calculator_nulldistribution_Fano(calc_cv)
    quan = map(1:m) do i
        r = roots(nulldistribution_fano(i) - mcFano[i])
        if eltype(r) isa Complex; r = findall(x -> x.im == 0, r) end
        length(r) == 0 ? 0 : minimum(abs.(r))
    end
    pval = map(x -> ccdf(Normal(), x), quan)
    padj = adjust(pval, BenjaminiHochberg())
    hvg = @. mcFano >= min_fano && padj <= FDR
    cv = [calc_cv.c[1] for _ in mcFano]
    DataFrame("names" => names, "cv" => cv, "mcFano" => mcFano, "quantile" => quan,
              "p_val" => pval, "padj_BH" => padj, "highly_variable" => hvg)
end

#end # module
