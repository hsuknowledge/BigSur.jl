import Arrow
using DataFrames: DataFrame
using SparseArrays: SparseMatrixCSC
using OhMyThreads: tmap!, tmapreduce
using CurveFit: CurveFitProblem, PowerCurveFitAlgorithm, solve
using Optim: optimize
using Combinatorics: stirlings2
using Roots: find_zero
using Distributions: ccdf, Normal
using MultipleTesting: adjust, BenjaminiHochberg

include("Algorithm.jl")
include("Utils.jl")

function findVariableGenes(mat::AbstractMatrix{T}, names;
                           mean_lower = 0.1, mean_upper = 100,
                           min_fano = 1.5, FDR = 0.05) where {T<:Number}
    m, n = size(mat)
    calc_mu = calculator_expected_counts(mat)
    calc_res = calculator_Pearson_residuals(calc_mu)
    calc_fano = calculator_modifiedcorrected_Fano(calc_res)
    predicate = x -> x > mean_lower && x < mean_upper
    sub_rows = findall(predicate, calc_mu.gene_totals ./ n)
    @info "finding fit for c in " * string(length(sub_rows)) * " genes"
    @time @info optimize_c_for_Fano!(calc_fano, sub_rows)
    @info "using this c to calculate modified corrected Fano factors for all"
    calc_pln = calculator_PoissonLogNormal_moments(calc_fano)
    calc_fano_cumulant = calculator_Fano_cumulants_halfway(calc_pln)
    calc_fano_cfpoly = calculator_Fano_CornishFisher(calc_fano_cumulant)
    roots = map(i -> try find_zero(calc_fano_cfpoly(i), 0) catch; 0 end, 1:m)
    pval = map(root -> ccdf(Normal(), root), roots)
    padj = adjust(pval, BenjaminiHochberg())
    hvg = @. calc_fano.cache >= min_fano && padj <= FDR
    DataFrame("names" => names, "mcFano" => calc_fano.cache, "root" => roots,
              "p_val" => pval, "padj_BH" => padj, "highly_variable" => hvg)
end
