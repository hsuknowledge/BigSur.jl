#module BigSur

using OhMyThreads: tmap!, @tasks
using GLM: lm, @formula, coef
using Optim: optimize
import Combinatorics # stirlings2
using Polynomials: Polynomial, roots
using Distributions: ccdf, Normal
using MultipleTesting: adjust, BenjaminiHochberg

include("Algorithm.jl")
include("Utils.jl")

function findVariableGenes(mat::AbstractMatrix{T}, names;
                           mean_lower = 0.1, mean_upper = 100,
                           min_fano = 1.5, FDR = 0.05) where {T<:Number}
    m, n = size(mat)
    calc_mu = calculator_expected_counts(mat)
    calc_fano = calculator_modifiedcorrected_Fano(calc_mu)
    predicate = x -> x > mean_lower && x < mean_upper
    sub_rows = findall(predicate, calc_mu.gene_totals ./ n)
    @info "finding fit for c in " * string(length(sub_rows)) * " genes"
    @time @info optimize_c_for_Fano!(calc_fano, sub_rows)
    @info "using this c to calculate modified corrected Fano factors for all"
    calc_fano_cumulant = calculator_unaveraged_Fano_cumulants(calc_fano)
    calc_fano_cfpoly = calculator_Fano_CornishFisher(calc_fano_cumulant)
    root = map(1:m) do i
        r = filter(x -> x.im == 0, roots(calc_fano_cfpoly(i)))
        length(r) == 0 ? 0 : minimum(abs.(r))
    end
    pval = map(x -> ccdf(Normal(), x), root)
    padj = adjust(pval, BenjaminiHochberg())
    hvg = @. calc_fano.cache >= min_fano && padj <= FDR
    DataFrame("names" => names, "mcFano" => calc_fano.cache, "root" => root,
              "p_val" => pval, "padj_BH" => padj, "highly_variable" => hvg)
end

#end # module
