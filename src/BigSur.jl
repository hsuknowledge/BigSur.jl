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
    predicate = x -> mean_lower < x < mean_upper
    sub_rows = findall(predicate, calc_mu.gene_totals ./ n)
    @info "finding fit for c in " * string(length(sub_rows)) * " genes"
    @time @info optimize_c_for_Fano!(calc_fano, sub_rows)
    @info "using this c to calculate modified corrected Fano factors for all"
    mcFano = calc_fano.cache
    nulldistribution_fano = calculator_nulldistribution_Fano(calc_fano)
    quan = map(1:m) do i
        r = filter(x -> x.im == 0, roots(nulldistribution_fano(i) - mcFano[i]))
        length(r) == 0 ? 0 : minimum(abs.(r))
    end
    pval = map(x -> ccdf(Normal(), x), quan)
    padj = adjust(pval, BenjaminiHochberg())
    hvg = @. calc_fano.cache >= min_fano && padj <= FDR
    cv = [calc_fano.c[1] for _ in mcFano]
    DataFrame("names" => names, "cv" => cv, "mcFano" => mcFano, "quantile" => quan,
              "p_val" => pval, "padj_BH" => padj, "highly_variable" => hvg)
end

#end # module
