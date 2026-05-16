function apply_cv!(model::BigSurModel{T}, cv::U; ## cv can be a ForwardDiff.Dual
                   to_all::Bool = false)::U where {T<:Real, U<:Real}
    df, gs = rowdata(model), genes(model)
    m, n = size(model)
    mask = copy(df.fit_cv)
    @assert sum(mask) > 0 "Run `set_cv_fitting_range!` first."
    xrows, erows = eachrow(measured_values(model)), eachrow(expected_values(model))
    pcols = eachcol(pearson_residual(model))
    if to_all
        @tasks for i in 1:length(gs)
            map!(Base.Fix{3}(pearson_residual, cv), pcols[i], xrows[gs[i]], erows[gs[i]])
            df.sum_res[gs[i]] = vvmapreduce(abs2, +, pcols[i])
            df.mcFano[gs[i]] = df.sum_res[gs[i]] / (n - 1)
        end
        return cv
    end
    mcFano = typeof(cv) == T ? df.mcFano : zeros(typeof(cv), m)
    @tasks for i in findall(mask) ## again, faster and less alloc than mapreduce
        acc = 0
        for (x, mu) in zip(xrows[i], erows[i])
            acc += pearson_residual2(x, mu, cv)
        end
        mcFano[i] = acc / (n - 1)
    end
    fanos = _unwrap_number.(mcFano[mask])
    mask[mask] .&= abs.(fanos .- median(fanos)) .<= 5 * mad(fanos)
    prob = CurveFitProblem(df.gene_means[mask], mcFano[mask])
    sol = solve(prob, PowerCurveFitAlgorithm())
    slope = coef(sol)[1] # [slope, intercept] (y=a^x*b)
    @info "search cv = " * _string(cv) * ", slope = " * _string(slope)
    if typeof(cv) == T ## only push if cv is the correct type, avoiding Dual numbers
        df.cv[1] = cv
        push!(cv_searched(model), (cv => slope))
    end
    slope
end

find_cv!(model::BigSurModel{<:Real}) = find_zero(Base.Fix1(apply_cv!, model), 1)
find_cv!(model::BigSurModel{<:Interval}) = begin
    region = interval(0, 1)
    rs = root_region.(roots(Base.Fix1(apply_cv!, model), region))
    while (length(rs) == 0 && sup(region) <= 4) ## we only support search upto <8
        region += sup(region)
        rs = root_region.(roots(Base.Fix1(apply_cv!, model), region))
    end
    length(rs) == 0 && return sup(region) ## return 8; search no further
    length(rs) == 1 && return rs[1]
    hull(rs...; dec = :auto)
end

function get_cumulants(model::BigSurModel{T}, g::Integer) where {T<:Real}
    n, c, emat, df = size(model)[2], cv(model), expected_values(model), rowdata(model)
    chi_pow = chipowers_table(8, c)
    S2 = stirlings2_table(8)
    B = binomial_table(8)
    k = @tasks for j in 1:n
        @set reducer = +
        @local begin
            m_pow, r = Vector{T}(undef, 9), Vector{T}(undef, 9)
            e = Vector{T}(undef, 4)
        end
        m_pow .= emat[g, j].^(0:8)
        map!(k -> poisson_lognormal_moment(k, m_pow, chi_pow, S2), r, 0:8)
        map!(k -> pearson_residual2_moment(k, m_pow, c, r, B), e, 1:4)
        noncentral_moment_to_cumulant(e)
    end
    df.k1[g], df.k2[g], df.k3[g], df.k4[g] = k
end

function quantile_polynomial(k1, k2, k3, k4) # input cumulants
    n = k1
    μ = k1 / (n - 1)
    σ = sqrt(k2) / (n - 1)
    S = k3 / k2^1.5
    K = k4 / k2^2
    Sc, Kc = input_correction_Maillard(S, K) # Some S and K are not covered
    !isnothing(Sc) || return quantile_Cornish_Fisher(μ, σ, S, K)
    quantile_Cornish_Fisher(μ, σ, Sc, Kc)
    # pval = ccdf(Normal(), roots(poly - mcFano))
end

function pearson_correlation!(model::BigSurModel{T}, PCC::AbstractMatrix{T}) where T
    (m, n), c, gs = size(model), cv(model), genes(model)
    @assert size(PCC) == (m, m)
    p = eachcol(pearson_residual(model))
    fano = rowdata(model).mcFano
    @tasks for i1 in 1:m
        @set scheduler = :greedy
        for i2 in i1+1:m
            den = (n-1) * sqrt(fano[gs[i1]] * fano[gs[i2]])
            PCC[i2, i1] = vvmapreduce(*, +, p[i1], p[i2]) / den
        end
    end
end

@inline function simulation_gene_levels(gene_totals, n)
    a = max(2, minimum(gene_totals))
    e = n / 50
    h = maximum(gene_totals)
    @. Int(round([ # ordered from least to most, 9 points
        a, a^(3/4) * e^(1/4), sqrt(a * e), a^(1/4) * e^(3/4),
        e, e^(3/4) * h^(1/4), sqrt(e * h), e^(1/4) * h^(3/4), h
    ]))
end

@inline function simulation_trials(sim_total, n)
    a = log10(sim_total)
    Int(round(4e7 / n / (a^(1/5) + 0.5a^3)))
end

@inline function poissonLogNormal_sample(m, c)
    chi = 1 + c^2
    rate = rand(LogNormal(log(m / sqrt(chi)), sqrt(log(chi))))
    rand(Poisson(rate))
end

function simulate_invsqrt_moments(model::BigSurModel)
end

function quantile_null_PCC(model::BigSurModel{T}, i1::Integer, i2::Integer) where {T<:Real}
end
