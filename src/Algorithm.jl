function calculator_expected_counts(mat::AbstractMatrix{T}) where {T<:Number}
    gene_totals = sum(mat, dims = 2) |> vec
    cell_totals = sum(mat, dims = 1) |> vec
    total = sum(cell_totals)
    @inline function (i, j)
        @boundscheck checkbounds(mat, i, j)
        gene_totals[i] * cell_totals[j] / total
    end
end

function calculator_Pearson_residuals(_μij)
    @inline function (i, j, c)
        @boundscheck checkbounds(_μij.mat, i, j)
        μij = _μij(i, j)
        (_μij.mat[i, j] - μij) / sqrt(μij * (1 + c^2 * μij))
    end
end

function calculator_modifiedcorrected_Fano(_Pij)
    m, n = size(_Pij._μij.mat)
    T = typeof(_Pij._μij.total)
    c = zeros(T, 1)
    cache = zeros(T, m)
    function (c_search = c[1], rows = nothing)
        rows = isnothing(rows) ? (1:m) : rows
        if c[1] != c_search # simply overwrite the cached value
            c[1] = c_search
            @views tmap!(cache[rows], rows) do i
                mapreduce(j -> _Pij(i, j, c_search)^2, +, 1:n) / (n - 1)
            end
        end
        @views cache[rows]
    end
end

function optimize_c_for_Fano!(_mcϕi, sub_rows)
    X = _mcϕi._Pij._μij.gene_totals[sub_rows] ./ _mcϕi.m
    function objective(c)
        Y = _mcϕi(c, sub_rows)
        prob = CurveFitProblem(X, Y)
        sol = solve(prob, PowerCurveFitAlgorithm())
        slope = sol.u[1]
        abs(slope)
    end
    result = optimize(objective, 0, 1)
    c = result.minimizer
    _mcϕi(c) # not returned; instead, fanos_calc.cache is always accessible
    (; c = c, best_slope = result.minimum)
end

function calculator_modifiedcorrected_PCC(_mcϕi)
    # inner product of rows i1 and i2 of residuals divided by respective fanos
    function (i1, i2)
        c, cache = _mcϕi.c[1], _mcϕi.cache
        tmapreduce(+, 1:_mcϕi.n) do j
            mapreduce(i -> _mcϕi._Pij(i, j, c) / sqrt(cache[i]), *, [i1, i2])
        end / (_mcϕi.n - 1)
    end
end

function calculator_PoissonLogNormal_moments(_mcϕi)
    chi = 1 + _mcϕi.c[1]^2
    @inline function (i, j)
        m = _mcϕi._Pij._μij(i, j)
        # note we return a vector with indices 1:11 for r[0] to r[10],
        # needed for the 5 moments of modified corrected fano factors
        map(0:10) do k
            mapreduce(a -> stirlings2(k, a) * chi^(a*(a-1)/2) * m^a, +, 0:k)
        end
    end
end

function calculator_Fano_cumulants_halfway(_PLNrij)
    @inline function(i)
        data = _PLNrij._mcϕi
        tmapreduce(+, 1:data.n) do j
            r = _PLNrij(i, j)
            m = data._Pij._μij(i, j)
            mcm = m + (data.c[1]m)^2
            e = map(1:5) do k
                mapreduce(a -> binomial(2k, a) * (-m)^(2k-a) * r[a+1], +, 0:2k) / mcm^k
            end
            [ # kappa1 to 5 summands of each cell at gene i
                e[1], # e[1] == 1
                -e[1]^2 + e[2],
                2e[1]^3 - 3e[2]e[1] + e[3],
                -6e[1]^4 + 12e[2]e[1]^2 - 3e[2]^2 - 4e[1]e[3] + e[4],
                24e[1]^5 - 60e[2]e[1]^3 + 20e[3]e[1]^2 - 10e[2]e[3] + 5(6e[2]^2-e[4])e[1] + e[5]
            ]
        end # ./ map(k -> (data.n - 1)^k, 1:5)
        # We skip the denominators, knowing they would cancel each other out.
        # The resulting numbers are not the actual cumulant values.
    end
end

function calculator_Fano_CornishFisher(_κϕi)
    @inline function(i)
        k = _κϕi(i)
        # The denominators in actual cumulants cancel each other out in the below ratios.
        # For ratios t, u, and sqrt(k[2]), an (n-1) is left;
        # for ratios a, b, their (n-1)^k cancel each other out completely.
        df = _κϕi._PLNrij._mcϕi.n - 1
        mcϕi = _κϕi._PLNrij._mcϕi.cache[i]
        t, u, v = k[3]/6k[2], k[5]/20k[2]^2, sqrt(k[2])
        a, b = k[3]^2/18k[2]^3, k[4]/4k[2]^2
        f = x -> 1 - mcϕi + ((-1 + 17a/3 - 2b  )t + u/2) / df +
                        x *  ( 1 +  5a/2 -  b/2)v        / df +
                      x^2 * (( 1 - 53a/3 + 5b  )t - u)   / df +
                      x^3 *  (   -   a   +  b/6)v        / df +
                      x^4 * ((   +  4a   -  b  )t + u/6) / df
        #ccdf(Normal(), find_zero(f, 0))
    end
end
