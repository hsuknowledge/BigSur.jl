"""
Amédée-Manesme, CO., Barthélémy, F. & Maillard, D. Computation of the corrected
Cornish–Fisher expansion using the response surface methodology: application to
VaR and CVaR. Ann Oper Res 281, 423–453 (2019).
https://doi.org/10.1007/s10479-018-2792-4

### Input correction for CF expansion
This is the reverse function approximation that, given input of actual skewness
and excess kurtosis we want on a CF expansion curve, calculates the values that
we would plug into the CF expansion formula. We should still check the validity
dZ/dz > 0 ∀z, using the corrected skewness and excess kurtosis values.
"""
function input_correction_Maillard(S, K)
    if 5 <= K <= 40
        if 0.5 < S <= 2.2;      case = 1
        elseif 0 < S <= 0.5;    case = 2
        else return (nothing, nothing) end
    elseif 0 <= K < 5
        if 0.5 <= S;            case = 3
        elseif 0.25 <= S < 0.5; case = 4
        elseif 0 < S < 0.25;    case = 5
        else return (nothing, nothing) end
    else
        return (nothing, nothing)
    end
    s, k, sk, lS, lK = sqrt(S), sqrt(K), sqrt(S * K), log(S), log(K)
    factors = [1, s, k, S, K, sk, s*S, k*K, s*K, k*S, S^2, K^2,
        sk*S, S*K, sk*K, S*K^2, S^2*K, sk^3, lS*lK, lS*K, lK*S, 1/S, 1/K]
    (sum(table_Sc[:, case] .* factors), sum(table_Kc[:, case] .* factors))
end

"""
Maillard, Didier, A User’s Guide to the Cornish Fisher Expansion (May 1, 2018).
Available at SSRN: https://ssrn.com/abstract=1997178

This is the second moment of the resulting function, given an input skewness Sc
and excess kurtosis Kc (assuming corrected version here). This is the magnitude
of the increased variance of the resulting function, so we are going to correct
for it in the input standard deviation by the square root of this value.
"""
@inline m2_Cornish_Fisher(Sc, Kc) = 1 + Sc^2/96 + 25Kc^4/1296 - Sc * Kc^2/36;

"""
Maillard, Didier, A User’s Guide to the Cornish Fisher Expansion (May 1, 2018).
Available at SSRN: https://ssrn.com/abstract=1997178

### Validity test for CF expansion
Rationale: dZ/dz > 0 ∀z for the bijective property to hold.
For DZ > 0 ∀z, as a quadratic function of z, its discriminant must be < 0.
Since this discriminant is a quadratic function of K that opens upward, its
coefficients in terms of s=S/6 must satisfy b^2-4ac > 0 ie. s^4 - 6s^2 + 1 > 0,
such that ∃K ∈ (K', K") satisfying a strictly negative discriminant of dZ/dz.
"""
function validity_Cornish_Fisher(S, K)
    domain_criterion = abs(abs(S) - 6sqrt(2)) - 6
    discriminant_DZ = 27K^2 - (216 + 66S^2)K + 40S^4 + 336S^2
    try domain_criterion > 0 catch; false end || return false
    try discriminant_DZ < 0 catch; false end
end

const table_Sc = [ ## Table 4, Cases 1-5
-1.816   -0.0189      2.111    0.172    0.00512     ## Constant
 6.812    0.161       0        0.132   -0.0240      ## S^{1/2}
-0.577    0.0215     -3.498   -0.296   -0.00778     ## K^{1/2}
-8.636    0.453      -2.870    0        1.277       ## S
 0.508    0.00139    -0.123   -0.0415   0.00499     ## K
 0       -0.0862      3.836    0.346    0.0386      ## S^{1/2} K^{1/2}
 4.235    0.326       2.956    1.491   -0.114       ## S^{3/2}
-0.00685 -0.00000851 -0.162   -0.0327  -0.000479    ## K^{3/2}
-0.848   -0.00168     0        0       -0.0336      ## S^{1/2} K
 2.671    0.230       0        0       -0.483       ## S K^{1/2}
-0.0969  -0.0136      2.008    0.134    0.265       ## S^2
-0.000304 0.00000232  0.0370   0.00278 -0.0000520   ## K^2
-1.259   -0.129      -4.884   -1.330   -0.0857      ## S^{3/2} K^{1/2}
 0.226   -0.000326    1.720    0.249    0.108       ## S K
 0.0191  -0.000151   -0.153    0.0333   0.00708     ## S^{1/2} K^{3/2}
 0.000196 0.0000493  -0.00138 -0.00129 -0.00487     ## S K^2
 0.0249   0.00662     0.239    0.205   -0.0332      ## S^2 K
-0.00666 -0.000649   -0.0883  -0.0597   0.0161      ## S^{3/2} K^{3/2}
-0.105    0.00396    -0.227   -0.0109  -0.000270    ## ln(S) ln(K)
 0.0987   0.000457   -0.436   -0.0507   0.000262    ## ln(S) K
-0.845   -0.221       0.700    0.114    0.0513      ## S ln(K)
 0.135    0.000228   -0.0739  -0.00419  0.000000429 ## S^{-1}
-0.416   -0.0250      0.0414   0.00152  0.000110    ## K^{-1}
]

const table_Kc = [ ## Table 3, Cases 1-5
 -5.962    0.0832     1.749  -1.612  -0.304     ## Constant
 21.53     0.0451     0       1.894   0.743     ## S^{1/2}
 -1.548    0.732     -6.604   1.938   0.597     ## K^{1/2}
-26.52    -0.601      3.425   0      -1.662     ## S
  1.820    0.124      1.313   0.273   0.676     ## K
  0        0.396      7.491  -1.018  -1.073     ## S^{1/2} K^{1/2}
 11.08     1.261    -11.83   -4.220   0.226     ## S^{3/2}
 -0.0443  -0.0195    -0.858  -0.141  -0.299     ## K^{3/2}
 -2.564   -0.0704     0       0       0.490     ## S^{1/2} K
  5.739   -0.528      0       0       2.314     ## S K^{1/2}
  0.342   -0.198      9.011   2.164   0.463     ## S^2
  0.00162  0.00181    0.141   0.0247  0.0432    ## K^2
 -3.773   -0.122     -3.346   2.786  -0.234     ## S^{3/2} K^{1/2}
  0.880    0.0836     0.638  -0.454  -0.891     ## S K
  0.0328   0.000231   0.110   0.0381 -0.0254    ## S^{1/2} K^{3/2}
  0.000901 0.0000956 -0.124  -0.0392 -0.00616   ## S K^2
  0.0717   0.0133    -0.642  -0.862  -0.272     ## S^2 K
 -0.0216  -0.00373    0.499   0.307   0.205     ## S^{3/2} K^{3/2}
 -0.721   -0.0305    -0.517   0.103   0.00942   ## ln(S) ln(K)
  0.349    0.00290   -0.650   0.0341 -0.00642   ## ln(S) K
  0.0928   0.240      0.834  -0.481  -0.164     ## S ln(K)
  0.366   -0.000296   0.136   0.0164 -0.0000209 ## S^{-1}
 -0.555   -0.444      0.0989 -0.00817 0.00151   ## K^{-1}
]
