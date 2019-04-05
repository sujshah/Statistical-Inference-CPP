cd("/Users/localadmin/Documents/onderzoek/compound poisson/cpp-julia7")

using StatsBase
using Distributions



type priorspec
# prior for psi_j
        alpha0::Float64
        beta0::Float64
# prior for mean mu1
        xi1::Float64
# prior for mean mu2
        xi2::Float64
# prior for variance of mu1 and mu2, conditional on sigma2 equals kappa*sigma^2
        kappa::Float64
# gamma(A,B) prior on tau
        alpha1::Float64
        beta1::Float64
end

type mcmcspec
# sdPsi: standard deviation of noncentered MH update for psi
	sdPsi::Float64
# delta: Uniform(-delta, delta) update on joint noncentred psi
	delta::Float64
end

println("Reading cppFunctions...")

include("cppFunctions.jl")
include("cppMarg.jl")

print("\t done")
