# inference.jl
include("base.jl")

"""
    gp_posterior(X_train, y_train, X_test, kernel_func=rbf_kernel, noise_var=0.1, kernel_params...)

Compute the posterior distribution of a Gaussian process at test points X_test.
Returns the mean and covariance matrix of the posterior.
"""
function gp_posterior(X_train, y_train,
                      X_test,
                      kernel_func=rbf_kernel, noise_var=0.1, kernel_params...)
    #TODO 
    return posterior_mean, posterior_cov
end

"""
    predict_with_uncertainty(X_train, y_train, X_test, kernel_func=rbf_kernel, noise_var=0.1, kernel_params...)

Make predictions with uncertainty bounds using a Gaussian process.
Returns predicted means and standard deviations.
"""
function predict_with_uncertainty(X_train, y_train,
                                  X_test,
                                  kernel_func=rbf_kernel, noise_var=0.1,
                                  kernel_params...)
    #TODO:
    return posterior_mean, posterior_std
end