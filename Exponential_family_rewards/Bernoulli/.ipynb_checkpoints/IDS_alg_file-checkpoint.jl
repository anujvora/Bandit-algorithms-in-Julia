## Bernoulli
function expected_risk_general_arms(nu,params)

    # @show params
    num_arms = params[1]
    arms_list = [i for i in params[2]]
    arm_chosen = params[3]
    alpha_par = params[4]
    beta_par = params[5]

    arms_small = copy(arms_list)
    deleteat!(arms_small,arm_chosen)
    
    integrand_term = 1
    integrand_term = pdf(Beta(alpha_par[arm_chosen]+1,beta_par[arm_chosen]+1),nu)*prod(cdf.(Beta.(alpha_par[arms_small] .+ 1,beta_par[arms_small] .+ 1),nu))
   
    return integrand_term
end

##########################################################################
function compute_integral(params)

    domain = params[1]
    num_arms = params[2] 
    alpha_par = params[3]
    beta_par = params[4]
    
    all_arms_list = 1:num_arms 
    prob_opt = zeros(num_arms,1)
    tolerances = 1e-2
    for a in 1:num_arms
        prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_par,beta_par)) #,all_arms_list,a,alpha_prior,beta_prior)
        sol = solve(prob, HCubatureJL(); reltol = tolerances, abstol = tolerances)
        prob_opt[a,1] = sol.u 
    end

    return prob_opt
end

##########################################################################
function Mij_integrand(nu,params)

    # @show params
    num_arms = params[1]
    arms_list = [i for i in params[2]]
    arm_optimal = params[3]
    arm_expec = params[4]
    alpha_par = params[5]
    beta_par = params[6]

    arms_small = copy(arms_list)
    tolerances = 1e-2
    Mij_integrand_term = 1
    if arm_expec == arm_optimal
        deleteat!(arms_small,arm_optimal)
        Mij_integrand_term = pdf(Beta(alpha_par[arm_optimal]+1,beta_par[arm_optimal]+1),nu)*prod(cdf.(Beta.(alpha_par[arms_small] .+ 1,beta_par[arms_small] .+ 1),nu))
    else
        deleteat!(arms_small,sort([arm_optimal,arm_expec]))
        # @show arms_small,[arm_optimal,arm_expec]
        expec_function(x,p)  = (x <= nu)*x
        prob = IntegralProblem(expec_function,(0,nu)) 
        sol = solve(prob, HCubatureJL(); reltol = tolerances, abstol = tolerances) ## HCubatureJL
        # expec_term = sol.u
        
        beta_expec = Beta(alpha_par[arm_expec]+1,beta_par[arm_expec]+1)
        Mij_integrand_term = pdf(Beta(alpha_par[arm_optimal]+1,beta_par[arm_optimal]+1),nu)*prod(cdf.(Beta.(alpha_par[arms_small] .+ 1,beta_par[arms_small] .+ 1),nu))*sol.u #*expectation(x -> (x <= nu)*x, beta_expec)
        # @show expectation(x -> (x <= nu)*x, beta_expec)
        ## expectation(x --> (x <= nu)*x, beta_expec) --- computes \int I{x <= nu} x f_j(x)
        ## indicator function used for the limits of the integral
    end
    
    return Mij_integrand_term
end
    
##########################################################################
function compute_Mij_terms(params)

    domain = params[1]
    num_arms = params[2] 
    alpha_par = params[3]
    beta_par = params[4]
    prob_opt = params[5]
    
    all_arms_list = 1:num_arms 
    
    Mij_terms_mat =  zeros((num_arms,num_arms))

    tolerances = 1e-2
    for i in 1:num_arms
        prob = IntegralProblem(Mij_integrand, domain, (num_arms,all_arms_list,i,i,alpha_par,beta_par)) 
        sol = solve(prob, HCubatureJL(); reltol = tolerances, abstol = tolerances) ## HCubatureJL
        Mij_terms_mat[i,i] = min(sol.u/prob_opt[i] ,1)
        for j in i+1:num_arms
            prob = IntegralProblem(Mij_integrand, domain, (num_arms,all_arms_list,i,j,alpha_par,beta_par)) 
            sol = solve(prob, HCubatureJL(); reltol = tolerances, abstol = tolerances)
            Mij_terms_mat[i,j] = min(sol.u/prob_opt[i],1) 
            
            # if isnan(sol.u)
            #     println("y")
            #     println(alpha_par,beta_par)
            # end
            #     Mij_terms_mat[i,j] = 1/2 
            # else
            #     Mij_terms_mat[i,j] = sol.u/prob_opt[i] 
            # end
            
            # @show sol.u
            prob = IntegralProblem(Mij_integrand, domain, (num_arms,all_arms_list,j,i,alpha_par,beta_par)) 
            sol = solve(prob, HCubatureJL(); reltol = tolerances, abstol = tolerances)
            Mij_terms_mat[j,i] = min(sol.u/prob_opt[j],1)
            # @show isnan(sol.u)
            # if isnan(sol.u)
            #     Mij_terms_mat[j,i] = 1/2 
            # else
            #     Mij_terms_mat[j,i] = sol.u/prob_opt[j] 
            # end 
            # @show sol.u
        end
    end
    # @show maximum(Mij_terms_mat)
    return Mij_terms_mat
end

#####################################################################
function compute_mutual_info_terms(params)

    prob_opt_curr = params[1]
    Mij_matrix = params[2]
    means_of_arms_dist = params[3]
    num_arms = params[4]
    
    KL_divergence = zeros((num_arms,num_arms))
    mutual_info = zeros(num_arms)
    
    for i in 1:num_arms
        # @show Mij_matrix
        # @show Mij_matrix[:,i]./means_of_arms_dist[i], (1 .- Mij_matrix[:,i])./(1 .- means_of_arms_dist[i])
        # @show [Mij_matrix[:,i],1 .-Mij_matrix[:,i]]
        # @show KL_divergence[i,:]
        for j in 1:num_arms
            KL_divergence[i,j] = Flux.kldivergence([Mij_matrix[j,i],1 .- Mij_matrix[j,i]], [means_of_arms_dist[i], 1 .- means_of_arms_dist[i]])
        end
        
        #Mij_matrix[:,i].*log.(Mij_matrix[:,i]./means_of_arms_dist[i]) .+  (1 .-Mij_matrix[:,i]).*log.((1 .- Mij_matrix[:,i])./(1 .- means_of_arms_dist[i]))
        
        mutual_info[i] = sum(prob_opt_curr .*  KL_divergence[i,:])

        # @show  KL_divergence[i,:]
        # @show ((1 .- Mij_matrix[:,i])./(1 .- means_of_arms_dist[i]))
    end
    # @show round.(Mij_matrix,sigdigits=3)
    # @show round.(KL_divergence,sigdigits=3)

    return mutual_info 
end
##########################################################################
function compute_best_policy(params)
    instant_regret = params[1]
    mutual_info = params[2]
    num_arms = params[3]
    best_policy = zeros(num_arms)

    optimal_val = Inf
    for i in 1:num_arms
        for j in i+1:num_arms
            # # @show mutual_info, instant_regret
            IDS_pair(q,p) = (q[1]*instant_regret[i] + (1-q[1])*instant_regret[j])/(q[1]*mutual_info[i] + (1-q[1])*mutual_info[j])
            opt_func = OptimizationFunction(IDS_pair, Optimization.AutoForwardDiff())
            prob = OptimizationProblem(opt_func, ones(1)/2, [0],lb = [0], ub = [1.0])
            
            result = solve(prob, BFGS()) #optimize(IDS_pair,zeros(1)/2)  #,BFGS())
            # @show result.u[1],result.objective
            if result.objective <= optimal_val 
                best_policy = zeros(num_arms)
                optimal_val = result.objective
                best_policy[i] = result.u[1]
                best_policy[j] = 1-best_policy[i]
            end
        end
    end
    # @show best_policy
    return best_policy
end
##########################################################################
function IDS(arms,seed_val,hori)
    
    Random.seed!(seed_val)
    curr_time = 0
    #print(curr_model)
    num_arms = length(arms)

    arm_chosen = 1
    arms_index = 1:num_arms
    tot_rew =  zeros((hori,num_arms))
    rewards = zeros((num_arms))
    times_sampled = zeros((num_arms))

    alpha_par = zeros((num_arms))
    beta_par = zeros((num_arms))

    means_of_arms = zeros(num_arms)
    means_of_arms_dist = zeros(num_arms)
    # all_arms_list = []
    all_arms_list = 1:num_arms

    prob_opt_curr = zeros(num_arms,1)

    alpha_prior = ones(num_arms)
    beta_prior = ones(num_arms)

    domain = (0, 1) # (lb, ub)

    tolerances = 1e-2
    for a in arms_index
        prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_prior,beta_prior)) #,all_arms_list,a,alpha_prior,beta_prior)
        sol = solve(prob, HCubatureJL(); reltol = tolerances, abstol = tolerances)
        prob_opt_curr[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
    end

    x_0 = ones(num_arms)/num_arms
  
    best_policy = ones(num_arms)/num_arms
    for curr_time in 2:hori

        alpha_par .= tot_rew[curr_time-1,:] #successes
        beta_par .= times_sampled .- tot_rew[curr_time-1,:] # failures

        domain = (0, 1)
        prob_opt_curr .= compute_integral((domain, num_arms,alpha_par,beta_par))
        # means_of_arms_dist .= (alpha_par .+ 1) ./(alpha_par .+ beta_par .+ 2)
        means_of_arms .= (alpha_par .+ 1) ./(alpha_par .+ beta_par .+ 2)
        means_of_arms_dist .= means_of_arms #./sum(means_of_arms)
        ###################################################################
        ## Determining the optimal policy using the IDS algorithm
    
        Mij_matrix = compute_Mij_terms((domain,num_arms,alpha_par,beta_par,prob_opt_curr))
        
        mutual_info = compute_mutual_info_terms((prob_opt_curr,Mij_matrix,means_of_arms_dist,num_arms))

        # @show round.(mutual_info,sigdigits=3)
        # @show round.(means_of_arms_dist,sigdigits=3)
        # @show round.(prob_opt_curr,sigdigits=3)
        instant_regret = sum(prob_opt_curr.*la.diag(Mij_matrix)) .- means_of_arms_dist
        
        best_policy = compute_best_policy((instant_regret,mutual_info,num_arms))
            
        # if curr_time % 100 == 0:
        #     println("best",round.(best_policy,sigdigits=3))
        #     println("post",round.(prob_opt_curr,sigdigits=3))
        #     println("mean",round.(means_of_arms,sigdigits=3))
        #     # println("unpr",prob_opt_curr .+ old_lambda)
        #     println("\n")
        max_prob_ind = argmax(best_policy[1:num_arms])[1]
        best_policy[max_prob_ind] = best_policy[max_prob_ind] - (sum(best_policy) - 1)
        # println(best_policy[1:num_arms],sum(best_policy))
        arm_chosen = rand(Categorical(best_policy[1:num_arms]))

        rewards[arm_chosen] = rand(Binomial(1,arms[arm_chosen]))
        tot_rew[curr_time,:] .= tot_rew[curr_time-1,:] .+ rewards
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1

        rewards = 0 .* rewards

    end
    # @show sum(tot_rew,dims = 2)
    return(sum(tot_rew,dims = 2),times_sampled)
            
end
