## Gamma distribution
function expected_risk_general_arms(nu,params)

    # @show params
    num_arms = params[1]
    arms_list = [i for i in params[2]]
    arm_chosen = params[3]
    alpha_par = params[4]
    beta_par = params[5]
    #print(total_rew,times_pulled,alpha_par,beta_par)

    #print(arms_list,arm_chosen)
    arms_small = copy(arms_list)
    #arms_small = [a for a in arms_list if a != arm_chosen]
    
    deleteat!(arms_small,arm_chosen)
    
    integrand_term = 1
    # @show arms_small, arm_chosen
    # for a in arms_small
    #     integrand_term = integrand_term*cdf(Beta(alpha_par[a]+1,beta_par[a]+1),nu)
    # end
    # @show nu, alpha_par, beta_par
    integrand_term = pdf(Gamma(alpha_par[arm_chosen],1/beta_par[arm_chosen]),nu)*prod(1 .- cdf.(Gamma.(alpha_par[arms_small] ,1 ./beta_par[arms_small]),nu))
    
    # integrand_term = integrand_term*pdf(Beta(alpha_par[arm_chosen]+1,beta_par[arm_chosen]+1),nu)
    #if arm == 0:
     #   print(integrand_term)
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
    
    for a in 1:num_arms
        prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_par,beta_par)) #,all_arms_list,a,alpha_prior,beta_prior)
        sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
        prob_opt[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
        #mean(Beta(alpha_par[a]+1,beta_par[a]+1))
    end

    return prob_opt
end
##########################################################################
function TS_decay_greedy_knownT_1_eps(arms,seed_val,hori)
    
    Random.seed!(seed_val)
    curr_time = 0
    #print(curr_model)

    num_arms = length(arms)

    arm_chosen = 1
    arms_index = 1:num_arms
    tot_rew =  zeros((hori,num_arms))
    rewards = zeros((hori,num_arms))
    times_sampled = zeros((num_arms))
    # times_sampled_hori = zeros((hori,num_arms))
   
    # threshold = zeros((hori,num_arms))#,dtype='float32')
    #prior = ones((hori,num_models))/num_models

    alpha_par = zeros(num_arms,1)
    beta_par = zeros(num_arms,1)

    means_of_arms_dist = zeros(num_arms,1)
    # all_arms_list = []
    all_arms_list = 1:num_arms

    prob_opt_curr = zeros(num_arms,1)

    alpha_prior = ones(num_arms)
    beta_prior = ones(num_arms)

    domain = (0, Inf) # (lb, ub)
        
    for a in arms_index
        prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_prior,beta_prior)) #,all_arms_list,a,alpha_prior,beta_prior)
        sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
        prob_opt_curr[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
    end

    x_0 = ones(num_arms)/num_arms

    for curr_time in 2:hori

        alpha_par = times_sampled .+ 1 #successes
        beta_par =  tot_rew[curr_time-1,:] .+ 1 # failures

        prob_opt_curr = compute_integral((domain, num_arms,alpha_par,beta_par))
        means_of_arms_dist = beta_par ./ alpha_par
        
        
        ###########################################################################
        # ## optimal using Projection algorithm
        decay_coeff = (curr_time/hori)^(2) 

        policy = zeros(num_arms)
        # decay_coeff = (curr_time/hori)^(2) 
        # policy = prob_opt_curr
        if decay_coeff != 1
            old_lambda =  (decay_coeff/(1)).*(means_of_arms_dist .- sum(means_of_arms_dist)/num_arms) 
            policy = prob_opt_curr .+ old_lambda
            # println(sum(policy))
            neg_ind_list = Int32[]
            remaining_ind = Int32[]
            # neg_ind_list = []
            while any(<(0),policy)
                min_ind_neg = argmin(policy)[1]
                neg_ind_list = push!(neg_ind_list,min_ind_neg) 
                remaining_ind = [i for i in 1:num_arms if !(i in neg_ind_list)]
                # println(remaining_ind,neg_ind_list)
                new_lambda =  (decay_coeff/(1)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind)
                policy = prob_opt_curr .+ new_lambda 
                policy[neg_ind_list] .= 0
            end
        else
            policy = zeros(num_arms)
            policy[argmax(means_of_arms_dist)] = 1      
        end
        ###################################################################

        best_policy = policy
        max_prob_ind = argmax(best_policy[1:num_arms])[1]
        best_policy[max_prob_ind] = best_policy[max_prob_ind] - (sum(best_policy) - 1)
       
        arm_chosen = rand(Categorical(best_policy[1:num_arms]))
   
        ###################################################################
        
        rewards[curr_time,arm_chosen] = rand(Gamma(arms[arm_chosen],1))
        # print(arm_chosen)#,rewards[curr_time,arm_chosen])
        tot_rew[curr_time,:] = tot_rew[curr_time-1,:] + rewards[curr_time,:]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled[:]
    end
    # @show sum(tot_rew,dims = 2)
    return(sum(tot_rew,dims = 2),times_sampled)
            
end