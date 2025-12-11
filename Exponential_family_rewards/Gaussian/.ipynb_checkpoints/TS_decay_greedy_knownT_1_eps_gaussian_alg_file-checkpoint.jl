# Gaussian
function expected_risk_general_arms(nu,params)

    # @show params
    num_arms = params[1]
    arms_list = [i for i in params[2]]
    arm_chosen = params[3]
    mean_par = params[4]
    variance_par = params[5]
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
    integrand_term = pdf(Normal(mean_par[arm_chosen],variance_par[arm_chosen]),nu)*prod(cdf.(Normal.(mean_par[arms_small] ,variance_par[arms_small]),nu))
    # integrand_term = integrand_term*pdf(Beta(alpha_par[arm_chosen]+1,beta_par[arm_chosen]+1),nu)
    #if arm == 0:
     #   print(integrand_term)
    return integrand_term
end


##########################################################################
function TS_decay_greedy_knownT_1_eps(arms,seed_val,hori)
    
    Random.seed!(seed_val)
    curr_time = 0

    num_arms = length(arms)

    arm_chosen = 1
    arms_index = 1:num_arms
    tot_rew =  zeros((hori,num_arms))
    rewards = zeros((hori,num_arms))
    times_sampled = zeros((num_arms))
    # times_sampled_hori = zeros((hori,num_arms))
   
    # threshold = zeros((hori,num_arms))#,dtype='float32')
    #prior = ones((hori,num_models))/num_models

    mean_par = zeros((num_arms))
    variance_par = zeros((num_arms))

    means_of_arms_dist = zeros(num_arms)
    # all_arms_list = []
    all_arms_list = 1:num_arms

    discount = 1
    stop_time = 2
    # main body of algorithm
    # p_arm0_optimal_prior = integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,0,alpha_par[curr_time],beta_par[curr_time]))[0]

    prob_opt_curr = zeros(num_arms,1)

    mean_prior = ones(num_arms)
    variance_prior = ones(num_arms)

    domain = (-Inf, Inf) # (lb, ub)
        
    for a in arms_index
        prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,mean_prior,variance_prior)) #,all_arms_list,a,alpha_prior,beta_prior)
        sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
        prob_opt_curr[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
    end

    x_0 = ones(num_arms)/num_arms

    best_policy = ones(num_arms)/num_arms
    
    for curr_time in 2:hori

        mean_par[:] = tot_rew[curr_time-1,:]./(times_sampled .+ 1)
        variance_par[:] = (1 ./(times_sampled .+ 1)).^(1/2)

        # @show alpha_par, beta_par
        # domain = (0, 1)
        for a in arms_index
            prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,mean_par,variance_par)) #,all_arms_list,a,alpha_prior,beta_prior)
            sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
            prob_opt_curr[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
            means_of_arms_dist[a] = mean_par[a] #mean(Beta(alpha_par[a]+1,beta_par[a]+1))
        end

###########################################################################
        # ## optimal using Lagrange multipliers
        decay_coeff = (curr_time/hori)^2 
        # decay_coeff = 1- 1/(curr_time+1)^(0.05)#log(log(curr_time+16)) 
        if decay_coeff != 1
            old_lambda =  (decay_coeff/(1)).*(means_of_arms_dist .- sum(means_of_arms_dist)/num_arms) 
            best_policy .= prob_opt_curr .+ old_lambda
            # println(best_policy)
            neg_ind_list = Int32[]
            while any(<(0),best_policy)
                min_ind_neg = argmin(best_policy)[1]
                neg_ind_list = push!(neg_ind_list,min_ind_neg) 
                remaining_ind = [i for i in 1:num_arms if !(i in neg_ind_list)]
                # println(remaining_ind,neg_ind_list)
                # new_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind)
                best_policy .= prob_opt_curr .+ (decay_coeff/(1)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind) 
                best_policy[neg_ind_list] .= 0
                
            end
        else
            best_policy = zeros(num_arms)
            best_policy[argmax(means_of_arms_dist)] = 1        
        end
        ###################################################################
        
        max_prob_ind = argmax(best_policy[1:num_arms])[1]
        best_policy[max_prob_ind] = best_policy[max_prob_ind] - (sum(best_policy) - 1)
  
        # println(best_policy[1:num_arms],sum(best_policy))
        arm_chosen = rand(Categorical(best_policy[1:num_arms]))
        rewards[curr_time,arm_chosen] = rand(Normal(arms[arm_chosen],1))
        # print(arm_chosen)#,rewards[curr_time,arm_chosen])
        tot_rew[curr_time,:] = tot_rew[curr_time-1,:] + rewards[curr_time,:]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled[:]
    end
    # @show sum(tot_rew,dims = 2)
     return(sum(tot_rew,dims = 2))#,times_sampled)
            
end