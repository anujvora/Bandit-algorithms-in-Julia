function compute_probability_with_monte_carlo(params) 

   
    num_arms = params[1]
    alpha_par = params[2]
    beta_par = params[3]
    
    num_samples = 10000

    MC_sample = zeros(num_samples,num_arms)
    
    for a in 1:num_arms
        MC_sample[:,a] = rand(Beta(alpha_par[a]+1,beta_par[a]+1),num_samples)
    end
    # rewards = MC_sample
    max_arm_counter = counter(argmax.(eachrow(MC_sample)))
    prob_vec = zeros(num_arms)
    for i in keys(max_arm_counter)
        prob_vec[i] = max_arm_counter[i]/num_samples
    end
    # @show  prob_vec #, #values(max_arm_counter) 

    return prob_vec
end


##########################################################################
function value_policy_alg_TS_decay_greedy_knownT_MC_sim(arms,seed_val,hori)
    
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

    alpha_par = zeros((num_arms))
    beta_par = zeros((num_arms))

    means_of_arms_dist = zeros(num_arms)
    # all_arms_list = []
    all_arms_list = 1:num_arms

    discount = 1
    stop_time = 2
    # main body of algorithm
    # p_arm0_optimal_prior = integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,0,alpha_par[curr_time],beta_par[curr_time]))[0]

    prob_opt_curr = zeros(num_arms,1)

    alpha_prior = ones(num_arms)
    beta_prior = ones(num_arms)

    domain = (0, Inf) # (lb, ub)

    prob_opt_curr = compute_probability_with_monte_carlo((num_arms,alpha_prior,beta_prior)) #integrate.quad_vec(expected_risk_general_arms, 0,1, 
    # println(sum(prob_opt_curr))
        
    # bounds_for_opt = [(0,1) for i in all_arms_list]
    x_0 = ones(num_arms)/num_arms
    # x_0[1] = 1
    # lb = zeros(num_arms) #[-1.0, -1.0]
    # ub = ones(num_arms)
        
    # print(prob_opt_curr)
    best_policy = ones(num_arms)/num_arms
    
    for curr_time in 2:hori

        alpha_par = tot_rew[curr_time-1,:] .+ 1 #successes
        beta_par =  times_sampled .+ 1 # failures

        # @show alpha_par, beta_par
        # domain = (0, 1)
        prob_opt_curr = compute_probability_with_monte_carlo((num_arms,alpha_par,beta_par)) 
        means_of_arms_dist = alpha_par ./ beta_par #mean(Beta(alpha_par[a]+1,beta_par[a]+1))
        ###########################################################################
        # ## optimal using Projection algorithm
        decay_coeff = (curr_time/hori)^2 
        if decay_coeff != 1
            old_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist)/num_arms) 
            best_policy = prob_opt_curr .+ old_lambda
            # println(sum(best_policy))
            neg_ind_list = []
            while any(<(0),best_policy)
                min_ind_neg = argmin(best_policy)[1]
                neg_ind_list = push!(neg_ind_list,min_ind_neg) 
                remaining_ind = [i for i in 1:num_arms if !(i in neg_ind_list)]
                # println(remaining_ind,neg_ind_list)
                new_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind)
                best_policy = prob_opt_curr .+ new_lambda 
                best_policy[neg_ind_list] .= 0
                # for i in 1:num_arms
                #     if best_policy[i] < 0 
                #         new_lambda =  old_lambda + (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist[i])/(num_arms-1)) .+ prob_opt_curr[i]/(num_arms-1)
                #         best_policy = prob_opt_curr .+ new_lambda 
                #         best_policy[i] = 0 #prob_opt_curr[i] + (decay_coeff/(1-decay_coeff))*(means_of_arms_dist[i] - sum(means_of_arms_dist)/num_arms)
                
            end
        else
            best_policy[argmax(means_of_arms_dist)] = 1        
        end
        # println(round.(best_policy,sigdigits=3))

        max_prob_ind = argmax(best_policy[1:num_arms])[1]
        best_policy[max_prob_ind] = best_policy[max_prob_ind] - (sum(best_policy) - 1)
        # println(best_policy[1:num_arms],sum(best_policy))
        arm_chosen = rand(Categorical(best_policy[1:num_arms]))
        # if sum(best_policy) == 1
        #     # println(best_policy)
        #     arm_chosen = rand(Categorical(optim_policy))
        # end
        ###################################################################
        ###################################################################
        
        rewards[curr_time,arm_chosen] = rand(Binomial(1,arms[arm_chosen]))
        # print(arm_chosen)#,rewards[curr_time,arm_chosen])
        tot_rew[curr_time,:] = tot_rew[curr_time-1,:] + rewards[curr_time,:]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled[:]
    end
    # @show sum(tot_rew,dims = 2)
    return(sum(tot_rew,dims = 2),times_sampled)
            
end