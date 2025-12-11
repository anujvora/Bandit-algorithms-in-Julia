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
function compute_projection(params)

    decay_coeff = params[1]
    prob_opt_curr = params[2]
    means_of_arms_dist = params[3]
    
    policy = prob_opt_curr
    # @show curr_time,decay_coeff
    # if  decay_coeff > 0
    #     lifted_vector = prob_opt_curr .+ ((1-decay_coeff)/decay_coeff).*means_of_arms_dist 
    #     sorted_indices = sortperm(-lifted_vector,dims=1)
    #     # @show lifted_vector #, sorted_indices
    #     theta_term = 0 
    #     i = 1 
    #     while i <= num_arms
    #         theta_term = (1/i)*(sum(lifted_vector[sorted_indices[1:i]]) - 1)
    #         # @show  lifted_vector[sorted_indices[i]], theta_term
    #         if  lifted_vector[sorted_indices[i]] - theta_term <= 0
    #             # @show  i,lifted_vector[sorted_indices[i]], theta_term
    #             # fix the theta_term with the last index
    #             # @show prob_opt_curr, means_of_arms_dist#sorted_indices,i,sorted_indices[1:i-1]
    #             theta_term = (1/(i-1))*(sum(lifted_vector[sorted_indices[1:i-1]]) - 1)
    #             # @show i
    #             break
    #         end 
    #         i += 1
    #     end

    #     policy = max.(lifted_vector  .- theta_term ,0)
        
    # else 
    #     policy = zeros(num_arms)
    #     policy[argmax(means_of_arms_dist)] = 1      
    # end

    # ## optimal using Lagrange multipliers
        # decay_coeff = 1-1/(1*log(log(curr_time+16)))
    #     # decay_coeff = (curr_time/hori)^(2) 
    #     # policy = prob_opt_curr
    # if decay_coeff != 1
    #     old_lambda =  (1-decay_coeff/(decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist)/num_arms) 
    #     policy = prob_opt_curr .+ old_lambda
    #     # println(sum(policy))
    #     neg_ind_list = Int32[]
    #     remaining_ind = Int32[]
    #     # neg_ind_list = []
    #     while any(<(0),policy)
    #         min_ind_neg = argmin(policy)[1]
    #         neg_ind_list = push!(neg_ind_list,min_ind_neg) 
    #         remaining_ind = [i for i in 1:num_arms if !(i in neg_ind_list)]
    #         # println(remaining_ind,neg_ind_list)
    #         new_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind)
    #         policy = prob_opt_curr .+ new_lambda 
    #         policy[neg_ind_list] .= 0
    #     end
    # else
    #     policy = zeros(num_arms)
    #     policy[argmax(means_of_arms_dist)] = 1      
    # end
        # println(round.(policy,sigdigits=3))

    return policy
end
##########################################################################
function value_policy_alg_TS_decay_greedy_knownT(arms,seed_val,hori)
    
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

    
    # main body of algorithm
    # p_arm0_optimal_prior = integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,0,alpha_par[curr_time],beta_par[curr_time]))[0]

    prob_opt_curr = zeros(num_arms,1)

    alpha_prior = ones(num_arms)
    beta_prior = ones(num_arms)

    domain = (0, Inf) # (lb, ub)
        
    for a in arms_index
        prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_prior,beta_prior)) #,all_arms_list,a,alpha_prior,beta_prior)
        sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
        prob_opt_curr[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
    end
    # prop_opt_curr = compute_integral((domain, num_arms,alpha_prior,beta_prior))
    # means_of_arms_dist = beta_par ./ alpha_par
    # println(sum(prob_opt_curr))
        
    # bounds_for_opt = [(0,1) for i in all_arms_list]
    x_0 = ones(num_arms)/num_arms
    # x_0[1] = 1
    # lb = zeros(num_arms) #[-1.0, -1.0]
    # ub = ones(num_arms)
        
    # print(prob_opt_curr)
    
    for curr_time in 2:hori

        alpha_par = times_sampled .+ 1 #successes
        beta_par =  tot_rew[curr_time-1,:] .+ 1 # failures

        # @show alpha_par, beta_par
        # domain = (0, 1)

        prob_opt_curr = compute_integral((domain, num_arms,alpha_par,beta_par))
        means_of_arms_dist = beta_par ./ alpha_par
        
        # for a in arms_index
        #     prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_par,beta_par)) #,all_arms_list,a,alpha_prior,beta_prior)
        #     sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
        #     prob_opt_curr[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
        #     means_of_arms_dist[a] = beta_par[a]/alpha_par[a] #mean(Beta(alpha_par[a]+1,beta_par[a]+1))
        # end

        
        
        ###########################################################################
        # ## optimal using Projection algorithm
        decay_coeff = (curr_time/hori)^(2) 

        policy = zeros(num_arms)
        # decay_coeff = (curr_time/hori)^(2) 
        # policy = prob_opt_curr
        if decay_coeff != 1
            old_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist)/num_arms) 
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
                new_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind)
                policy = prob_opt_curr .+ new_lambda 
                policy[neg_ind_list] .= 0
            end
        else
            policy = zeros(num_arms)
            policy[argmax(means_of_arms_dist)] = 1      
        end
        # best_policy = compute_projection((decay_coeff,prob_opt_curr,means_of_arms_dist))
        # decay_coeff = 1-1/(1*log(log(curr_time+16)))
        
        # max_prob_ind = argmax(best_policy[1:num_arms])[1]
        # best_policy[max_prob_ind] = best_policy[max_prob_ind] - (sum(best_policy) - 1)
        # # println(best_policy[1:num_arms],sum(best_policy))
        # arm_chosen = rand(Categorical(best_policy[1:num_arms]))
        # bp_old = best_policy
        ###################################################################
        ###########################################################################
        
        best_policy = policy
        max_prob_ind = argmax(best_policy[1:num_arms])[1]
        best_policy[max_prob_ind] = best_policy[max_prob_ind] - (sum(best_policy) - 1)
       
        arm_chosen = rand(Categorical(best_policy[1:num_arms]))
        # # if sum(best_policy) == 1
        # #     # println(best_policy)
        # #     arm_chosen = rand(Categorical(optim_policy))
        # # end

        
        # println(sum((bp_old-best_policy).^2))
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