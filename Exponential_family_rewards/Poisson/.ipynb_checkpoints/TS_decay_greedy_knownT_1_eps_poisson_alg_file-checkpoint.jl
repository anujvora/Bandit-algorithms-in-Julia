# Poisson
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
    
    integrand_term = pdf(Gamma(alpha_par[arm_chosen],1/beta_par[arm_chosen]),nu)*prod(cdf.(Gamma.(alpha_par[arms_small] ,1 ./beta_par[arms_small]),nu))
   
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
        
    for a in arms_index
        prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_prior,beta_prior)) #,all_arms_list,a,alpha_prior,beta_prior)
        sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
        prob_opt_curr[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
    end
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

        prob_opt_curr = compute_integral((domain, num_arms,alpha_par,beta_par))
        means_of_arms_dist = alpha_par ./ beta_par 
            
        # for a in arms_index
        #     prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_par,beta_par)) #,all_arms_list,a,alpha_prior,beta_prior)
        #     sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
        #     prob_opt_curr[a,1] = sol.u #integrate.quad_vec(expected_risk_general_arms, 0,1, args=(all_arms_list,a,alpha_prior,beta_prior))[0]
        #     means_of_arms_dist[a] = alpha_par[a]/beta_par[a] #mean(Beta(alpha_par[a]+1,beta_par[a]+1))
        # end

        ###########################################################################
        # # ## optimal using Projection algorithm
        # decay_coeff = 1-(curr_time/hori)^(2) 
        # # decay_coeff = 1-1/(1*log(log(curr_time+16)))
     
        # best_policy = prob_opt_curr
        # # @show curr_time,decay_coeff
        # if decay_coeff < 1 && decay_coeff != 0
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

        #     best_policy[sorted_indices[1:i-1]] .= lifted_vector[sorted_indices[1:i-1]] .- theta_term 
        #     best_policy[sorted_indices[i:end]] .= 0
        #     # @show sorted_indices,best_policy, sum(best_policy) #, length(best_policy)

        # else 
        #     best_policy = zeros(num_arms)
        #     best_policy[argmax(means_of_arms_dist)] = 1      
        # end
        
        # max_prob_ind = argmax(best_policy[1:num_arms])[1]
        # best_policy[max_prob_ind] = best_policy[max_prob_ind] - (sum(best_policy) - 1)
        # # println(best_policy[1:num_arms],sum(best_policy))
        # arm_chosen = rand(Categorical(best_policy[1:num_arms]))
        
        ###################################################################
        ###########################################################################
        ## optimal using Lagrange multipliers
        # decay_coeff = 1-1/(1*log(log(curr_time+16)))
        decay_coeff = (curr_time/hori)^(1/2) 
        if decay_coeff != 1
            old_lambda =  (decay_coeff/(1)).*(means_of_arms_dist .- sum(means_of_arms_dist)/num_arms) 
            best_policy = prob_opt_curr .+ old_lambda
            # println(sum(best_policy))
            neg_ind_list = Int32[]
            # neg_ind_list = []
            while any(<(0),best_policy)
                min_ind_neg = argmin(best_policy)[1]
                neg_ind_list = push!(neg_ind_list,min_ind_neg) 
                remaining_ind = [i for i in 1:num_arms if !(i in neg_ind_list)]
                # println(remaining_ind,neg_ind_list)
                new_lambda =  (decay_coeff/(1)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind)
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
        
        rewards[curr_time,arm_chosen] = rand(Poisson(arms[arm_chosen]))
        # print(arm_chosen)#,rewards[curr_time,arm_chosen])
        tot_rew[curr_time,:] = tot_rew[curr_time-1,:] + rewards[curr_time,:]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled[:]
    end
    # @show sum(tot_rew,dims = 2)
    return(sum(tot_rew,dims = 2))
            
end