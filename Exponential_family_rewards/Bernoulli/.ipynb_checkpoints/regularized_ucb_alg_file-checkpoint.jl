## regularized ucb

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
    integrand_term = pdf(Beta(alpha_par[arm_chosen]+1,beta_par[arm_chosen]+1),nu)*prod(cdf.(Beta.(alpha_par[arms_small] .+ 1,beta_par[arms_small] .+ 1),nu))
    # integrand_term = integrand_term*pdf(Beta(alpha_par[arm_chosen]+1,beta_par[arm_chosen]+1),nu)
    #if arm == 0:
     #   print(integrand_term)
    return integrand_term
end


##########################################################################
function regularized_ucb(arms,seed_val,hori)
    
    Random.seed!(seed_val)
    curr_time = 0

    num_arms = length(arms)

    arm_chosen = 1
    arms_index = 1:num_arms
    tot_rew =  zeros((hori,num_arms))
    rewards = zeros((num_arms))
    times_sampled = zeros((num_arms))

    alpha_par = zeros((num_arms))
    beta_par = zeros((num_arms))

    means_of_arms_dist = zeros(num_arms)
    emp_means = zeros(num_arms)
    ucb = zeros(num_arms)
    # all_arms_list = []
    all_arms_list = 1:num_arms

    prob_opt_curr = zeros(num_arms,1)

    alpha_prior = ones(num_arms)
    beta_prior = ones(num_arms)

    curr_time = 1
    for a in 1:num_arms
        rewards[a] = rand(Binomial(1,arms[a]))
        tot_rew[curr_time,:] .= rewards
        times_sampled[a] = times_sampled[a] + 1
        rewards = 0 .* rewards
        curr_time += 1
    end
    
    x_0 = ones(num_arms)/num_arms
  
    best_policy = ones(num_arms)/num_arms
    for curr_time in 2:hori

        alpha_par .= tot_rew[curr_time-1,:] #successes
        beta_par .= times_sampled .- tot_rew[curr_time-1,:] # failures

        emp_means = alpha_par ./ times_sampled
        ucb .= emp_means .+ (2*log(curr_time) ./ times_sampled).^(1/2)

        domain = (0, 1)
        
        prob_opt_curr .= compute_integral((domain, num_arms,alpha_par,beta_par))
        means_of_arms_dist .= (alpha_par .+ 1) ./(alpha_par .+ beta_par .+ 2)
        
        # println(ucb)
        ###################################################################
      
        # # decay_coeff = 1-1/(1*log(log(curr_time+16)))

        # decay_coeff = 1-(curr_time/hori)^(2) 
        # best_policy = ucb ./ sum(ucb)
        # # @show curr_time,decay_coeff
        # if decay_coeff < 1 && decay_coeff != 0
        #     lifted_vector = ((1-decay_coeff)/decay_coeff).*ucb 
        #     sorted_indices = sortperm(-lifted_vector)
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
        #     best_policy[argmax(ucb)] = 1      
        # end

        # # @show best_policy
        
        # max_prob_ind = argmax(best_policy[1:num_arms])[1]
        # best_policy[max_prob_ind] = best_policy[max_prob_ind] - (sum(best_policy) - 1)
        # # println(best_policy[1:num_arms],sum(best_policy))
        # arm_chosen = rand(Categorical(best_policy[1:num_arms]))

        ###################################################################
        # ## optimal using Lagrange multipliers
        ###################################################################
        # ## optimal using Lagrange multipliers
        decay_coeff = (curr_time/hori)^(0.4) 
        # @show decay_coeff
        # decay_coeff = 1- 1/(curr_time+1)^(0.05)#log(log(curr_time+16)) 
        means_of_arms_dist .= ucb
        if decay_coeff != 1
            old_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist)/num_arms) 
            best_policy .= prob_opt_curr .+ old_lambda
            # println(sum(best_policy))
            neg_ind_list = Int32[]
            while any(<(0),best_policy)
                
                min_ind_neg = argmin(best_policy)[1]
                neg_ind_list = push!(neg_ind_list,min_ind_neg) 
                remaining_ind = [i for i in 1:num_arms if !(i in neg_ind_list)]
                # println(remaining_ind,neg_ind_list)
                # new_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind)
                best_policy .= prob_opt_curr .+ (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist[remaining_ind])/length(remaining_ind))  .+ sum(prob_opt_curr[neg_ind_list])/length(remaining_ind) 
                best_policy[neg_ind_list] .= 0
                
            end
        else
            best_policy = zeros(num_arms)
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
      
        # rewards[curr_time,arm_chosen] = rand(Binomial(1,arms[arm_chosen]))
        # tot_rew[curr_time,:] = tot_rew[curr_time-1,:] + rewards[curr_time,:]
        # times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1



        rewards[arm_chosen] = rand(Binomial(1,arms[arm_chosen]))
        tot_rew[curr_time,:] .= tot_rew[curr_time-1,:] .+ rewards
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1

        rewards = 0 .* rewards

    end
    # @show sum(tot_rew,dims = 2)
    return(sum(tot_rew,dims = 2),times_sampled)
            
end