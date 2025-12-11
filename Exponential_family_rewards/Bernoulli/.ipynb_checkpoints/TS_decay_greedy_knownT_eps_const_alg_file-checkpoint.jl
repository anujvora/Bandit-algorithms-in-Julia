## Bernoulli
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

function compute_value_policy_TS_decay_greedy_knownT(policy, params) 

    # @show params[4]
    num_arms = params[1]
    prior_prob = params[2]
    arms_list = params[3]
    means_for_trans = params[4] #reshape(params[4],1,num_arms)
    coeff = params[5]
    # @show sum(means_for_trans.*policy)
    # p_arm0_optimal_prior = integrate.quad(expected_risk_general_arms, 0,1, args=(all_arms_list,0,alpha_par[curr_time],beta_par[curr_time-1]))[0]
    
    # opt_vec = identity(num_arms)

    policy = reshape(policy,num_arms,1)
    prior_prob = reshape(prior_prob,num_arms,1)
   
    # post_prob_rew1[:,1] = array([0.33,0.67])
    # print(policy)
    # coeff = sqrt(curr_time/horizon)
    # @show (la.I-policy[1:num_arms].*ones(num_arms,num_arms)).^2 ,sum((la.I-policy[1:num_arms].*ones(num_arms,num_arms)).^2,dims=1)
    policy_in_mat =  (policy[1:num_arms,1].*ones(num_arms,num_arms))
    # @show (la.I- policy_in_mat).^2
    # @show sum((la.I- policy_in_mat).^2,dims = 1)
    # @show (1-coeff)*( sum( (sum((la.I- policy_in_mat).^2,dims = 1))*prior_prob ))/2
    # @show (1-coeff)*( sum((sum((la.I- policy_in_mat).^2,dims = 1)).* prior_prob))/2,  (coeff)*sum(means_for_trans.* policy)
    value_policy = (  (1-coeff)*( sum( (sum((la.I- policy_in_mat).^2,dims = 1))*prior_prob ))/2  -  (coeff)*sum(means_for_trans.* policy) )
                    
                    
    # @show policy, value_policy
    return value_policy
end
##########################################################################
function TS_decay_greedy_knownT_eps_const(arms,seed_val,hori)
    
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
        
    for a in arms_index
        prob = IntegralProblem(expected_risk_general_arms, domain, (num_arms,all_arms_list,a,alpha_prior,beta_prior)) #,all_arms_list,a,alpha_prior,beta_prior)
        sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
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
        # ## optimal using Lagrange multipliers
        decay_coeff = (2/10)*(curr_time/hori)^2 
        # decay_coeff = 1- 1/(curr_time+1)^(0.05)#log(log(curr_time+16)) 
        if decay_coeff != 1
            old_lambda =  (decay_coeff/(1-decay_coeff)).*(means_of_arms_dist .- sum(means_of_arms_dist)/num_arms) 
            best_policy .= prob_opt_curr .+ old_lambda
            # println(best_policy)
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
