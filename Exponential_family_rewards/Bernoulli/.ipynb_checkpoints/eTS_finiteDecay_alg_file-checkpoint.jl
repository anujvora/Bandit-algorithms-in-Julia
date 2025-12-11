
function eTS_finiteDecay_alg(arms,seed_val,hori)

    
    Random.seed!(seed_val)
    curr_time = 0
    num_arms = length(arms)
    arm_chosen = 1
    arms_index = 1:length(arms)
    # tot_rew =  zeros((hori,length(arms)))
    tot_rew =  zeros((hori,num_arms))
    rewards = zeros((hori,num_arms))
    times_sampled = zeros((num_arms))
    # times_sampled_hori = zeros((hori,length(arms)))
    alpha_par = zeros((num_arms))
    beta_par = zeros((num_arms))
    beta_dist_samp = zeros((num_arms))

    # all_arms_list = []
    all_arms_list = 1:num_arms
   
    # main body of algorithm
    curr_time = 2
    while curr_time <= hori

        alpha_par = tot_rew[curr_time-1,:] #successes
        beta_par = times_sampled .- tot_rew[curr_time-1,:] # failures
        
        bet = sqrt(curr_time/hori)
        bet_samp = rand(Binomial(1,bet)) ## prob of success is p, so prob of 1 is bet
        
        if bet_samp == 0
            # print(alpha_par[curr_time,:]+1,beta_par[curr_time,:]+1)
            beta_dist_samp = rand.(Beta.(alpha_par .+1,beta_par .+1))
            arm_chosen_beta = argmax(beta_dist_samp[:])
            arm_chosen = arm_chosen_beta
        
        elseif bet_samp == 1
            means_of_arms_dist = (alpha_par[:] .+ 1)./(alpha_par[:] .+1 + beta_par[:] .+ 1)
            arm_chosen_mu = argmax(means_of_arms_dist)
            arm_chosen = arm_chosen_mu

        end
        # print(arm_chosen)
        rewards[curr_time,arm_chosen] = rand(Binomial(1,arms[arm_chosen]))
        
        tot_rew[curr_time,:] = tot_rew[curr_time-1,:] + rewards[curr_time,:]
        # @show tot_rew[curr_time,:],rewards[curr_time,arm_chosen]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled

        # emp_means[curr_time] = divide(tot_rew[curr_time],times_sampled)
        curr_time = curr_time + 1
    end
    # @show sum(tot_rew,dims = 2), sum(tot_rew,dims = 2)[hori]
    return(sum(tot_rew,dims = 2),times_sampled) #,times_sampled_hori)

end