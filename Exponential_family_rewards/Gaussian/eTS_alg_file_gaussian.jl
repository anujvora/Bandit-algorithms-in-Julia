
function eTS_alg(arms,seed_val,hori)
    
    Random.seed!(seed_val)
    curr_time = 0
    num_arms = length(arms)
    arm_chosen = 1
    arms_index = 1:length(arms)
    # tot_rew =  zeros((hori,length(arms)))
    tot_rew =  zeros((hori,num_arms))
    rewards = zeros((hori,length(arms)))
    times_sampled = zeros((num_arms))
    # times_sampled_hori = zeros((hori,length(arms)))
    mean_par = zeros((num_arms))
    variance_par = zeros((num_arms))
    gaussian_dist_samp = zeros((num_arms))
    #kl_ucb = zeros((hori,length(arms)))
    #emp_means =  zeros((hori,length(arms)))

    # threshold = zeros((hori,num_arms))

    # all_arms_list = []
    all_arms_list = 1:num_arms
   
    # main body of algorithm
    curr_time = 2
    while curr_time <= hori

        mean_par[:] = tot_rew[curr_time-1,:]./(times_sampled .+ 1)
        variance_par[:] = (1 ./(times_sampled .+ 1)).^(1/2)

		bet = 1-1/(1*log(log(curr_time+16)))
        bet_samp = rand(Binomial(1,bet)) ## prob of success is p, so prob of 1 is bet
        
        if bet_samp == 0
            # for a in all_arms_list
            #     # @show alpha_par
            #     gaussian_dist_samp[a] = rand(Normal(mean_par[a]+1,variance_par[a]+1))
            # end
            gaussian_dist_samp = rand.(Normal.(mean_par,variance_par))
            arm_chosen = argmax(gaussian_dist_samp)
        elseif bet_samp == 1
            # means_of_arms_dist = (alpha_par[:]+1)/(alpha_par[:]+1+beta_par[:]+1)
            arm_chosen_mu = argmax(mean_par[:])
            arm_chosen = arm_chosen_mu
#            print(beta_dist_samp[curr_time])
        end

        # @show arm_chosen
        rewards[curr_time,arm_chosen] = rand(Normal(arms[arm_chosen],1))
        
        tot_rew[curr_time,:] = tot_rew[curr_time-1,:] + rewards[curr_time,:]
        # @show tot_rew[curr_time,:],rewards[curr_time,arm_chosen]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled

        # emp_means[curr_time] = divide(tot_rew[curr_time],times_sampled)
        curr_time = curr_time + 1
    end
    # @show sum(tot_rew,dims = 2), sum(tot_rew,dims = 2)[hori]
    return(sum(tot_rew,dims = 2)) #,times_sampled_hori)

end

