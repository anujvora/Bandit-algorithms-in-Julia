
function thom_samp_alg(arms,seed_val,hori)
    
    Random.seed!(seed_val)
    curr_time = 0
    num_arms = length(arms)
    arm_chosen = 0
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

        for a in all_arms_list
            # @show alpha_par
            gaussian_dist_samp[a] = rand(Normal(mean_par[a]+1,variance_par[a]+1))
        end
#            print(beta_dist_samp[curr_time])

        # first_argmax = argmax(beta_dist_samp[:])
        # all_argmax = argwhere(beta_dist_samp[:] == beta_dist_samp[curr_time,first_argmax])
        # all_argmax = transpose(all_argmax)
        arm_chosen = argmax(gaussian_dist_samp)

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
