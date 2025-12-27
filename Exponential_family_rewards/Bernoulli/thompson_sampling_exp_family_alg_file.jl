include("auxiliary_functions_exp_family.jl")

function thom_samp_alg(arms,seed_val,hori,dist_family)
    
    Random.seed!(seed_val)
    curr_time = 0
    num_arms = length(arms)
    arm_chosen = 1
    
    # PRE-ALLOCATE ALL ARRAYS (reduces allocation pressure significantly)
    tot_rew = zeros(hori)
    rewards = zeros(num_arms)
    times_sampled = zeros(Int32, num_arms)
    alpha_par = zeros(num_arms)
    beta_par = zeros(num_arms)
    dist_samp = zeros(num_arms)
   
    # main body of algorithm
    for curr_time in 2:hori
        # first_argmax = argmax(beta_dist_samp[:])
        # all_argmax = argwhere(beta_dist_samp[:] == beta_dist_samp[curr_time,first_argmax])
        # all_argmax = transpose(all_argmax)
        # Sample arm and get reward
        dist_samp = get_sample(alpha_par, beta_par, dist_family)
        arm_chosen = argmax(dist_samp)
        rewards[arm_chosen] = get_reward(arms, arm_chosen, dist_family)
        
        # Update rewards in-place
        tot_rew[curr_time] = tot_rew[curr_time - 1] + rewards[arm_chosen]
        times_sampled[arm_chosen] += 1

        (alpha_par[arm_chosen], beta_par[arm_chosen]) = update_pars(alpha_par[arm_chosen], beta_par[arm_chosen], rewards[arm_chosen], dist_family)
        rewards[arm_chosen] = 0.0

        previous_arm_chosen = arm_chosen
    end
    return (tot_rew, times_sampled)  

end
