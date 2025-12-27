include("auxiliary_functions_exp_family.jl")
function eTS_alg(arms,seed_val,hori,dist_family)

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
    means_of_arms_dist = zeros(num_arms)

     # main body of algorithm
    for curr_time in 2:hori
        bet = 1-1/(1*log(log(curr_time+16)))
        
        # bet = 1-(1/(curr_time+1))
        bet_samp = rand(Binomial(1,bet)) ## prob of success is p, so prob of 1 is bet
        if bet_samp == 0
            # print(alpha_par[curr_time,:]+1,beta_par[curr_time,:]+1)
            dist_samp = get_sample(alpha_par, beta_par, dist_family)
            arm_chosen = argmax(dist_samp)
        
        elseif bet_samp == 1
            @. means_of_arms_dist = (alpha_par + 1) / (alpha_par + beta_par + 2)
            arm_chosen = argmax(means_of_arms_dist)
        end

        # print(arm_chosen)
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