## Bernoulli
include("auxiliary_functions_exp_family.jl")
#########################################################################

###############################################################
function compute_expec_Ra_and_a_opt(a, a_opt, num_arms, num_discrete_pts, nu_values, partial_expec_R, pdf_matrix, cdf_matrix)

    expec_Ra_and_a_opt = 0
    delta = nu_values[2]-nu_values[1]
    @inbounds for k in 1:num_discrete_pts
        prod_cdf = 1.0
        for l in 1:num_arms
            if l != a_opt && l!= a
                prod_cdf *= cdf_matrix[k, l]
            end
        end
        if a == a_opt 
            expec_Ra_and_a_opt += delta * nu_values[k] * pdf_matrix[k, a_opt] * prod_cdf 
        else
            expec_Ra_and_a_opt += delta * pdf_matrix[k, a_opt] * prod_cdf * partial_expec_R[k,a]
        end
    end
    return expec_Ra_and_a_opt
end
##########################################################################
function update_Mij_terms!(Mij_matrix, domain, num_arms, alpha_par, beta_par, prob_opt_arm, nu_values, partial_expec_R, pdf_matrix, cdf_matrix)

    # domain = params[1]
    # num_arms = params[2] 
    # alpha_par = params[3]
    # beta_par = params[4]
    # prob_opt = params[5]
    # nu_values = params[6]
    # partial_expec_R = params[7]
    # pdf_matrix = params[8]
    # cdf_matrix = params[9]
    num_discrete_pts = length(nu_values)
    # Mij_terms_mat =  zeros((num_arms,num_arms))

    for i in 1:num_arms
        for j in 1:num_arms
            Mij_matrix[i,j] = 0.0
            prob_j = prob_opt_arm[j]
            if prob_j > 1e-10
                Mij_matrix[i,j] = compute_expec_Ra_and_a_opt(i, j, num_arms, num_discrete_pts, nu_values, partial_expec_R, pdf_matrix, cdf_matrix,) / prob_j
            # else
            end
        end
    end
    # return Mij_matrix
end

#####################################################################
function update_mutual_info!(mutual_information, prob_opt_arm, Mij_matrix, means_of_arms_dist, num_arms, KL_divergence)

    ## Computes the mutual information for Beta-Bernoulli 
    # prob_opt_arm = params[1]
    # Mij_matrix = params[2]
    # means_of_arms_dist = params[3]
    # num_arms = params[4]
    # KL_divergence = params[5]

    ## Beta-Bernoulli
    # KL = KullbackLeibler()
    for i in 1:num_arms 
        # M_ij_val = max.(0.001, min.(0.999, Mij_matrix[i,:]))  # Bound away from 0 and 1
        mean_i = max(0.001, min(0.999, means_of_arms_dist[i]))
        # @. KL_divergence[i,:] = M_ij_val*log(M_ij_val/mean_i) + (1-M_ij_val)*log((1-M_ij_val)/(1-mean_i)) #M_ij_val*log.(M_ij_val ./ mean_i) + (1 .- M_ij_val)*log.((1 .- M_ij_val) ./(1-mean_i))
        for j in 1:num_arms
            # Compute KL divergence between M_ij and posterior mean
            M_ij_val = max(0.001, min(0.999, Mij_matrix[i,j]))  # Bound away from 0 and 1
            # KL_divergence[i,j] = KL([M_ij_val, 1.0 - M_ij_val], [mean_j, 1.0 - mean_j])
            KL_divergence[i,j] = M_ij_val*log(M_ij_val/mean_i) + (1-M_ij_val)*log((1-M_ij_val)/(1-mean_i))
        end
        # Mutual information: weighted sum of KL divergences weighted by prob of optimal arm
        MI_sum = 0
        for j in 1:num_arms
            MI_sum += prob_opt_arm[j] * KL_divergence[i,j]
        end
        mutual_information[i] = MI_sum #sum(prob_opt_arm .* KL_divergence[i,:])
    end
end

#####################################################################
function update_optimal_policy!(sampling_policy, expec_best_R, means_of_arms_dist, mutual_information, num_arms, q_discretize, objective_func_discrete_pts, Delta_terms)

    # Compute regret gaps
    # Delta_terms = zeros(num_arms)
    # sampling_policy = zeros(num_arms)
    @. Delta_terms = expec_best_R - means_of_arms_dist
    
    # Use IDS objective: optimize allocation over all arms using Lagrangian approach
    # For numerical stability, use KL divergences weighted by policy
    
    min_objective_func = Inf
    q_best = 0.5
    
    # Find best two-arm allocation
    best_i, best_j = 1, 2
    
    # @allocated begin
        @inbounds for i in 1:num_arms
            for j in i+1:num_arms
                # Avoid division by zero - check mutual information values
                # if mutual_information[i] > 1e-10 || mutual_information[j] > 1e-10
                @. objective_func_discrete_pts = (q_discretize * Delta_terms[i] + (1-q_discretize) * Delta_terms[j])^2 / (max(1e-10, q_discretize * mutual_information[i] + (1-q_discretize) * mutual_information[j]))
                # for (k, q_val) in enumerate(q_discretize) #1:num_discrete_pts
                #     # @show Delta_terms
                #     # @show size(q_discretize)
                #     objective_func_discrete_pts[k] = (q_val * Delta_terms[i] + (1-q_val) * Delta_terms[j])^2 / (max(1e-10, q_val * mutual_information[i] + (1-q_val) * mutual_information[j]))
                # end
                min_obj = minimum(objective_func_discrete_pts)
                if min_obj < min_objective_func
                    best_i, best_j = i, j
                    min_objective_func = min_obj
                    q_best = q_discretize[argmin(objective_func_discrete_pts)]
                end
                # end
            end
        end
    # end
    # Compute optimal q for best arm pair
    # @. objective_func_discrete_pts = (q_discretize * Delta_terms[best_i] + (1-q_discretize) * Delta_terms[best_j])^2 / (max(1e-10, q_discretize * mutual_information[best_i] + (1-q_discretize) * mutual_information[best_j]))
    # q_opt = q_discretize[argmin(objective_func_discrete_pts)]
    # @show q_best, q_discretize
    sampling_policy[best_i] = q_best
    sampling_policy[best_j] = 1.0 - q_best

    
    # Normalize to ensure valid probability distribution
    policy_sum = sum(sampling_policy)
    if policy_sum > 0
        @. sampling_policy = sampling_policy / policy_sum
    else
        sampling_policy[1] = 1.0
    end
    
    # return sampling_policy
end
##########################################################################
function IDS(arms,seed_val,hori,dist_family)
    Random.seed!(seed_val)
    curr_time = 0
    num_arms = length(arms)
    arm_chosen = 1
    domain = (0, 1) # (lb, ub)
    previous_arm_chosen = 0
    num_discrete_pts = 200
    
    # PRE-ALLOCATE ALL ARRAYS (reduces allocation pressure significantly)
    tot_rew = zeros(hori)
    rewards = zeros(num_arms)
    times_sampled = zeros(Int32, num_arms)
    alpha_par = zeros(num_arms)
    beta_par = zeros(num_arms)
    means_of_arms_dist = zeros(num_arms)
    prob_opt_arm = zeros(num_arms)
    # prob_opt_flat = zeros(num_arms)
    partial_expec_R = zeros(num_discrete_pts, num_arms)
    best_policy = zeros(num_arms)
    
    
    nu_values = range(domain[1], domain[2], num_discrete_pts)
    # nu_values = Vector(domain[1]: 1/(num_discrete_pts-1) :domain[2])
    pdf_matrix = zeros(num_discrete_pts, num_arms)
    cdf_matrix = zeros(num_discrete_pts, num_arms)
    Mij_matrix = zeros(num_arms,num_arms)
    mutual_information = zeros(num_arms)
    KL_divergence = zeros((num_arms,num_arms))
    mutual_information = zeros(num_arms)
    Delta_terms = zeros(num_arms)

    num_discrete_pts_q = 50
    q_discretize = range(0, 1, num_discrete_pts_q)
    # q_discretize = Vector(0:1/(num_discrete_pts_q-1):1)
    # @show size(nu_values), size(q_discretize)
    objective_func_discrete_pts = zeros(num_discrete_pts_q)

    x_0 = ones(num_arms)/num_arms
    best_policy_init = ones(num_arms)/num_arms
    best_policy .= best_policy_init
    # @show tot_rew, hori, #rewards
    # Initial exploration phase: sample each arm at least once
    for init_t in 1:num_arms

        rewards[init_t] = get_reward(arms, init_t, dist_family)
        if init_t == 1
            tot_rew[init_t] = rewards[init_t]
        else
            # @show tot_rew, rewards
            tot_rew[init_t] = tot_rew[init_t-1] + rewards[init_t]
        end
        times_sampled[init_t] = 1
        (alpha_par[init_t], beta_par[init_t]) = update_pars(alpha_par[init_t], beta_par[init_t], rewards[init_t], dist_family)
        rewards[init_t] = 0.0
    end
    
    for curr_time in num_arms+1:hori

        # First, compute means of arm distributions (posterior means)
        @. means_of_arms_dist = (alpha_par + 1) / (alpha_par + beta_par + 2)

        # Pre-compute ALL PDF and CDF values once
        if previous_arm_chosen == 0
            (pdf_matrix, cdf_matrix) =  get_all_pdf_cdf_values(num_arms, nu_values, alpha_par, beta_par, dist_family, pdf_matrix, cdf_matrix)
        else
            update_one_pdf_cdf_values!(pdf_matrix, cdf_matrix, previous_arm_chosen, num_arms, nu_values, alpha_par, beta_par, dist_family)
        end
    
        # Compute integral in-place to avoid allocation
        # prob_opt_arm .= 
        update_optimal_arm_prob_discrete!(prob_opt_arm, domain, num_arms, alpha_par, beta_par, dist_family, previous_arm_chosen, pdf_matrix, cdf_matrix, num_discrete_pts)
        update_expec_discrete!(partial_expec_R, domain, num_arms, pdf_matrix, num_discrete_pts, nu_values)
        update_Mij_terms!(Mij_matrix, domain, num_arms, alpha_par, beta_par, prob_opt_arm, nu_values, partial_expec_R, pdf_matrix, cdf_matrix)
        update_mutual_info!(mutual_information, prob_opt_arm, Mij_matrix, means_of_arms_dist, num_arms, KL_divergence)
        expec_best_R = 0
        for i in 1:num_arms
            expec_best_R += prob_opt_arm[i] * Mij_matrix[i,i]
        end
        
        # Compute optimal sampling policy
        fill!(best_policy,0.0) 
        # @time begin
        update_optimal_policy!(best_policy, expec_best_R, means_of_arms_dist, mutual_information, num_arms, q_discretize, objective_func_discrete_pts, Delta_terms)
        # end
        # Sample arm and get reward
        # @show best_policy
        arm_chosen = rand(Categorical(best_policy))
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
