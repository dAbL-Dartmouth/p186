# ******************************************************************************
# Author      : Srivamshi Pittala
# Advisor     : Prof. Chris Bailey-Kellogg
# Project     : Profectus T2
# Decription  : Helper function to extract probabilities from KM curves
# ******************************************************************************

extractProbabilityFromKM = function(survival_rates,time_km,max_chall){
  
  prev_prob = 1
  follower_idx = 1
  
  km_obs = numeric(max_chall)
  
  for(chall_idx in 1:max_chall){
    
    if(chall_idx == time_km[follower_idx]){
      
      km_obs[chall_idx] = survival_rates[follower_idx]
      prev_prob = survival_rates[follower_idx]
      follower_idx = follower_idx + 1
      
    }else{
      
      km_obs[chall_idx] = prev_prob
      
    }
    
  }
  
  return(km_obs)
  
}