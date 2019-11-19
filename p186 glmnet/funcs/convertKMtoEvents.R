convertKMtoEvents = function(KMprobs,max_chall,num_samples){
  
  surv_table = matrix(0,nrow=max_chall,ncol=2)
  surv_table[,1] = trunc(KMprobs * num_samples)
  surv_table[1,2] = num_samples - surv_table[1,1]
  for(timeIdx in 2:max_chall){
    
    surv_table[timeIdx,2] = surv_table[(timeIdx-1),1] - surv_table[timeIdx,1]
    
  }
  
  event_table = matrix(1,nrow=num_samples,ncol=2)
  colnames(event_table) = c('Challenges','censor')
  end_idx = cumsum(surv_table[,2])
  start_idx = 0
  
  for(stepsIdx in 1:max_chall){
    
    if(surv_table[stepsIdx,2]!=0){
      event_table[(start_idx+1):end_idx[stepsIdx],1] = stepsIdx
    }
    
    start_idx = end_idx[stepsIdx]
    
  }
  
  if(start_idx!=num_samples){
    
    event_table[(start_idx+1):num_samples,1] = max_chall
    event_table[(start_idx+1):num_samples,2] = 0
    
  }
  
  return(list('event_table'=event_table))
  
}