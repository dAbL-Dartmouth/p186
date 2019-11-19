# ******************************************************************************
# Author      : Srivamshi Pittala
# Advisor     : Prof. Chris Bailey-Kellogg
# Project     : Profectus T2
# Decription  : Helper function for backward feature selection
# Cite        : TBD
# ******************************************************************************

takeOffOneFeat = function(feats,survObject,LL_prev,feat_idx_list,plots,dir_res){
  
  #cat('In takeOffOneFeat\n')
  
  numFeat = ncol(feats)
  
  res = matrix(0,nrow=numFeat,ncol=2)
  colnames(res) = c('LL_null','LL_opt')
  rownames(res) = colnames(feats)
  
  for(featIdx in 1:numFeat){
    
    feats_sub = feats[,-featIdx]
    
    cox_model = coxph(survObject~.,data=feats_sub)
    
    res[featIdx,'LL_null'] = cox_model$loglik[1]
    res[featIdx,'LL_opt'] = cox_model$loglik[2]
    
  }
  
  rm_idx = which.max(res[,'LL_opt'])
  
  if(plots){
    
    pdf(paste(dir_res,'sel_',numFeat,sep=""))
    
    yrange = range(LL_prev,res[,'LL_opt'])
    plot(0,type='n',xlim=c(1,numFeat),ylim=c(yrange[1],yrange[2]),main=paste('Removing ',rownames(res)[rm_idx]," (",rm_idx,") ",sep=""))
    lines(1:numFeat,res[,'LL_opt'],type='b',col='black')
    lines(1:numFeat,rep(LL_prev,numFeat),type='l',col='blue')
    
    dev.off()
    
  }
  
  #cat(rm_idx,length())
  return(list('res'=res[rm_idx,],'feats'=feats[,-rm_idx],'feat_idx_list'=feat_idx_list[-rm_idx]))
  
}