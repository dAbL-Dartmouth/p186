# ******************************************************************************
# Author      : Srivamshi Pittala
# Advisor     : Prof. Chris Bailey-Kellogg
# Project     : NIH 10-332
# Description : Performs binomial logistic classification using elastic net
# Cite        : TBD
# ******************************************************************************

glmnetBiClass = function(feats,predVec,weights,numFeat,intc,alpha,cvFolds,repeatRun){
  
  augMatrix = cbind(feats,predVec)
  
  # if class label is NA, remove the sample
  rmList = which(is.na(predVec))
  if(length(rmList)!=0){
    
    if(cvFolds==nrow(augMatrix))
      cvFolds = cvFolds - length(rmList)
    
    augMatrix = augMatrix[-rmList,]
  }
  
  cat("removed : ",length(rmList)," subjects.\n")
  cat("num folds : ",cvFolds,"\n")
  
  feats = augMatrix[,1:numFeat]
  func = augMatrix[,(numFeat+1)]
  
  # Repeated Cross-validation
  cv_repeat = matrix(numeric(1),repeatRun,2)
  cv_permut = matrix(numeric(1),repeatRun,2)
  
  colnames(cv_repeat) = c('min','se1')
  colnames(cv_permut) = c('min','se1')
  
  for(testIdx in seq(1,repeatRun)){
    
    cv_glm = cv.glmnet(feats,func,nfolds=cvFolds,family="binomial",standardize=FALSE,weights=weights,alpha=alpha,type.measure="class",keep=TRUE,intercept=intc)
    
    cv_perm = cv.glmnet(feats[sample(nrow(feats)),],func,nfolds=cvFolds,family="binomial",standardize=FALSE,weights=weights,alpha=alpha,type.measure="class",keep=TRUE,intercept=intc)
    
    cv_repeat[testIdx,'min'] = cv_glm$cvm[match(cv_glm$lambda.min,cv_glm$lambda)]
    cv_repeat[testIdx,'se1'] = cv_glm$cvm[match(cv_glm$lambda.1se,cv_glm$lambda)]
    
    cv_permut[testIdx,'min'] = cv_perm$cvm[match(cv_perm$lambda.min,cv_perm$lambda)]
    cv_permut[testIdx,'se1'] = cv_perm$cvm[match(cv_perm$lambda.1se,cv_perm$lambda)]
    
  }
  
  # Final run for visualization
  
  set.seed(8357)
  folds_list = createFolds(func,cvFolds,list=FALSE)
  
  cv_glm = cv.glmnet(feats,func,nfolds=cvFolds,foldid=folds_list,family="binomial",standardize=FALSE,weights=weights,alpha=alpha,type.measure="class",keep=TRUE,intercept=intc)
  
  rm(.Random.seed,envir=globalenv())
  
  #save the prevalidated array for viz
  fit_preval_min = cv_glm$fit.preval[,match(cv_glm$lambda.min,cv_glm$lambda)]
  fit_preval_1se = cv_glm$fit.preval[,match(cv_glm$lambda.1se,cv_glm$lambda)]
  
  #get feature coefficients for the model
  cv_glm_coef_min = coef(cv_glm,s="lambda.min")
  cv_glm_coef_1se = coef(cv_glm,s="lambda.1se")
  
  mse_min = cv_glm$cvm[match(cv_glm$lambda.min,cv_glm$lambda)]
  mse_1se = cv_glm$cvm[match(cv_glm$lambda.1se,cv_glm$lambda)]
  
  #create barplot of selected predictors at Lambda.1se and Lambda.min
  coeff_temp_min = cv_glm_coef_min[2:length(cv_glm_coef_min),1]
  coeff_temp_min_idx = order(coeff_temp_min,decreasing=TRUE,na.last=NA)
  coeff_temp_min = coeff_temp_min[coeff_temp_min_idx]
  
  coeff_temp_1se = cv_glm_coef_1se[2:length(cv_glm_coef_1se),1]
  coeff_temp_1se_idx = order(coeff_temp_1se,decreasing=TRUE,na.last=NA)
  coeff_temp_1se = coeff_temp_1se[coeff_temp_1se_idx]
  
  min_model = list("lambda"=cv_glm$lambda.min,"mse"=mse_min,"preval"=fit_preval_min,"coeff"=coeff_temp_min,"order"=coeff_temp_min_idx)
  se1_model = list("lambda"=cv_glm$lambda.1se,"mse"=mse_1se,"preval"=fit_preval_min,"coeff"=coeff_temp_1se,"order"=coeff_temp_1se_idx)
  
  best_model = min_model
  
  return(list("removed"=rmList,"final_func"=func,"best_alpha"=alpha,"final_fit"=cv_glm,"best_model"=best_model,"min_model"=min_model,"se1_model"=se1_model,"repeat_mse_min"=mean(cv_repeat[,'min']),"repeat_mse_min_sd"=sd(cv_repeat[,'min']),"repeat_mse_se1"=mean(cv_repeat[,'se1']),"repeat_mse_se1_sd"=sd(cv_repeat[,'se1']),"cv_repeat"=cv_repeat,"permut_mse_min"=mean(cv_permut[,'min']),"permut_mse_min_sd"=sd(cv_permut[,'min']),"permut_mse_se1"=mean(cv_permut[,'se1']),"permut_mse_se1_sd"=sd(cv_permut[,'se1']),"cv_permut"=cv_permut,"coeff_min"=coeff_temp_min,"coeff_min_idx"=coeff_temp_min_idx,"coeff_se1"=coeff_temp_1se,"coeff_se1_idx"=coeff_temp_1se_idx))
  
}