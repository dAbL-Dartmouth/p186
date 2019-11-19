# ******************************************************************************
# Modified by : Joshua Weiner for Co-immunization of DNA and Protein in the Same Anatomical Sites Induces Superior Protective Immune Responses against SHIV Challenge
# Originally Produced by
# Author      : Srivamshi Pittala
# Advisor     : Prof. Chris Bailey-Kellogg
# Project     : Profectus T2
# Description : Performs group clssification 

# ******************************************************************************

# Copyright (C) <2018>  <Srivamshi Pittala>

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

rm(list = ls())

source('funcsToImport.R')

dir_res = paste('results_predClassBinary_biophysical/',sep="")
dir.create(dir_res)


# -------------------------------------------
# Sec 01: Hyper-parameters
# -------------------------------------------

# glmnet parameters
alphas = 1  # elastic net parameter can vary between 0 (for Ridge) to 1 (for LASSO)
cvFolds = 10  # number of folds for cross-validation
repeatRun = 100 # number of repetitions of cross-validation
intc = TRUE # intercept to the linear model?
weights_bal = TRUE # balance classes?

log_file = paste(dir_res,'log_file',sep="")
file.create(log_file)
cat(rep('#',8),format(Sys.time(),"%d/%m/%Y %X"),rep('#',8),'\n\n',file=log_file,append=T)

cat('\nGlmnet Binomial Classification parameters','\n',sep="",file=log_file,append=T)
cat('Alpha range : ',paste(alphas,'',sep=','),'\n',file=log_file,append=T)
cat('Balancing classes',weights_bal,'\n',file=log_file,append=T)
cat('Intercept for logistic regression',intc,'\n',file=log_file,append=T)
cat('Number of folds : ',cvFolds,'\n',file=log_file,append=T)
cat('Number of repeated evaluations : ',repeatRun,'\n',file=log_file,append=T)

# -------------------------------------------
# Sec 02: Data
# -------------------------------------------

luminex = read.csv('data/luminex_tp5.csv', header=TRUE, row.names=1)

subjects = read.csv('data/subjects.csv', header=TRUE, row.names=1)

featNames = colnames(luminex)

# -------------------------------------------
# Sec 04: Glmnet classification
# -------------------------------------------

# Create labels for classification
classes = as.matrix((subjects[,'groupID']==2)*1)
feats = luminex

numFeat = ncol(feats)

# -------------
# Scale features and set NAs to 0
# -------------

feats = scale(feats)
na_idx = which(is.na(feats),arr.ind=TRUE)
if(length(na_idx)!=0){
  
  feats[na_idx] = 0
  
}

cat('\n\nClassification Results ','\n',file=log_file,append=T)
cat('\nPredictions for groups classes\n',file=log_file,append=T)

label = classes
weights = rep(1,length(label))
weights[which(label==0)] = sum(label==1)/sum(label==0)

class_model = glmnetBiClass(feats,label,weights,numFeat,intc,alphas,cvFolds,repeatRun)



# -------------
# log-odds
# -------------

pred_prob = class_model$final_fit$fit.preval[,match(class_model$final_fit$lambda.min,class_model$final_fit$lambda)]
pred_class = (pred_prob>0.5)*1
confMat = confusionMatrix(as.factor(pred_class),as.factor(label))
write.csv(confMat$table,"confmatrix.csv")

df = as.data.frame(cbind(label,pred_prob,subjects[,'group']))
colnames(df) = c('label','lp','Group')
df$label = as.factor(df$label)
df$Group = as.factor(df$Group)
pred_table = confMat$table
class_tot = colSums(pred_table)


write.csv(df,"box_pred.csv")
# -------------
# Calculate coefficients
# -------------

coeff_min = class_model$coeff_min
coeff_min_idx = class_model$coeff_min_idx
coeff_min_nz_idx = which(coeff_min!=0)


coeffs_sel = coeff_min_nz_idx[c(1,length(coeff_min))]
coeff_min_sel = class_model$coeff_min[coeffs_sel]

write.csv(coeff_min_sel,"top2coeffsel.csv")

feats_sel = luminex[,coeffs_sel]


cat('Rept : Mean of Classification Error',class_model$repeat_mse_min,'(',class_model$repeat_mse_min_sd,')\n')
cat('Perm : Mean of Classification Error',class_model$permut_mse_min,'(',class_model$permut_mse_min_sd,')\n')

robust_test = as.data.frame(cbind(1-class_model$cv_repeat[,'min'],1-class_model$cv_permut[,'min']))
colnames(robust_test) = c('Actual','Permuted')

df_test = as.data.frame(cbind(c(rep(1,repeatRun),rep(2,repeatRun)),as.vector(as.matrix(robust_test))))
colnames(df_test) = c('label','Acc')
df_test$label = as.factor(df_test$label)

diff_test = wilcox.test(robust_test[,1],robust_test[,2],alternative="two.sided")
eff_test = cliff.delta(robust_test[,1],robust_test[,2])
eff_interp = as.character(eff_test$magnitude)

write.csv(df_test,file=paste('robust.csv',sep=""),row.names = T)



write.csv(df_test, "robustness.csv")


coeffs_sel = coeff_min_idx[c(1,length(coeff_min))]
feats_sel = feats[,coeffs_sel]
write.csv(feats_sel, file="scaledfeatsfinalmodel.csv")
write.csv(coeff_min, file="coeff_minfinalmodel.csv")


df=scale(feats_sel)
df[is.na(df)]<- 0
pca = prcomp(df,center=FALSE, scale=FALSE)
N <- pca$x


df = data.frame(cbind(N[,1:2],subjects[,'groupID']))
colnames(df) = c('f1','f2','label')
df$label = as.factor(df$label)



write.csv(df,"PC.csv")

# print to log file
cat('Repeated Balanced Accuracy',class_model$repeat_mse_min,'(',class_model$repeat_mse_min_sd,')\n',file=log_file,append=T)
cat('Permuted Balanced Accuracy',class_model$permut_mse_min,'(',class_model$permut_mse_min_sd,')\n',file=log_file,append=T)

cat('\n\n',rep('#',8),format(Sys.time(),"%d/%m/%Y %X"),rep('#',8),'\n\n',file=log_file,append=T)
