# ******************************************************************************
# Modified by : Joshua Weiner for Co-immunization of DNA and Protein in the Same Anatomical Sites Induces Superior Protective Immune Responses against SHIV Challenge
# Originally Produced by
# Author      : Srivamshi Pittala
# Advisor     : Prof. Chris Bailey-Kellogg
# Project     : Profectus T2
# Decription  : Performs survival analysis using the input features
# ******************************************************************************

doFullSurvivalAnalysis = function(feats,subjects,selected_feat_idx,num_folds,stop_limit,num_repeat,top_feat_thresh,seed_final,scolors,group_colors,km_colors,lcolors_original,set_plots,dir_surv){
  
  # -------------
  # Visualize selected features
  # -------------
  
  if(set_plots){
    
    ldata = scale(feats)
    ldata[ldata>3] = 3; ldata[ldata < -3] = -3
    lr = ceiling(max(abs(min(ldata,na.rm=TRUE)), max(ldata,na.rm=TRUE)))
    lbreaks = seq(-lr,lr,0.1)
    
    pdf(paste(dir_surv,'final-selection-raw.pdf',sep=""))
    heatmap.4(ldata, col=bluered, scale='none', trace='none', cexRow=0.5, cexCol=0.65, margin=c(10,8), breaks=lbreaks, symkey=FALSE, dendrogram='none',hclust=hclust.ward,RowSideColors=scolors,NumRowSideColors=4,ColSideColors=lcolors_original,NumColSideColors=2,Rowv=FALSE, Colv=FALSE, na.color='black',lmat=rbind(c(6,0,5),c(0,0,2),c(4,1,3)), lhei=c(2,0.4,6.0),lwid=c(0.3,0.1,0.3))
    dev.off()
    
    pdf(paste(dir_surv,'final-selection-by-challenge.pdf',sep=""))
    chall_sort = sort(subjects[,'Challenges'], index.return=TRUE)
    heatmap.4(ldata[chall_sort$ix,], col=bluered, scale='none', trace='none', cexRow=0.5, cexCol=0.6, margin=c(8,5), breaks=lbreaks, symkey=FALSE, dendrogram='none',hclust=hclust.ward,RowSideColors=scolors[chall_sort$ix,],NumRowSideColors=4,ColSideColors=lcolors_original,NumColSideColors=2,Rowv=FALSE, Colv=FALSE, na.color='black',lmat=rbind(c(6,0,5),c(0,0,2),c(4,1,3)), lhei=c(2,0.4,6.0),lwid=c(0.3,0.1,0.3))
    dev.off()
    
  }
  
  # -------------
  # Repeated cross-validation
  # -------------
  
  res_repeat = matrix(0,nrow=num_repeat,ncol=2)
  colnames(res_repeat) = c('cindex_tr','cindex')
  
  feat_counts = numeric(length(selected_feat_idx))
  
  for(repeatIdx in 1:num_repeat){
    
    dir_repeat = paste(dir_surv,"repeat_",repeatIdx,"/",sep="")
    
    if(set_plots){
      
      dir.create(dir_repeat)
      
    }
    
    
    cat(rep("#",30),"\n",sep="")
    cat('Doing repeat test : ',repeatIdx,'\n')
    
    model_rpt = coxSurvivalWithBackSearch(feats,subjects,selected_feat_idx,group_colors,num_folds,stop_limit,set_plots,dir_repeat)
    
    res_repeat[repeatIdx,'cindex_tr'] = model_rpt$cindex_train
    res_repeat[repeatIdx,'cindex'] = model_rpt$cindex_test
    
    feat_counts = feat_counts + colSums(model_rpt$feat_counts,na.rm=TRUE)
    
  }
  
  write.csv(res_repeat,file=paste(dir_surv,"res_repeat.csv",sep=""),row.names=FALSE,na="")
  
  # -------------
  # Train and test set performance in one plot
  # -------------
  
  train_test = cbind(res_repeat[,c('cindex_tr','cindex')])
  colnames(train_test) = c('train','test')
  write.csv(train_test,file=paste(dir_surv,'train_test.csv',sep=""),row.names = T)
  
  df_test = as.data.frame(cbind(c(rep(1,num_repeat),rep(2,num_repeat)),as.vector(as.matrix(train_test))))
  colnames(df_test) = c('label','C_index')
  df_test$label = as.factor(df_test$label)
  
  pdf(paste(dir_surv,'train_test.pdf',sep=""))
  p = ggplot(df_test,aes(x=label,y=C_index)) + geom_boxplot(width=0.6,notch = F,outlier.shape = NA, na.rm=T, size=1,colour="black",aes(fill=label)) + scale_x_discrete(labels=c('Train','Test')) + scale_y_continuous(limits = c(0.5,1), breaks=seq(0.5,1,0.1)) + theme(axis.line = element_line(colour = "black",size=1), axis.title.x = element_text(size=15,colour='black'), axis.title.y = element_text(size=20,colour='black') ,axis.text.x = element_text(size=20,colour='black'), axis.text.y = element_text(size=12,colour='black'),panel.background = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor.y = element_blank(), panel.grid.major.y = element_blank(), legend.position='bottom') + scale_fill_manual(values=c('darkorange3','deeppink4')) + xlab("") + ylab('Concordance Index\n')
  p = p + geom_hline(yintercept=0.5,colour='black',size=0.78,linetype='dashed',alpha=0.7)
  p = p + geom_hline(yintercept=median(train_test[,'train'],na.rm=T),colour='darkorange3',size=1.2,linetype='dashed',alpha=0.7)
  p = p + geom_hline(yintercept=median(train_test[,'test'],na.rm=T),colour='deeppink4',size=1.2,linetype='dashed',alpha=0.7)
  print(p)
  dev.off()
  
  # -------------
  # Select the most frequent features
  # -------------
  
  feat_percent = (feat_counts*100)/(num_folds*num_repeat)
  
  pdf(paste(dir_surv,'cox_feat_freq.pdf',sep=""))
  par(mar=c(12,4,2,0.5))
  barplot(feat_percent,main=c(" Mean frequencies of features "),ylab='Percent',las=2,names.arg=colnames(feats),cex.names=0.6)
  abline(h = top_feat_thresh,lwd=2,lty=2)
  dev.off()
  
  top_feat_idx = which(feat_percent>top_feat_thresh)
  
  if(length(top_feat_idx)<2){
    
    top_feat_order = order(feat_percent,decreasing = T)
    top_feat_idx = top_feat_order[c(1,2)]
    
  }
  
  # -------------
  # Build final model with most frequent features
  # -------------

  dir_final = paste(dir_surv,"final/",sep="")
  dir.create(dir_final)

  model_final = coxSurvivalFinalModel(feats[,top_feat_idx],subjects,seed_final,scolors,group_colors,km_colors,num_folds,plots=TRUE,dir_final)
  cindex_final = model_final$cindex_test

  feats_top = feats[,top_feat_idx]
  write.csv(feats_top,file=paste(dir_final,'feats.csv',sep=""),row.names = T,col.names = T)

  return(list('cindex_repeat'=res_repeat[,'cindex'],'cindex_final'=cindex_final,'top_feat_idx'=top_feat_idx))
  
}