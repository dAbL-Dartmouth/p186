plotConfusionBinary = function(conf_mat,labels,dir_class){
  
  actual = as.data.frame(table(labels))
  names(actual) = c("Reference","ReferenceFreq")
  
  df = data.frame(conf_mat$table)
  df = merge(df,actual,by=c('Reference'))
  df$Percent = df$Freq/df$ReferenceFreq
  
  pdf(paste(dir_class,'confusion.pdf',sep=''))
  p = ggplot(df,aes(x=Reference, y=Prediction,fill=Percent),color='black',size=4) + geom_tile(color="black",size=0.3) + labs(x="Reference",y="Prediction")
  #p = p + geom_text(aes(x=Reference, y=Prediction, label=sprintf("%.2f", Percent)),data=df, size=10, colour="black") + scale_fill_gradient(low="white",high="#458B00",limits=c(0,1))
  p = p + geom_text(aes(x=Reference, y=Prediction, label=sprintf("%.2f", Percent)),data=df, size=10, colour="black") + scale_fill_gradient(low="white",high="#458B00",limits=c(0,1))
  print(p)
  dev.off()
  
  
}

plotConfusionMulti = function(conf_mat,labels,dir_class){
  
  actual = as.data.frame(table(labels))
  names(actual) = c("Reference","ReferenceFreq")
  
  df = data.frame(conf_mat$table)
  df = merge(df,actual,by=c('Reference'))
  df$Rate = df$Freq/df$ReferenceFreq
  
  pdf(paste(dir_class,'confusion.pdf',sep=''))
  p = ggplot(df,aes(x=Reference, y=Prediction,fill=Rate),color='black',size=4) + geom_tile(color="black",size=0.5) + labs(x="\nReference",y="Prediction\n") + scale_x_discrete(labels=c('Empty','LTA-1','IL-12','IL-12+LTA1')) + scale_y_discrete(labels=c('Empty','LTA-1','IL-12','IL-12+LTA1')) + theme(axis.title.x = element_text(size=15,colour='black'), axis.title.y = element_text(size=15,colour='black'), axis.text.y=element_text(angle=45,size=16,colour = 'black'),axis.text.x=element_text(size=16,colour = 'black'),axis.ticks=element_blank(),panel.background = element_blank(), panel.grid.major.x = element_blank(), panel.grid.minor.y = element_blank(), panel.grid.major.y = element_blank(), legend.position='bottom')
  p = p + geom_text(aes(x=Reference, y=Prediction, label=sprintf("%.2f", Rate)),data=df, size=10, colour="black") + scale_fill_gradient(low="white",high="#458B00",limits=c(0,1))
  print(p)
  dev.off()
  
}