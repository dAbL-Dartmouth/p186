

createColumnColors = function(featNames,reagent_names,reagent_colors,antigen_names,antigen_colors){
  
  # create column colors
  lcolors = matrix(nrow=length(featNames), ncol=2, dimnames=list(featNames, c('reagent','antigen')))
  
  for(reagent_name in reagent_names){
    
    lcolors[grep(reagent_name,featNames),'reagent'] = reagent_colors[reagent_name]
    
  }
  
  for(antigen_name in antigen_names){
    
    lcolors[grep(antigen_name,featNames),'antigen'] = antigen_colors[antigen_name]
    
  }
  
  return(lcolors)
  
}