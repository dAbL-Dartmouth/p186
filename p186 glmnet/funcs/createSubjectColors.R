createSubjectColors = function(subjects,group_colors,km_colors,challenge_colors){
  
  # create row colors
  scolors = matrix(nrow=nrow(subjects), ncol=3, dimnames=list(rownames(subjects), c('group_all','groupID','Challenges')))
  
  for (i in 1:nrow(scolors)) {
    scolors[i,'groupID'] = group_colors[subjects[i,'groupID']]
    scolors[i,'groupID'] = km_colors[subjects[i,'groupID']]
    scolors[i,'Challenges'] = challenge_colors[subjects[i,'Challenges']]
    if (subjects[i,'censor']==0) scolors[i,'Challenges'] = 'green'   #is not affected by challenges
  }
  
  return(scolors)
  
}

createSubjectColors_old = function(subjects,group_colors,km_colors,challenge_colors){
  
  # create row colors
  scolors = matrix(nrow=nrow(subjects), ncol=3, dimnames=list(rownames(subjects), c('groupID','groupID','Challenges')))
  
  for (i in 1:nrow(scolors)) {
    scolors[i,'groupID'] = group_colors[subjects[i,'groupID']]
    scolors[i,'groupID'] = km_colors[subjects[i,'groupID']]
    scolors[i,'Challenges'] = challenge_colors[subjects[i,'Challenges']]
    if (subjects[i,'censor']==0) scolors[i,'Challenges'] = 'green'   #is not affected by challenges
  }
  
  return(scolors)
  
}