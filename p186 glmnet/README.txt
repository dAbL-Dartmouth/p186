# ******************************************************************************
# # Modified by : Joshua Weiner for Co-immunization of DNA and Protein in the Same Anatomical Sites Induces Superior Protective Immune Responses against SHIV Challenge
# Originally Produced by
# Author      : Srivamshi Pittala (srivamshi.pittala.gr@dartmouth.edu)
# Advisor     : Prof. Chris Bailey-Kellogg (cbk@cs.dartmouth.edu) & Prof. Margaret E. Ackerman
# Project     : Profectus T2
# Description : Help file for running the scripts to analyze luminex and functional measurements
#				associated with the Profectus HIV vaccine trial T2
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

#***********************
Input Files (in the directory in/)
1. luminex_tp5               || Biophysical measurements
2. subjects.csv		: group and survival information
#***********************

#***********************
For binomial logistic classification, use the following three scripts
#***********************

#-----------------------
1. predClassBinary_biophysical.R
Takes about 5 min to complete
#-----------------------
===> This script performs binomial logistic classification to identify vaccine groups using the biophysical measurements
		(1) Classification is done using the lasso-regularized binomial logistic regression on the features. The best regularization parameter (lambda) is chosen to be the one with lowest classification error. This is repeated multiple times. A final model is trained and evaluated by using a fixed seed to determine folds.
		(2) Permutation tests are performed by shuffling the rows of the class labels, but keeping the rows of feature matrix the same. The permuted data are sent through the same pipeline as were the actual data. This is repeated multiple times, independently permuting the class labels every time.
		(3) Robustness is estimated by comparing the accuracies from using actual features to those of using permuted features.

#***********************
For generating the figures as in the manuscript, graphpad prism was used to plot output data
#***********************


#-----------------------
System configuration 
#-----------------------
OS 	: Windows 10 Pro
CPU : Intel Xeon @ 3.2GHz
RAM : 8 GB

#-----------------------
Software and packages (version)
#-----------------------
01. R (3.4.3) and RStudio (1.1.423)
02. glmnet (2.0-16)
03. gplots (3.0.1)
04. ggplot2 (3.0.0)
05. effsize (0.7.1)
06. corrplot (0.84)
07. caret (6.0-80)	(Install with dependencies=TRUE)
08. polycor (0.7-9)
09. e1071(1.7-0)

#-----------------------
Functions used by the scripts (in the directory funcs/)
#-----------------------
01. convertKMtoEvents.R
02. createColumnColors.R
03. createSubjectColors.R
04. doFullSurvivalAnalysis.R
05. extractProbabilityFromKM.R
06. glmnetBiClass.R
07. heatmap4.R
08. plotConfusion.R
09. takeOffOneFeat.R

