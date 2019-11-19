# ******************************************************************************
# Modified by : Joshua Weiner for Co-immunization of DNA and Protein in the Same Anatomical Sites Induces Superior Protective Immune Responses against SHIV Challenge
# Originally Produced by
# Author      : Srivamshi Pittala
# Advisor     : Prof. Chris Bailey-Kellogg
# Project     : Profectus T2
# Description : Imports the necessary packages, functions, and defines global variables
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

library(caret)
library(corrplot)
library(e1071)
library(effsize)
library(ggplot2)
library(glmnet)
library(gplots)
library(polycor)
library(survcomp)
library(survival)

# -----------------------------------------------------
source('funcs/convertKMtoEvents.R')
source('funcs/doFullSurvivalAnalysis.R')
source('funcs/extractProbabilityFromKM.R')
source('funcs/glmnetBiClass.R')
source('funcs/heatmap4.R')
source('funcs/plotConfusion.R')
source('funcs/takeOffOneFeat.R')

# -----------------------------------------------------

# group type
group_id = 1:2

