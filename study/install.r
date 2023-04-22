install.packages('plyr', repos = "http://cran.us.r-project.org")

if (!require("devtools")){
    install.packages("devtools")
}
devtools::install_github("lihualei71/cfcausal")
