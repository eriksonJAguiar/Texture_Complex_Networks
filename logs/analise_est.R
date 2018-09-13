
file <- read.csv("Documents/Texture_Complex_Networks/logs/predicoes.csv", sep=";")

slbp = shapiro.test(c(file$lbp_rf,file$lbp_svm))
src = shapiro.test(c(file$lbp_rf,file$rc_svm))


p_slbp = slbp$p.value
p_src = src$p.value

shapiroframe <- data.frame('lbp' = p_slbp, 'rc' = p_src)

value <-c(file$lbp_rf,file$lbp_svm,file$lbp_rf,file$rc_svm)

n <- 6
k <- length(value)/6
len <- length(value)

z <- gl(n,k,len,labels = c("lbp","rc"))


w <- pairwise.wilcox.test(value, z, exact = FALSE)
p_values <- w$p.value