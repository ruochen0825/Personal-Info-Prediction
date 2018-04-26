users <- read.csv("users.csv")
likes <- read.csv("likes.csv")
ul <- read.csv("users-likes.csv")


#Constructing a user-like matrix
ul$user_row <- match(ul$userid,users$userid)
ul$like_row <- match(ul$likeid,likes$likeid)


freq <- as.data.frame(table(ul$userid))
colnames(freq) <- c("userid", "freq")
users <- merge(users, freq, by="userid")

# construct sparse matrix
require(Matrix)
M <- sparseMatrix(i=ul$user_row, j=ul$like_row,x=1)
rownames(M) <- users$userid
colnames(M) <- likes$name
dim(M)

# Likes per person on average
nrow(ul)/dim(M)[1]
# Result = 95.8414, which is far less than the orginal avg(170)  => consider to trim the matrix

# Trim the matrix
repeat {
  i <- sum(dim(M))
  M <- M[rowSums(M) >= 18,]
  if (sum(dim(M)) == i) break
}
nrow(ul)/dim(M)[1]
# I tried rowsum from 15 - 20. I chose 18, with Likes per person on average = 170.6793. 
# Closest to original paper.

users <- users[match(rownames(M), users$userid),]


# There is not trimming that is explicitly wrote in the paper. So I will perform SVD without trimming
library(irlba)
library(ROCR)

# Cross-validation
folds <- sample(1:10, size = nrow(users), replace = T)

# logistic for gender
set.seed(seed = 68)
pred_g <- rep(NA, n = nrow(users))
pred_p <- rep(NA, n = nrow(users))
pred_o <- rep(NA, n = nrow(users))
pred_a <- rep(NA, n = nrow(users))
pred_c <- rep(NA, n = nrow(users))
pred_e <- rep(NA, n = nrow(users))
pred_agr <- rep(NA, n = nrow(users))
pred_n <- rep(NA, n = nrow(users))
for (i in 1:10){
  test <- folds == i
  Msvd <- irlba(M[!test,], nv = 100)
  v_rot <- unclass(varimax(Msvd$v)$loadings)
  u_rot <- as.data.frame(as.matrix(M %*%
                                     v_rot))
  # logistic for gender
  fit_g <- glm(users$gender~., data = u_rot,
               subset = !test)
  pred_g[test] <- predict(fit_g, u_rot[test,],type = "response")
  
  # logistic for political
  fit_p <- glm(users$political~., data = u_rot,
               subset = !test)
  pred_p[test] <- predict(fit_p, u_rot[test,],type = "response")
  
  # linear for age
  fit_a <- glm(users$age~., data = u_rot,
               subset = !test)
  pred_a[test] <- predict(fit_a, u_rot[test,])
  
  #ope
  fit_o <- glm(users$ope~., data = u_rot,
               subset = !test)
  pred_o[test] <- predict(fit_o, u_rot[test,])
  
  #con
  fit_c <- glm(users$con~., data = u_rot,
               subset = !test)
  pred_c[test] <- predict(fit_c, u_rot[test,])
  
  #ext
  fit_e <- glm(users$ext~., data = u_rot,
               subset = !test)
  pred_e[test] <- predict(fit_e, u_rot[test,])
  
  #agr
  fit_agr <- glm(users$agr~., data = u_rot,
                 subset = !test)
  pred_agr[test] <- predict(fit_agr, u_rot[test,])
  
  #neu
  fit_n <- glm(users$neu~., data = u_rot,
               subset = !test)
  pred_n[test] <- predict(fit_n, u_rot[test,])
}

# logistic for gender
temp_g <- prediction(pred_g,users$gender)
performance(temp_g,"auc")@y.values

political_df <- data.frame( "pred" = pred_p, "Actual" = users$political)
p_nomissing <- na.omit(political_df)
# logistic for political
temp_p <- prediction(p_nomissing$pred,p_nomissing$Actual)
performance(temp_p,"auc")@y.values

# age
cor(users$age, pred_a)
cor(users$ope, pred_o)
cor(users$con, pred_c)
cor(users$ext, pred_e)
cor(users$agr, pred_agr)
cor(users$neu, pred_n)

multiclass.roc(users$age,pred_a)
multiclass.roc(users$ope, pred_o)
multiclass.roc(users$con, pred_c)
multiclass.roc(users$ext, pred_e)
multiclass.roc(users$agr, pred_agr)
multiclass.roc(users$neu, pred_n)

# plot acc over number of likes

agedf <- data.frame(users$age,pred_a,users$freq)
agedf <- agedf[order(agedf$users.freq), ]
age_acc <- rep(NA, n = 300)
for (i in 1:300){
  curdf <- agedf[agedf$users.freq == i,]
  age_acc[i] = cor(curdf$users.age, curdf$pred_a)
}

opedf <- data.frame(users$ope, pred_o, users$freq)
opedf <- opedf[order(opedf$users.freq), ]
ope_acc <- rep(NA, n = 300)
for (i in 1:300){
  curdf <- opedf[opedf$users.freq == i,]
  ope_acc[i] = cor(curdf$users.ope, curdf$pred_o)
}

genderdf <- data.frame(users$gender, pred_g, users$freq)
genderdf <- genderdf[order(genderdf$users.freq), ]
gender_acc <- rep(NA, n = 300)
for (i in 1:300){
  curdf <- genderdf[genderdf$users.freq == i,]
  if (length(unique(curdf$users.gender)) != 2) {
    gender_acc[i] = 0
  } else {
    gender_acc[i] = performance(prediction(curdf$pred_g,curdf$users.gender),"auc")@y.values
  }
}
gender_acc <-unlist(gender_acc, use.names=FALSE)

picdf <- data.frame(age_acc, ope_acc,gender_acc)

