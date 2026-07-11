#R language
library(numbers)
library(readr)
data<-read.table("pathtodataset.txt")
z=list() 
o<-NA
r<-NA
mydata<-list()
for(o in 1:length(data[,1])){
mydata[[o]]<-data[1:o,]
}

for(p in 1:length(mydata[[o]][,1])){
d1<-mydata[[p]][,1:5]
d<-combn(c(1:max(data)),2)
n<-length(mydata[[p]][,1])
dfg<-combn(mydata[[p]][n,1:5],2)

F<- function(x) {
sum(rowSums(cbind(as.matrix(apply(d1==d[1,x],1,sum)), as.matrix(apply(d1==d[2,x],1,sum))))==2)
}

F<-as.matrix(sapply(1:1225,F))
z[[p]]<-sapply(1:10,function(x) F[which((d[1,]==dfg[1,x]) & (d[2,]==dfg[2,x])),])
}

output <- matrix(unlist(z), ncol = 10, byrow = TRUE)
output
hist(F)


E<-rep(t(seq(1:14)), each=50)
E<-cbind(E,seq(1:50))
E
d2<-as.matrix(data[,6])
FF<-NA
for(i in 1:1225){
s<-as.matrix(apply(d1==E[i,2],1,sum))
ss1<-as.matrix(apply(d2==E[i,1],1,sum))
s3<-rowSums(cbind(s, ss1))
FF[i]<-sum(s3 == 2)
}
FF
hist(FF)
c<-combn(c(1:50), 5)
hist(F)
FF<-cbind(E,FF);FF
plot(d2)
hist(d2)


c<-combn(c(1:max(data)), 5)
d<-combn(c(1:max(data)),2)
C<-combn(c(1:max(data)), 5)
G<-NA
DD<-NA
for(i in 1:3107515){
D<-C[,i]
Y<-F[which(d[1,]==D[1] & d[2,]==D[2])]
Z<-F[which(d[1,]==D[1] & d[2,]==D[3])]
Q<-F[which(d[1,]==D[1] & d[2,]==D[4])]
W<-F[which(d[1,]==D[1] & d[2,]==D[5])]
#DD[i]<-F[which(d[1,]==D[1] & d[2,]==D[6])]
R<-F[which(d[1,]==D[2] & d[2,]==D[3])]
T<-F[which(d[1,]==D[2] & d[2,]==D[4])]
U<-F[which(d[1,]==D[2] & d[2,]==D[5])]
#DD[i]<-F[which(d[1,]==D[2] & d[2,]==D[2])]
O<-F[which(d[1,]==D[3] & d[2,]==D[4])]
P<-F[which(d[1,]==D[3] & d[2,]==D[5])]
#DD[i]<-F[which(d[1,]==D[3] & d[2,]==D[2])]
S<-F[which(d[1,]==D[4] & d[2,]==D[5])]
#DD[i]<-F[which(d[1,]==D[4] & d[2,]==D[2])]
#DD[i]<-F[which(d[1,]==D[5] & d[2,]==D[2])]
G[i]<-sum(Y+Z+Q+W+R+T+U+O+P+S)
}
G

poi<-apply(output,1,sum)
f<-length(poi)+1
g<-c(NA)
h<-c(NA)
for (i in 1:f){
u<-runif(i)
g[i]<-eulersPhi(i)
h<-moebius(i)
}
g
poi

g1<-g[1:length(g)-1]
poi
cor(g1,poi)

EM<-glm(poi~g1)
summary(EM)
predict(EM)
summary(output)

predicted<-predict(EM, data.frame(g[2:length(g)]), type='response')
summary(output)
###Averagings last 5
VALUE<-which(G==round(mean(poi[(length(poi)-5):length(poi)])))
GUESS<-t(c[,VALUE])		
####
#VALUE<-which(G==poi[length(poi)])
#GUESS<-t(c[,VALUE])

#Manhatan distance
#target_row <- data[1372, 1:5]  # Adjust the indexing if your data has more columns or rows
# Manhattan distance function
#manhattan_distance <- function(row1, row2) {
#  sum(abs(row1 - row2))
#}
#distances <- apply(GUESS, 1, function(row) manhattan_distance(row, target_row))
#closest_row_index <- which.min(distances)
#closest_row <- GUESS[closest_row_index, ]
#print(closest_row)
#print(min(distances))
#print(closest_row_index)
