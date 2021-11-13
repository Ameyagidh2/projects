print("Hello world")
x<-1:5
y<-c(6,7,8)
x+2
x<-x*2
x
print(y)
y<-y<-c(6,7,8,9,10)
x+y
z<-x+y
ls()
sn.csv<-read.csv("A:\\python\\R programming\\02_04\\social_network1.csv",header=T)
str(sn.csv)
sn.spss.csv<-read.csv('A:\\python\\R programming\\02_04\\social_networksavspss_to_csv_converted.csv',header=T)
str(sn.spss.csv)

install.packages('foreign')
library(foreign)
sn.spss.fw<-read.spss("A:\\python\\R programming\\02_04\\social_network.sav",to.data.frame=T,use.value.labels=T)
str(sn.spss.fw)
browseURL('https://play.google.com/store/apps/details?id=com.nianticlabs.pokemongo&hl=en_IN&gl=US')
library()
search()
install.packages('psych')
library(psych)
install.packages('ggplot2')
library(ggplot2)
vignette()
browseVignettes()
sn2.csv<-read.csv("social_network1.csv",header=T)
site.freq<-table(sn2.csv$Site)
barplot(site.freq)
? barplot
barplot(site.freq[order(site.freq,decreasing=T)])
fbba <- c(rep("gray", 4),
        rgb(59, 89, 152, maxColorValue=255),
        rgb(59, 89, 52, maxColorValue=255))
barplot(site.freq[order(site.freq)],horiz=T,col=fbba,xlim=c(0,100),border = NA,xlab = 'Social media sites names',main='Preferences of people \n study of 202 people')
# Histogram plot
histogramplot.a<-table(sn2.csv$Age)
#hist(sn2.csv$Age)
hist(histogramplot.a,col='beige',xlab='Age of respondants',main='Data of age of respondants|n A survey of 202 people')
colors()[18]

sn3.ameya<-read.csv("social_network1.csv",header=T)
sn3.a<-(sn3.ameya$Site)
barplot(sn3.a[order(sn3.a)],horiz=T)

table(sn3.ameya)
table(sn3.ameya$Site)
site.freq<-table(sn3.ameya$Site)
site.freq<-site.freq[order(site.freq,decreasing=T)]
site.freq
prop.table(site.freq)
a<-round(prop.table(site.freq),2)
a
summary(sn3.ameya$Age)
describe(sn3.ameya)

#describing and recoding variables
describe(sn3.ameya)
hist(sn3.ameya$Times)

#Reordering variable 
#scaling  changes x axis values
describe(sn3.ameya)
describe(sn3.ameya$Times)
hist(sn3.ameya$Times)
scaleofTime<-scale(sn3.ameya$Times)
scaleofTime
hist(scaleofTime)
sn3.ameya

#logarithm when values are changing add +1 to take zero frequency values also into account

logvalues_4<-log(sn3.ameya$Times)
hist(logvalues_4)
describe(logvalues_4)


logvalues_5<-log(sn3.ameya$Times+1)
hist(logvalues_5)
describe(logvalues_5)



#Ranking tier method at tie give random values
rank1<-rank(sn3.ameya$Times,ties.method = "random")
describe(rank1)
hist(rank1)

#dichtoming for outliers
dic<-ifelse(sn3.ameya$Times>1,1,0)
dic

n1<-rnorm(1000000)
n2<-rnorm(1000)
n3<-n1*n2
hist(n3)
describe(n3)
n4=n1+n2
describe(n4)
hist(n4)

#Project google data


google.coll<-read.csv("google_correlate.csv",header=T)
google.coll
names(google.coll)
str(google.coll)
#data split  splitting data according to region
data.v<-split(google.coll$data_viz,google.coll$region)
data.v  
boxplot(data.v,col="lavender")  
abline(h=0)
p1<-sapply(data.v,mean)
p1  
barplot(p1)
describeBy(google.coll$data_viz,google.coll$region)
split(google.coll$data_viz,google.coll$region)
#scatter plot
plot(google.coll$degree,google.coll$data_viz,main='Plot of college degree and interest in data visualization\n202 "people data]n Plotted using R"' )
lines(lowess(google.coll$degree,google.coll$data_viz),col='red')
pairs(~data_viz+degree+facebook+nba,data=google.coll,pch=20,main="Plots for data useage from various places all over usa")
pairs.panels(google.coll[c(3,7,4,5)],gap=0)
plot3d(google.coll$data_viz,google.coll$degree,google.coll$facebook,xlab="data_viz",ylab='degree',zlab="Facebook")


# Correlation
install.packages("ggplot2", dependencies = TRUE)
libary('ggplot2')
google.corr<-read.csv("google_correlate.csv",header=T)
names(google.corr)
g2<-google.corr[c(3,7,4,4)]
cor(g2)
install.packages("Hmisc")
library('Hmisc')
cor.test(google.corr$data_viz,google.corr$degree)
rcorr(as.matrix(g2))

#Regression how output data_viz related to other variables
reg<-lm(data_viz~degree+facebook+nba,has_nba,region,data=google.corr)
summary(reg)  
#crosstables
sn<-read.csv("social_network1.csv",header=T)
s1<-table(sn$Gender,sn$Site)
s1
#checking according to row and then according to columns 
margin.table(s1,1)
margin.table(s1,2)
#percentage according to row and column
round(prop.table(s1),2)
round(prop.table(s1,1),2)
#inferential test to check if really a relation exists
chisq.test(s1)
#p value<0.05 then statistically significant
t.test(google.corr$nba~google.corr$has_nba)
#pvalue <0.05 so use less ie not related to each other
#2 variables as an input for a corresponding output
anoval<-aov(data_viz~ region+stats_ed,data=google.corr)
summary(anoval)


anoval1<-aov(data_viz~ region+stats_ed+region:stats_ed,data=google.corr)
summary(anoval1)

anoval3<-aov(region+stats_ed,data=google.corr)
summary(anoval1)
#randomnormal value generator
random=rnorm(1000)
#random
qqnorm(random)
#Iris data set project
iris2=read.csv("iris.csv",header=T)
summary(iris2)
colnames(iris2)
#editing data sets names
names(iris2)<-c('sepal_length','sepal_width','petal length','petal width','Species')
names(iris2)
colnames(iris2)
rownames(iris2)
s1<-iris2$sepal_length
s1
dim(iris2)
mean(s1)
max(s1)
which(s1==5.1)
m1<-s1[which.max(s1)]
m1
