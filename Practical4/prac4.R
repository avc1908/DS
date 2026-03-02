mistime2<-read.csv(file.choose(),header=T)
t.test(mistime2$time_g1,alternative="greater",mu=90)
