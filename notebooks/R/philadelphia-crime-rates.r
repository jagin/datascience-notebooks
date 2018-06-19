
library(ggplot2)

CrimeData <- read.csv("../../data/CrimeData.csv")

str(CrimeData)

head(CrimeData)

head(CrimeData$Dispatch_Date_Time)

CrimeData$DateTime <- as.POSIXct(CrimeData$Dispatch_Date_Time, format="%Y-%m-%d %H:%M:%S", tz="EST")

?POSIXct

head(CrimeData$DateTime)

CrimeData$Date <- as.Date(CrimeData$DateTime, tz="EST")
str(CrimeData)

by_date <- aggregate(CrimeData$Date, by = list(Date = CrimeData$Date), FUN = length)

?aggregate

str(by_date)

colnames(by_date) <- c("Date", "Total")

str(by_date)

head(by_date)

ggplot(by_date, aes(Date, Total, color=Total)) + geom_line()

CrimeData$Hour <- strftime(CrimeData$DateTime, format = '%H', tz='EST')

str(CrimeData)

by_hour <- aggregate(CrimeData$Hour, by = list(Hour = CrimeData$Hour), FUN=length)

str(by_hour)

colnames(by_hour) <- c("Hour", "Total")

str(by_hour)

by_hour

by_hour$Hour <- as.integer(by_hour$Hour)

str(by_hour)

ggplot(by_hour, aes(Hour, Total)) +
  geom_line(colour="Red") +
  ggtitle("Crimes By Hour") +
  xlab("Hour of the Day") +
  ylab("Total Crimes")

CrimeData$Month <- strftime(CrimeData$DateTime, format = '%m', tz='EST')

str(CrimeData)

by_month <- aggregate(CrimeData$Month, by = list(Month = CrimeData$Month), FUN=length)

str(by_month)

colnames(by_month) <- c("Month", "Total")

str(by_month)

by_month$Month <- as.integer(by_month$Month)

str(by_month)

by_month

ggplot(by_month, aes(Month, Total)) +
  geom_bar(fill="Maroon", stat="identity")+
  ggtitle("Crimes By Month") +
  xlab("Month of the Day") +
  ylab("Total Crimes")

by_category <- aggregate(CrimeData$Text_General_Code,
                         by = list(Typec = CrimeData$Text_General_Code),
                         FUN = length)

# rename columns
colnames(by_category) <- c("Type", "Total")

by_category

by_category_sorted <- by_category[order(by_category$Total, decreasing=T),]

by_category_sorted

top10crimes <- by_category_sorted[1:10,]

top10crimes

ggplot(top10crimes, aes(x=reorder(Type,Total), y=Total)) +
  geom_bar(aes(fill=Type), stat="identity") +
  coord_flip()

by_hq <- aggregate(CrimeData$Dc_Dist, by = list(HQ = CrimeData$Dc_Dist), FUN=length)

#rename columns
colnames(by_hq) <- c("HQ", "Total")

ggplot(by_hq, aes(reorder(HQ, -Total), Total)) +
  geom_bar(color = "gray", stat="identity")
