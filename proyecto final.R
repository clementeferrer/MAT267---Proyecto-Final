### Librerias ###

library(tidyverse)
library(fpp2)
library(padr)
library(lubridate)
library(Metrics)
library(astsa)
library(MLmetrics)
library(ggplot2)
library(Quandl)
library(SpatialPack)
library(nortest)
theme_set(theme_bw())

### R2 ###

R2 <- function(true, fitted){
  1 - (sum((true - fitted)^2)/sum((true - mean(true))^2))
}

### Limpieza de los datos ###

gold <- read.csv("C:/Users/ccfer/Downloads/gold_price_data.csv", colClasses = c("Date", NA))
gold <- gold[-c(1:37), ]
gold <- mutate(gold, Date = ymd(Date))
gold <- pad(gold, interval = "day") 
gold <- na.locf(gold)

### Convertir a serie de tiempo ###

gold_ts <- ts(data = gold$Value, start = 1979,frequency = 365)

autoplot(gold_ts)+
  geom_line(color = "blue", size = 0.1) +
  labs(x = "Date", y = "Price [USD]",
       title = "Gold Price")

### Cross Validation ###

train <- head(gold_ts, round(length(gold_ts) * 0.95))
h <- length(gold_ts) - length(train)
test <- tail(gold_ts, h)

### Modelo 1 - Holt Winter Multiplicativo ###

gold_decompose <- decompose(gold_ts, type = "multiplicative")
autoplot(gold_decompose)

gold_HWmodel <- HoltWinters(train, seasonal = "multiplicative")

gold_HWforecast <- forecast(train, h, level = 95)

autoplot(gold_ts) +
  autolayer(gold_HWmodel$fitted[,1], lwd = 0.5, 
            series = "HW Model", ) +
  autolayer(gold_HWforecast$mean, lwd = 0.5,
            series = "Forecast") +
  autolayer(gold_HWforecast$lower, lwd = 0.1,
            series = "Lower Bound CI 95%") +
  autolayer(gold_HWforecast$upper, lwd = 0.1,
            series = "Upper Bound CI 95%") +
  labs(x = "Date", y = "Price [USD]",
       title = "Gold Price Prediction using Seasonal HW")

rmse(test, gold_HWforecast$mean)
R2(test, gold_HWforecast$mean)

### Modelo 2 - ARIMA ###

par(mfrow = c(3,2))
acf(train, main = "Serie original")
pacf(train,main = "Serie original")
acf(diff(train), main = "Serie diferenciada")
pacf(diff(train), main = "Serie diferenciada")
acf(diff(diff(train)), main = "Serie doblemente diferenciada")
pacf(diff(diff(train)), main = "Serie doblemente diferenciada")

gold_arima <- Arima(train, order = c(3, 2, 1))
gold_ARIMA_NSforecast <- forecast(gold_arima, h,level=95)

autoplot(gold_ts) + 
  autolayer(gold_arima$fitted, lwd = 0.5, 
            series = "ARIMA Model") +
  autolayer(gold_ARIMA_NSforecast$mean, lwd = 0.5,
            series = "Forecast") +
  autolayer(gold_ARIMA_NSforecast$lower, lwd = 0.1,
            series = "Lower Bound CI 95%") +
  autolayer(gold_ARIMA_NSforecast$upper, lwd = 0.1,
            series = "Upper Bound CI 95%") +
  labs(x = "Date", y = "Price [USD]",
       title = "Gold Price Prediction using ARIMA(3,2,1)")

rmse(test, gold_ARIMA_NSforecast$mean)
R2(test, gold_ARIMA_NSforecast$mean)

### Modelo 3 - SARIMA ###

gold_arima_stl <- stlm(y = train, modelfunction=Arima, order=c(3,2,1))

gold_ARIMAforecast <- forecast(gold_arima_stl, h,level=95)

autoplot(gold_ts) + 
  autolayer(gold_arima_stl$fitted, lwd = 0.5, 
            series = "ARIMA Model") +
  autolayer(gold_ARIMAforecast$mean, lwd = 0.5,
            series = "Forecast") +
  autolayer(gold_ARIMAforecast$lower, lwd = 0.1,
            series = "Lower Bound CI 95%") +
  autolayer(gold_ARIMAforecast$upper, lwd = 0.1,
            series = "Upper Bound CI 95%") +
  labs(x = "Date", y = "Price [USD]",
       title = "Gold Price Prediction using seasonal ARIMA(3,2,1)")

rmse(test, gold_ARIMAforecast$mean)
R2(test, gold_ARIMAforecast$mean)

### Modelo 4 - ARIMA with Harmonic Regression ###

per <- spec.pgram(gold_ts, taper=0, log = "no", main = "Periodograma", col = "blue",
                  xlim = c(0, 1))
sort(per$spec)
s <- per$spec[per$spec>400000]
f <- per$freq[per$spec>400000]

points(f, s, cex = 1, pch = 19, col ="red")

harmonics <- fourier(train, K = 1)

fit <- Arima(train, order = c(3, 2, 1), xreg = harmonics)

newharmonics <- fourier(train, K = 1, h)
gold_harmonic <- forecast(fit, xreg = newharmonics, level = 95)

autoplot(gold_ts) + 
  autolayer(gold_harmonic$fitted, lwd = 0.5, 
            series = "Harmonic Regression Model") +
  autolayer(gold_harmonic$mean, lwd = 0.5,
            series = "Forecast") +
  autolayer(gold_harmonic$lower, lwd = 0.1,
            series = "Lower Bound CI 95%") +
  autolayer(gold_harmonic$upper, lwd = 0.1,
            series = "Upper Bound CI 95%") +
  labs(x = "Date", y = "Price [USD]",
       title = "Gold Price Prediction using Harmonic Regression with ARIMA(3,2,1) errors)")

rmse(test, gold_harmonic$mean)
R2(test, gold_harmonic$mean)

### Chequear supuestos ###

checkresiduals(fit)
checkresiduals(gold_HWmodel)
checkresiduals(gold_arima_stl)
checkresiduals(gold_arima)

ad.test(fit$residuals)

### Predicción ###

harmonics <- fourier(gold_ts, K = 1)

fit <- Arima(gold_ts, order = c(3, 2, 1), xreg = harmonics)

newharmonics <- fourier(gold_ts, K = 1, h)
gold_harmonic <- forecast(fit, xreg = newharmonics, level = 95)

autoplot(gold_ts) + 
  autolayer(gold_harmonic$fitted, lwd = 0.5, 
            series = "Harmonic Regression Model") +
  autolayer(gold_harmonic$mean, lwd = 0.5,
            series = "Forecast") +
  autolayer(gold_harmonic$lower, lwd = 0.1,
            series = "Lower Bound CI 95%") +
  autolayer(gold_harmonic$upper, lwd = 0.1,
            series = "Upper Bound CI 95%") +
  labs(x = "Date", y = "Price [USD]",
       title = "Gold Price Prediction using Harmonic Regression with ARIMA(3,2,1) errors)")


### Silver Comovement ###

silver = Quandl("LBMA/SILVER", api_key="h_TbsUf7pyA9Dj1MpELq")
silver <- silver[nrow(silver):1,]

silver <- silver[-c(1:2791), ]
head(silver)
silver <- silver[-c(10415:11000),]
silver[c(3,4)]<-list(NULL)
silver <- mutate(silver, Date = ymd(Date))
silver <- pad(silver, interval = "day") 
silver <- na.locf(silver)
silver <- silver[-c(1:3), ]
silver <- silver[-c(15049:15912),]

x <- silver$USD
y <- gold$Value

print(cor(x, y, method = "pearson"))

silver_ts <- ts(data = x, start = 1979,frequency = 365)
autoplot(silver_ts)+
  geom_line(color = "blue", size = 0.1) +
  labs(x = "Date", y = "Price [USD]",
       title = "Silver Price")
