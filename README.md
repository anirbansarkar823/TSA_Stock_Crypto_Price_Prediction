#### This repo is about method and techniques to use the concepts of time-series for solving various use-cases like Stock Price prediction, etc.

#### Various terminologies we came across, when dealing with time-series data:
    1. **Seasonality**:  refers to the upward and downward movements that we see in (say) sales throughout a year (or over a period of time), a pattern which repeats itself multiple times throughout that period, for e.g., during festive seasons the sales are more (upward) for a product, but we may see a poor sales (downward) for the same product in winter season

    2. **Decomposition**: The process of finding the additive and multiplicative parts of a time series is called decomposition.
        * Additive time series is a combination of **Trend + Seasonality + Irregularities**. On the other hand, Multiplicative time series is a combination of **Trend * Seasonality * Irregularities**.

    3. **Smothening of time series data**:
        * For smoothing a time series, we can use the moving average (rolling mean) technique. Here, we set a lag value n (say 3), and thus predict every [n+1]th (say 4th) value using the average of previous n values (here, 1st, 2nd and 3rd values).

        *  But if want to give different weitages to different values (say more weitage to recent observation, and less weitage to remote past observation) then we go for **exponential smoothing**.
            * Double exponential smoothing, can capture the trend component (if it is present in the time series) and using Triple exponential smoothing, we can capture both trend and seasonality component.

    4. **Stationarity**: If a time series have constant mean and variance over the time, we call it a stationary time series. For time series forcasting, we need stationary data. If the data is not stationary, we use transformations (log, sqrt, etc.) and differentiation (lag-1, lag-3, lag-n, etc.) to make the time series stationary. After making the time series stationary, we can use various statistical models (AR or MA model) for forecasting.

    5. **Adfuller test**: ADF test is a statistical test to tell whether the time series is stationary or not. 
        Here, H0 (null hypothesis) is that the time series is non-stationary, 
        H1 (alternate hypothesis) is that the time series is stationary. 
        So if p-value is less then significance level (say 0.05), we can reject the null hypothesis and say the series is stationary.

    6. Lags: the observations at yt and y(t–k) are separated by k time units. K is called the lag.

    7. **PACF vs ACF**:
            * ACF are plots between time series and lagged version of itself in-order to find the correlation. Used to find the q value in ARIMA (MA component). ACF considers both direct and indirect correlations between lagged values (say corr(yt, yt-3) and corr(yt, yt-3 && yt, yt-3 via yt-2))

            * In PACF & ACF graphs, wherever the graph (bars) crosses the upper/lower thershold, it is called the optimum value. Bars below the thresholds are statistically insignificant.

            * PACF are plots between time series and lagged version of itself but 'after eleminating the variance'. Used to find the p value in ARIMA (AR component). PACF only considers the direct correlations between lagged values.
    
    8. Various Time Series Models:
        * In auto-regressive models (AR model), the current values depends on outcomes of previous days, i.e., predicting current time series values based on previous time series values. So there is a linear relationship, thus a linear regression model can be built. In AR models, the lag values is decided using PACF graph.
            > AR models are suitable for time-series data, when do not have trend and seasonal component.

        * Moving averages (MA models) depends on previous errors (unlike outcomes, as mentioned in AR models) and current error term i.e., considering the errors of previous data to predict the current value. As error terms are random, no linear models can be built. In MA models, we use ACF to predict lag value.
        
        * ARMA: when time series is stationary, i.e., constant mean and variance over time

        * ARIMA (AR + Integrated + MA) models are used for forecasting when the time series data has some linear trends (upwards or downwards), but no seasonality. We first remove the linear trend using lag-shift differtiation {z(t) = y(t) - y(t-d); where, d is the order of integrated part, i.e., the differencing to get rid of trend} to make the time series stationary.
            if the time series is non-stationary, i.e., it has trend component over time.
            Then we use ARIMA model to make the predictions. 
            Finally we get back the real predictions by simply adding the predictions to d-lag values {y(t) = z(t) + y(t-d)}.

        * SARIMA (Seasonality + AR + Integrated (trend) + MA) models are used for forecasting when the time series data has some correlations with previous data (AR or MA component), some linear trends and seasonality as well.
            if the time series is non-stationary and has both trend and seasonality component



#### Drawbacks of ARIMA vs FBProphet:

    * ARIMA is bad for multistep forcasting, if we have to predict daily prices for some number of days, using current data. So, ARIMA works best, when data is hourly, and we wanted to predict the hourly prices

    * FBProphet, even if the data is hourly (timeframe) we can predict data for next month, next day, next year. So FBProphet has an edge over ARIMA here.

    * FBProphet can handle holidays like diwali, holi as we can find such holidays where chances of prices being high is more

#### FBProphet requirements:

    * the independent variable must be named as 'ds' and the dependednt variable must be named as 'y'

    * No need to transform the data into stationary series, FBProphet handles it

    * We can identify what kind of seasonality it is? yearly, weekly, etc. 
        But in ARIMA this is not possible. We can only tell if there is any seasonality present or not, but not the kind

    * model.make_future_dataframe(periods=10, freq='2D') # daily frequency 2 days
    model.make_future_dataframe(periods=10, freq='M') # Monthly frequency
	

#### LSTM:
	* LSTM can be used on supervised dataset {0 to n-1 as traning and nth point as testing}
	* The feature needs to be scaled (normalized) before passing to LSTM for learning
	* The dataset will have index as datetime and values as target variable