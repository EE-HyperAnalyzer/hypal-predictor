This repo contains the API for Hypal Predictor.

From CoreAPI we receive the candle-like data (OHCL format) from sensors with timeframe of 1s. If this API sees a new sensor, we should add model (LinearModel for tests) and waiting of configuration. User from telegram bot sends the configuration contains timeframe and candles_to_train.

The next steps is like state machine:
1. Model has state `WAITING`
2. After receiving new timeframe, model has state `GATHERING`
3. After receiving all candles_to_train, model has state `TRAINING`
4. After training, model has state `READY`

Also, 1s candles should be aggregated to corresponding timeframe.
(If model has timeframe = 2:s and candles_to_train=100 then we should wait for 200 candles that aggregated to 100 2:s candles)

Service should be able to unload models from memory if they are not used for a long time and loaded from disk if needed.

For `TRAINING` state:
- There should be a training queue
- Max parrallel training is a parameter

For `READY` state:
- There should be a max loaded models

Anomaly = MSE(REAL_HORIZONT, PREDICTED_HORIZONT) > anomaly_threshold

Model can be in critical state if it has anomaly and continue being in this state until MSE(REAL_HORIZONT, PREDICTED_HORIZONT) <= anomaly_threshold.
There are notifications like:
- Anomaly detected!
- Anomaly exited!

For every detector exists a set of models on every timeframe that predicts the next candle-like data (OHCL format) and handles anomalies.
