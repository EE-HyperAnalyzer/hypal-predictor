from pathlib import Path


from hypal_predictor.dbag import DbagReader
from hypal_predictor.model.builtin import LinearModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABAGS_PATH = PROJECT_ROOT / "databags" / "day_1.dbag"


TRAIN_SIZE = 0.8
INP_SEQ_LEN = 50
OUT_SEQ_LEN = 100
TIMEFRAMES = ["1:s", "5:s"]


def main():
    dbag = DbagReader(DATABAGS_PATH)
    model = LinearModel(input_horizon_length=INP_SEQ_LEN)

    for sensor in dbag.get_detectors_names():
        data = dbag.get_detector_data(sensor)
        _t = int(len(data) * TRAIN_SIZE)
        train_data = data[:_t]
        train_data_candle = [d.candle for d in train_data]
        model.fit(train_data_candle)

        valid_data = data[_t:]
        valid_candles = [d.candle for d in valid_data]

        _t = model.get_context_length() - 1
        buffer = valid_candles[:_t]
        predicted_candles = []
        for candle in valid_candles[_t:]:
            buffer.append(candle)
            prediction = model.predict(buffer)
            predicted_candles.append(prediction)
            buffer.pop(0)

        print(predicted_candles, valid_candles)


if __name__ == "__main__":
    main()
