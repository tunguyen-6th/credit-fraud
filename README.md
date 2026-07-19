# Credit Card Fraud Detection — Real-Time ML Pipeline

A real-time credit card fraud detection system built with **Apache Spark** (batch ML training + Structured Streaming) and **Apache Kafka**, with a WebSocket bridge for streaming scored transactions to live dashboards/clients.

The pipeline has three stages:

1. **Train** a Random Forest classifier on historical transaction data (`train_fraud_model.py`).
2. **Score** transactions in real time as they arrive on a Kafka topic, using the trained model (`spark_streaming_processor.py`).
3. **Broadcast** the scored transactions to WebSocket clients for live consumption (`kafka_websocket_bridge.py`).

## Architecture

```
                        ┌─────────────────────┐
  fraudTest.csv  ─────► │ train_fraud_model.py │ ─────► models/fraud_detection_rf_pipeline
                        └─────────────────────┘                    │
                                                                    ▼
 producer(s) ──► Kafka topic ──► spark_streaming_processor.py ──► Kafka topic ──► kafka_websocket_bridge.py ──► WebSocket clients
              "raw-transactions"   (loads saved Spark ML          "processed-           (bridges Kafka →
                                     pipeline, scores each         transactions"          WebSocket for a
                                     transaction, computes         "transaction-          live dashboard)
                                     risk tier + action,           metrics"
                                     writes windowed
                                     aggregate metrics)
```

## Components

### `train_fraud_model.py`
Trains and evaluates a fraud detection model using Spark MLlib.

- Loads `fraudTest.csv` (~555K labeled transactions, `is_fraud` target).
- Drops identifier/PII columns not useful as model features (`cc_num`, `merchant`, `first`, `last`, `street`, `city`, `trans_num`, `job`, `zip`, `unix_time`, `merch_lat`, `merch_long`).
- Builds a Spark ML `Pipeline`: `StringIndexer` → `OneHotEncoder` for categorical columns, `VectorAssembler` for numeric + encoded features, feeding a `RandomForestClassifier`.
- Handles class imbalance by computing a `classWeight` column (ratio of majority/minority class) and passing it to the classifier's `weightCol`.
- Tunes `numTrees` (30/50) and `maxDepth` (6/8) via `TrainValidationSplit` (80/20 train/validation), optimizing AUC-ROC.
- Evaluates the best model on a held-out 20% test split: AUC-ROC, AUC-PR, accuracy, F1, precision, recall, and a confusion matrix (plotted to PNG).
- Saves the full fitted pipeline (indexers + encoders + assembler + model) to `models/fraud_detection_rf_pipeline`, ready to be loaded by the streaming processor.

Run:
```bash
python train_fraud_model.py path/to/fraudTest.csv
```

### `spark_streaming_processor.py`
Consumes raw transactions from Kafka, scores them with the saved pipeline, and republishes the results.

- Reads JSON transaction events from the Kafka topic `raw-transactions` via Spark Structured Streaming.
- Loads the saved `PipelineModel` and applies it to each micro-batch.
- Derives `fraud_probability` (probability of class 1), `ml_prediction`, a `fraud_status` tier (`HIGH_RISK` / `MEDIUM_RISK` / `LOW_RISK`), and a `recommended_action` (`BLOCK` / `HOLD_FOR_REVIEW` / `FLAG` / `APPROVE`).
- Writes per-transaction results to the `processed-transactions` Kafka topic, and 1-minute windowed aggregate metrics (transaction counts, total/avg/max/min amount, avg fraud probability, predicted fraud count) to the `transaction-metrics` topic.
- Also streams a subset of columns to the console for local monitoring.

Run:
```bash
spark-submit spark_streaming_processor.py
```

### `kafka_websocket_bridge.py`
A lightweight bridge that forwards messages from the `processed-transactions` Kafka topic to connected WebSocket clients, so a browser-based dashboard can display fraud scores live.

Run:
```bash
python kafka_websocket_bridge.py
# Connect a WebSocket client to ws://localhost:8765
```

## Dataset

The model is trained on the [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (`fraudTest.csv`), which contains simulated card transactions labeled fraudulent (`is_fraud = 1`) or legitimate (`is_fraud = 0`), along with cardholder, merchant, and location metadata. The dataset is **not** included in this repository (large file with PII-shaped fields) — supply your own copy and pass its path to `train_fraud_model.py`.

## Evaluation results

The most recent training run's metrics, confusion matrix, and full training log are saved in [`evaluation_results/`](evaluation_results/).

## Requirements

- Python 3.9+
- Apache Spark 3.5.x (PySpark) with the `spark-sql-kafka-0-10` package for streaming
- Apache Kafka (local broker at `localhost:9092` by default)
- Java (JDK 17) for Spark
- Python packages: `pyspark`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `kafka-python`, `websockets`

## Known limitations

- Kafka and the WebSocket server run with no authentication or TLS — not suitable for handling real PII/payment data as-is.
- High-cardinality string columns (`trans_date_trans_time`, `dob`) are currently one-hot encoded as categorical features, which adds noise/dimensionality rather than useful signal.
- The WebSocket bridge assigns every client the same Kafka consumer group ID, so concurrent clients split partitions instead of each receiving the full stream.
- Risk thresholds in `fraud_status` (40%) and `recommended_action` (60%/30%) are independently chosen and not reconciled with the model's own 0.5 decision boundary.
