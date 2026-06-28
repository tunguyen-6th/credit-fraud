# ============================================================================
# 2. UPDATED SPARK STREAMING PROCESSOR - spark_streaming_processor.py
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLStreamingProcessor:
    def __init__(self, model_path="models/fraud_detection_rf_pipeline"):
        self.spark = SparkSession.builder \
            .appName("ML-Fraud-Detection-Streaming") \
            .config("spark.jars.packages", 
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .config("spark.sql.streaming.schemaInference", "true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """
        Load pre-trained Random Forest pipeline model
        """
        try:
            logger.info(f"Loading pipeline model from {self.model_path}...")
            self.model = PipelineModel.load(self.model_path)
            logger.info("✅ Pipeline model loaded successfully!")
            logger.info(f"Pipeline stages: {len(self.model.stages)}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            logger.error("Please run train_fraud_model.py first to train the model")
            return False
    
    def define_schema(self):
        """
        Define schema for incoming transactions matching fraudTest.csv structure
        """
        return StructType([
            StructField("transaction_id", StringType(), False),
            StructField("trans_date_trans_time", StringType(), True),
            StructField("merchant", StringType(), True),
            StructField("category", StringType(), True),
            StructField("amt", DoubleType(), False),
            StructField("first", StringType(), True),
            StructField("last", StringType(), True),
            StructField("gender", StringType(), True),
            StructField("street", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
            StructField("zip", IntegerType(), True),
            StructField("lat", DoubleType(), True),
            StructField("long", DoubleType(), True),
            StructField("city_pop", IntegerType(), True),
            StructField("job", StringType(), True),
            StructField("dob", StringType(), True),
            StructField("trans_num", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("server_timestamp", StringType(), True)
        ])
    
    def read_from_kafka(self):
        """
        Read streaming data from Kafka
        """
        schema = self.define_schema()
        
        raw_stream = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "raw-transactions") \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", "1000") \
            .load()
        
        # Parse JSON from Kafka
        parsed_stream = raw_stream.select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp")
        ).select("data.*", "kafka_timestamp")
        
        return parsed_stream
    
    def preprocess_for_model(self, df):
        """
        Preprocess incoming data to match training data format
        Add class weight column (set to 1.0 for prediction)
        """
        # Add classWeight column (required by model but not used in prediction)
        df = df.withColumn("classWeight", lit(1.0))
        
        # Add processing timestamp
        df = df.withColumn("processing_timestamp", current_timestamp())
        
        return df
    
    def predict_fraud(self, df):
        """
        Use trained pipeline model to predict fraud
        """
        if self.model is None:
            logger.error("Model not loaded!")
            return df
        
        # Make predictions using the complete pipeline
        predictions_df = self.model.transform(df)
        
        # Extract fraud probability (probability of class 1)
        predictions_df = predictions_df.withColumn(
            "fraud_probability",
            col("probability").getItem(1) * 100
        )
        
        # Convert prediction to integer
        predictions_df = predictions_df.withColumn(
            "ml_prediction",
            col("prediction").cast("int")
        )
        
        # Determine fraud status based on prediction and probability
        predictions_df = predictions_df.withColumn(
            "fraud_status",
            when(col("ml_prediction") == 1, lit("HIGH_RISK"))
            .when(col("fraud_probability") > 40, lit("MEDIUM_RISK"))
            .otherwise(lit("LOW_RISK"))
        )
        
        # Use fraud_probability as fraud_score
        predictions_df = predictions_df.withColumn(
            "fraud_score",
            col("fraud_probability")
        )
        
        # Add recommended action
        predictions_df = predictions_df.withColumn(
            "recommended_action",
            when(col("ml_prediction") == 1, lit("BLOCK"))
            .when(col("fraud_probability") > 60, lit("HOLD_FOR_REVIEW"))
            .when(col("fraud_probability") > 30, lit("FLAG"))
            .otherwise(lit("APPROVE"))
        )
        
        # Add model metadata
        predictions_df = predictions_df.withColumn(
            "model_version", lit("RandomForest_fraudTest_v1.0")
        )
        
        predictions_df = predictions_df.withColumn(
            "prediction_timestamp", current_timestamp()
        )
        
        return predictions_df
    
    def select_output_columns(self, df):
        """
        Select relevant columns for output
        """
        return df.select(
            "transaction_id",
            "merchant",
            "category",
            "amt",
            "gender",
            "state",
            "lat",
            "long",
            "city_pop",
            "timestamp",
            "ml_prediction",
            "fraud_probability",
            "fraud_score",
            "fraud_status",
            "recommended_action",
            "model_version",
            "prediction_timestamp",
            "processing_timestamp"
        )
    
    def aggregate_metrics(self, df):
        """
        Calculate real-time aggregations
        """
        df_with_watermark = df.withWatermark("processing_timestamp", "5 minutes")
        
        windowed_metrics = df_with_watermark \
            .groupBy(
                window(col("processing_timestamp"), "1 minute"),
                col("fraud_status")
            ) \
            .agg(
                count("*").alias("transaction_count"),
                sum("amt").alias("total_amount"),
                avg("amt").alias("avg_amount"),
                max("amt").alias("max_amount"),
                min("amt").alias("min_amount"),
                avg("fraud_probability").alias("avg_fraud_probability"),
                sum(when(col("ml_prediction") == 1, 1).otherwise(0)).alias("predicted_fraud_count")
            ) \
            .select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("fraud_status"),
                col("transaction_count"),
                col("total_amount"),
                col("avg_amount"),
                col("max_amount"),
                col("min_amount"),
                col("avg_fraud_probability"),
                col("predicted_fraud_count")
            )
        
        return windowed_metrics
    
    def write_to_kafka(self, df, output_topic, checkpoint_location):
        """
        Write processed data back to Kafka
        """
        kafka_df = df.select(
            to_json(struct(*[col(c) for c in df.columns])).alias("value")
        )
        
        query = kafka_df \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("topic", output_topic) \
            .option("checkpointLocation", checkpoint_location) \
            .outputMode("append") \
            .start()
        
        return query
    
    def write_to_console(self, df, query_name, mode="append"):
        """
        Write to console for monitoring
        """
        query = df \
            .writeStream \
            .outputMode(mode) \
            .format("console") \
            .option("truncate", "false") \
            .option("numRows", "10") \
            .queryName(query_name) \
            .start()
        
        return query
    
    def run_pipeline(self):
        """
        Main ML streaming pipeline execution
        """
        try:
            logger.info("🚀 Starting ML-Powered Streaming Pipeline...")
            
            # Load pre-trained pipeline model
            if not self.load_model():
                logger.error("Cannot proceed without model. Exiting...")
                return
            
            # Read from Kafka
            raw_transactions = self.read_from_kafka()
            logger.info("✅ Connected to Kafka - reading from raw-transactions topic")
            
            # Preprocess for model
            preprocessed = self.preprocess_for_model(raw_transactions)
            logger.info("✅ Data preprocessed for model")
            
            # ML Predictions using pipeline
            predictions = self.predict_fraud(preprocessed)
            logger.info("✅ Random Forest pipeline predictions applied")
            
            # Select output columns
            output_transactions = self.select_output_columns(predictions)
            
            # Write individual processed transactions to Kafka
            transactions_query = self.write_to_kafka(
                output_transactions,
                "processed-transactions",
                "/tmp/checkpoint/ml-processed-transactions"
            )
            logger.info("✅ Writing ML predictions to processed-transactions topic")
            
            # Calculate and write aggregated metrics
            aggregated_metrics = self.aggregate_metrics(output_transactions)
            metrics_query = self.write_to_kafka(
                aggregated_metrics,
                "transaction-metrics",
                "/tmp/checkpoint/ml-transaction-metrics"
            )
            logger.info("✅ Writing aggregated metrics to transaction-metrics topic")
            
            # Console output for monitoring
            console_query = self.write_to_console(
                output_transactions.select(
                    "transaction_id", 
                    "merchant",
                    "amt",
                    "fraud_probability",
                    "fraud_status",
                    "ml_prediction",
                    "recommended_action"
                ),
                "ml-fraud-detection-console"
            )
            logger.info("✅ Console monitoring enabled")
            
            logger.info("\n" + "="*60)
            logger.info("🎯 ML Streaming Pipeline is running!")
            logger.info("="*60)
            logger.info("Model: Random Forest Pipeline (trained on fraudTest.csv)")
            logger.info("Input Topic: raw-transactions")
            logger.info("Output Topic: processed-transactions")
            logger.info("Metrics Topic: transaction-metrics")
            logger.info("="*60 + "\n")
            
            # Wait for termination
            self.spark.streams.awaitAnyTermination()
            
        except Exception as e:
            logger.error(f"❌ Error in ML streaming pipeline: {str(e)}")
            raise
        finally:
            logger.info("Shutting down streaming pipeline...")

if __name__ == "__main__":
    processor = MLStreamingProcessor()
    processor.run_pipeline()