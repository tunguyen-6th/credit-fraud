# ============================================================================
# OPTIMIZED MODEL TRAINING SCRIPT - train_fraud_model.py
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudModelTrainer:
    def __init__(self, data_path="fraudTest.csv"):
        self.data_path = data_path
        self.spark = SparkSession.builder \
            .appName("Fraud-Detection-Model-Training") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.default.parallelism", "10") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Create models directory if it doesn't exist
        self.models_dir = "models"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            logger.info(f"✅ Created directory: {self.models_dir}")
        
    def load_data(self):
        """
        Load fraud detection dataset from CSV
        """
        logger.info(f"Loading data from {self.data_path}...")
        start_time = time.time()
        
        df = self.spark.read.csv(self.data_path, header=True, inferSchema=True)
        
        # Cache the dataframe for faster processing
        df.cache()
        
        count = df.count()
        load_time = time.time() - start_time
        
        logger.info(f"✅ Data loaded in {load_time:.2f}s! Total records: {count:,}")
        
        # Check class distribution
        logger.info("\nClass distribution:")
        class_distribution = df.groupBy('is_fraud').count().orderBy('is_fraud')
        class_distribution.show()
        
        return df
    
    def check_missing_values(self, df):
        """
        Quick check for missing values
        """
        logger.info("\nChecking for missing values...")
        
        # Quick check - only count if there are any nulls
        null_cols = []
        for column_name in df.columns:
            null_count = df.filter(col(column_name).isNull()).count()
            if null_count > 0:
                null_cols.append(column_name)
                logger.info(f"Column '{column_name}': {null_count} missing values")
        
        if not null_cols:
            logger.info("✅ No missing values found")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data: drop unnecessary columns and select features
        """
        logger.info("\nPreprocessing data...")
        
        # Drop unnecessary columns
        columns_to_drop = ['merch_lat', 'merch_long', 'zip', '_c0', 'unix_time', 
                          'cc_num', 'merchant', 'first', 'last', 'street', 
                          'city', 'trans_num', 'job']
        
        df_reduced = df.drop(*columns_to_drop)
        
        logger.info(f"Dropped {len(columns_to_drop)} columns")
        
        return df_reduced
    
    def identify_features(self, df):
        """
        Identify categorical and numerical features
        """
        # Identify categorical columns (string type)
        categorical_cols = [col_name for col_name, col_type in df.dtypes if col_type == 'string']
        
        # Identify numerical columns
        numerical_cols = ['amt', 'lat', 'long', 'city_pop']
        
        logger.info(f"\nCategorical features: {categorical_cols}")
        logger.info(f"Numerical features: {numerical_cols}")
        
        return categorical_cols, numerical_cols
    
    def build_pipeline(self, categorical_features, numerical_features):
        """
        Build ML pipeline with feature engineering and Random Forest classifier
        """
        logger.info("\nBuilding ML pipeline...")
        
        # String Indexers for categorical features
        indexers = [
            StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") 
            for col in categorical_features
        ]
        
        # One-Hot Encoders for categorical features
        encoders = [
            OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") 
            for col in categorical_features
        ]
        
        # Combine feature columns
        encoded_categorical_features = [col + "_vec" for col in categorical_features]
        all_feature_cols = numerical_features + encoded_categorical_features
        
        logger.info(f"Total feature columns: {len(all_feature_cols)}")
        
        # Vector Assembler
        assembler = VectorAssembler(
            inputCols=all_feature_cols, 
            outputCol='features',
            handleInvalid="skip"
        )
        
        # Random Forest Classifier (OPTIMIZED FOR SPEED)
        rf = RandomForestClassifier(
            featuresCol='features',
            labelCol='is_fraud',
            weightCol='classWeight',
            numTrees=50,      # Reduced from 100 for faster training
            maxDepth=8,       # Reduced from 10
            maxBins=32,
            seed=42,
            subsamplingRate=0.8  # Use 80% of data per tree for speed
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])
        
        logger.info("✅ ML Pipeline created successfully!")
        
        return pipeline, rf
    
    def add_class_weights(self, df):
        """
        Add class weights to handle imbalanced dataset
        """
        logger.info("\nCalculating class weights...")
        
        num_majority = df.filter(df['is_fraud'] == 0).count()
        num_minority = df.filter(df['is_fraud'] == 1).count()
        
        logger.info(f"Majority class (is_fraud = 0): {num_majority:,}")
        logger.info(f"Minority class (is_fraud = 1): {num_minority:,}")
        
        class_weight_value = num_majority / num_minority
        
        df_weighted = df.withColumn(
            "classWeight",
            when(df['is_fraud'] == 1, class_weight_value).otherwise(1.0)
        )
        
        logger.info(f"✅ Class weight for minority class: {class_weight_value:.2f}")
        
        return df_weighted
    
    def train_model_fast(self, pipeline, rf, train_data):
        """
        Train model with FAST hyperparameter tuning using TrainValidationSplit
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 TRAINING RANDOM FOREST MODEL (FAST MODE)")
        logger.info("="*70)
        
        start_time = time.time()
        
        # SMALLER parameter grid for faster training
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [30, 50]) \
            .addGrid(rf.maxDepth, [6, 8]) \
            .build()
        
        logger.info(f"Parameter grid size: {len(paramGrid)} combinations")
        
        # Create evaluator (optimize for AUC-ROC)
        evaluator = BinaryClassificationEvaluator(
            labelCol='is_fraud',
            rawPredictionCol='rawPrediction',
            metricName='areaUnderROC'
        )
        
        # Use TrainValidationSplit instead of CrossValidator for MUCH faster training
        tvs = TrainValidationSplit(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            trainRatio=0.8,  # 80% for training, 20% for validation (faster than CV)
            parallelism=4    # Process 4 models in parallel
        )
        
        logger.info("Training with TrainValidationSplit (faster than CrossValidator)...")
        logger.info("This should take 2-5 minutes...")
        
        # Train model
        tvs_model = tvs.fit(train_data)
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Get best model
        best_model = tvs_model.bestModel
        
        # Extract Random Forest model from pipeline
        rf_model = best_model.stages[-1]
        
        logger.info(f"\nBest model parameters:")
        logger.info(f"Number of trees: {rf_model.getNumTrees}")
        logger.info(f"Max depth: {rf_model.getMaxDepth()}")
        
        return tvs_model, best_model
    
    def evaluate_model(self, model, test_data):
        """
        Evaluate model performance on test data
        """
        logger.info("\n" + "="*70)
        logger.info("📊 EVALUATING MODEL PERFORMANCE")
        logger.info("="*70)
        
        eval_start = time.time()
        
        # Make predictions
        predictions = model.transform(test_data)
        
        # Cache predictions for faster multiple evaluations
        predictions.cache()
        _ = predictions.count()  # Force evaluation
        
        eval_time = time.time() - eval_start
        logger.info(f"Evaluation time: {eval_time:.2f} seconds")
        
        # Show sample predictions
        logger.info("\nSample predictions:")
        predictions.select('is_fraud', 'prediction', 'probability').show(10, truncate=False)
        
        # Binary Classification Metrics
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol='is_fraud',
            rawPredictionCol='rawPrediction'
        )
        
        auc_roc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
        auc_pr = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})
        
        logger.info(f"\n🎯 AUC-ROC: {auc_roc:.4f}")
        logger.info(f"🎯 AUC-PR: {auc_pr:.4f}")
        
        # Multiclass Classification Metrics
        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol='is_fraud', predictionCol='prediction', metricName='accuracy'
        )
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol='is_fraud', predictionCol='prediction', metricName='f1'
        )
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol='is_fraud', predictionCol='prediction', metricName='weightedPrecision'
        )
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol='is_fraud', predictionCol='prediction', metricName='weightedRecall'
        )
        
        accuracy = accuracy_evaluator.evaluate(predictions)
        f1_score = f1_evaluator.evaluate(predictions)
        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        
        logger.info(f"🎯 Accuracy: {accuracy:.4f}")
        logger.info(f"🎯 F1-Score: {f1_score:.4f}")
        logger.info(f"🎯 Weighted Precision: {precision:.4f}")
        logger.info(f"🎯 Weighted Recall: {recall:.4f}")
        
        # Confusion Matrix
        logger.info("\nGenerating confusion matrix...")
        y_true = predictions.select("is_fraud").toPandas()
        y_pred = predictions.select("prediction").toPandas()
        
        cm = confusion_matrix(y_true, y_pred)
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"True Negatives:  {cm[0][0]:,}")
        logger.info(f"False Positives: {cm[0][1]:,}")
        logger.info(f"False Negatives: {cm[1][0]:,}")
        logger.info(f"True Positives:  {cm[1][1]:,}")
        
        # Calculate metrics per class
        tn, fp, fn, tp = cm.ravel()
        
        fraud_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fraud_f1 = 2 * (fraud_precision * fraud_recall) / (fraud_precision + fraud_recall) if (fraud_precision + fraud_recall) > 0 else 0
        
        logger.info(f"\n📊 Fraud Class (Class 1) Metrics:")
        logger.info(f"Precision: {fraud_precision:.4f}")
        logger.info(f"Recall:    {fraud_recall:.4f}")
        logger.info(f"F1-Score:  {fraud_f1:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Unpersist predictions
        predictions.unpersist()
        
        return predictions, cm
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix heatmap
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Not Fraud', 'Predicted Fraud'],
                    yticklabels=['Actual Not Fraud', 'Actual Fraud'],
                    cbar_kws={'label': 'Count'})
        plt.title('Random Forest Model - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        filename = 'confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Confusion matrix saved to {filename}")
        plt.close()
    
    def save_model(self, model, path="models/fraud_detection_rf_pipeline"):
        """
        Save the complete pipeline model
        """
        logger.info(f"\n💾 Saving model to {path}...")
        
        save_start = time.time()
        
        try:
            # Ensure the directory exists
            model_dir = os.path.dirname(path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logger.info(f"Created directory: {model_dir}")
            
            # Save the model
            model.write().overwrite().save(path)
            
            save_time = time.time() - save_start
            logger.info(f"✅ Model saved successfully in {save_time:.2f} seconds!")
            logger.info(f"📁 Model location: {os.path.abspath(path)}")
            
            # Verify the model was saved
            if os.path.exists(path):
                logger.info("✅ Verified: Model directory exists")
                
                # List contents
                contents = os.listdir(path)
                logger.info(f"Model directory contains {len(contents)} files/folders")
            else:
                logger.error("❌ ERROR: Model directory not found after save!")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_training_pipeline(self):
        """
        Execute complete training pipeline (OPTIMIZED VERSION)
        """
        total_start = time.time()
        
        try:
            logger.info("\n" + "="*70)
            logger.info("🚀 STARTING OPTIMIZED FRAUD DETECTION MODEL TRAINING")
            logger.info("="*70)
            
            # Step 1: Load data
            df = self.load_data()
            
            # Step 2: Quick missing values check
            df = self.check_missing_values(df)
            
            # Step 3: Preprocess data
            df_processed = self.preprocess_data(df)
            
            # Step 4: Identify features
            categorical_features, numerical_features = self.identify_features(df_processed)
            
            # Step 5: Add class weights
            df_weighted = self.add_class_weights(df_processed)
            
            # Step 6: Split data
            logger.info("\nSplitting data into train (80%) and test (20%) sets...")
            split_start = time.time()
            
            train_data, test_data = df_weighted.randomSplit([0.8, 0.2], seed=42)
            
            # Cache for faster access
            train_data.cache()
            test_data.cache()
            
            train_count = train_data.count()
            test_count = test_data.count()
            
            split_time = time.time() - split_start
            
            logger.info(f"Training data count: {train_count:,} (in {split_time:.2f}s)")
            logger.info(f"Testing data count: {test_count:,}")
            
            # Step 7: Build pipeline
            pipeline, rf = self.build_pipeline(categorical_features, numerical_features)
            
            # Step 8: Train model (FAST MODE)
            tvs_model, best_model = self.train_model_fast(pipeline, rf, train_data)
            
            # Step 9: Evaluate model
            predictions, cm = self.evaluate_model(best_model, test_data)
            
            # Step 10: Save model
            self.save_model(best_model)
            
            total_time = time.time() - total_start
            
            logger.info("\n" + "="*70)
            logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info("\nModel ready for deployment in streaming pipeline!")
            logger.info(f"Model saved at: models/fraud_detection_rf_pipeline")
            logger.info(f"Full path: {os.path.abspath('models/fraud_detection_rf_pipeline')}")
            
            # Cleanup
            train_data.unpersist()
            test_data.unpersist()
            df.unpersist()
            
            return best_model, predictions
            
        except Exception as e:
            logger.error(f"❌ Error in training pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            total_time = time.time() - total_start
            logger.info(f"\nTotal pipeline execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    import sys
    
    # Allow custom data path as command line argument
    data_path = sys.argv[1] if len(sys.argv) > 1 else "fraudTest.csv"
    
    logger.info(f"Using data file: {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        logger.error(f"❌ ERROR: Data file not found: {data_path}")
        logger.error("Please ensure fraudTest.csv is in the current directory")
        sys.exit(1)
    
    trainer = FraudModelTrainer(data_path=data_path)
    model, predictions = trainer.run_training_pipeline()
    
    logger.info("\n🎉 Training completed! Model is ready for streaming deployment.")
    logger.info("\nNext steps:")
    logger.info("1. Verify model exists: ls -la models/fraud_detection_rf_pipeline")
    logger.info("2. Run streaming processor: spark-submit spark_streaming_processor.py")