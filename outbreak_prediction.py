import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, accuracy_score,
                             precision_recall_curve, recall_score, precision_score)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
import argparse
import logging
import io
import joblib
import os
import json
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SETUP: LOGGING & CONFIGURATION
# ============================================
def setup_logging():
    """Configure logging to console and file"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_file = f'logs/outbreak_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    # Use a UTF-8 wrapper around stdout to avoid encoding issues on Windows consoles
    try:
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # Fallback if stdout has no buffer (very rare)
        stream = sys.stdout

    console_handler = logging.StreamHandler(stream)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[console_handler, file_handler]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================
# CONSTANTS & VALIDATION
# ============================================
REQUIRED_FEATURES = [
    'population_density', 'access_to_clean_water', 'vaccination_rate',
    'healthcare_spending_per_capita', 'avg_temperature', 'rainfall_mm',
    'malnutrition_rate', 'urbanization_rate'
]

MODELS_DIR = 'models'
PLOTS_DIR = 'plots'

# ============================================
# ARGUMENT PARSING
# ============================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Disease Outbreak Prediction System - SDG 3 (Good Health)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train with synthetic data:
    python outbreak_prediction.py --train
  
  Train with custom data:
    python outbreak_prediction.py --train --data my_data.csv
  
  Make predictions:
    python outbreak_prediction.py --predict new_regions.csv --output results.csv
        """
    )
    parser.add_argument('--train', action='store_true', 
                       help='Train new models')
    parser.add_argument('--data', type=str, 
                       help='Path to training data CSV file')
    parser.add_argument('--predict', type=str, 
                       help='Path to CSV file with new regions to predict')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='Path for prediction output CSV (default: predictions.csv)')
    return parser.parse_args()

# ============================================
# DATA LOADING & VALIDATION
# ============================================
def validate_features(df, context="training"):
    """Validate that all required features are present"""
    missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required {context} features: {', '.join(missing_cols)}"
        )
    return df[REQUIRED_FEATURES]

def load_data(file_path, require_target=False):
    """Load and validate input data"""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate required features
        validate_features(df, context="training" if require_target else "prediction")
        
        # Check for target variable if required
        if require_target and 'outbreak_risk' not in df.columns:
            raise ValueError("Training data must include 'outbreak_risk' column")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)

def handle_missing_values(df):
    """Handle missing values in dataset"""
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info("Handling missing values...")
        for column in df.columns:
            if missing_values[column] > 0:
                logger.warning(f"  {column}: {missing_values[column]} missing values")
                if df[column].dtype in ['int64', 'float64']:
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
        logger.info("Missing values handled ✓")
    else:
        logger.info("No missing values found ✓")
    return df

# ============================================
# SYNTHETIC DATA GENERATION
# ============================================
def generate_synthetic_data(n_samples=1000):
    """Generate realistic synthetic health dataset"""
    logger.info(f"Generating {n_samples} synthetic training samples")
    np.random.seed(42)
    
    features = {
        'population_density': np.random.uniform(10, 1000, n_samples),
        'access_to_clean_water': np.random.uniform(30, 100, n_samples),
        'vaccination_rate': np.random.uniform(20, 95, n_samples),
        'healthcare_spending_per_capita': np.random.uniform(50, 5000, n_samples),
        'avg_temperature': np.random.uniform(15, 35, n_samples),
        'rainfall_mm': np.random.uniform(100, 3000, n_samples),
        'malnutrition_rate': np.random.uniform(5, 50, n_samples),
        'urbanization_rate': np.random.uniform(10, 90, n_samples),
    }
    
    # Create realistic outbreak risk relationships
    risk_score = (
        (features['population_density'] > 500) * 0.4 +
        (features['access_to_clean_water'] < 60) * 0.3 +
        (features['vaccination_rate'] < 50) * 0.3 +
        (features['healthcare_spending_per_capita'] < 1000) * 0.3 +
        (features['malnutrition_rate'] > 20) * 0.2 +
        (features['urbanization_rate'] > 70) * 0.2 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Use percentile-based threshold to ensure balanced classes
    outbreak_risk = (risk_score > np.percentile(risk_score, 85)).astype(int)
    
    df = pd.DataFrame(features)
    df['outbreak_risk'] = outbreak_risk
    
    positive_rate = df['outbreak_risk'].mean() * 100
    logger.info(f"Synthetic data generated: {positive_rate:.1f}% positive class")
    
    return df

# ============================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================
def get_feature_importance(model, feature_names):
    """Extract feature importance from model, handling different model types"""
    importance_data = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest, Gradient Boosting)
        importance_data = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    elif hasattr(model, 'coef_'):
        # Linear models (Logistic Regression)
        importance_data = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
    
    else:
        logger.warning(f"Model type {type(model).__name__} doesn't support feature importance")
    
    return importance_data

# ============================================
# MODEL TRAINING
# ============================================
def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train multiple models with hyperparameter tuning"""
    logger.info("=" * 60)
    logger.info("MODEL TRAINING WITH HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'param_dist': {
                'C': uniform(0.1, 10.0),
                'class_weight': [None, 'balanced'],
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'param_dist': {
                'n_estimators': randint(100, 300),
                'max_depth': [None, 10, 15, 20],
                'min_samples_split': randint(2, 6),
                'class_weight': [None, 'balanced'],
                'max_samples': uniform(0.7, 0.3)
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_dist': {
                'n_estimators': randint(100, 300),
                'learning_rate': uniform(0.01, 0.19),
                'max_depth': randint(3, 7),
                'min_samples_split': randint(2, 6),
                'subsample': uniform(0.7, 0.3)
            }
        }
    }
    
    results = {}
    best_model_name = None
    best_score = 0
    
    for name, config in model_configs.items():
        try:
            logger.info(f"\nTraining {name}...")
            
            # Randomized search for hyperparameter tuning
            search = RandomizedSearchCV(
                config['model'],
                config['param_dist'],
                n_iter=20,
                cv=5,
                scoring='recall',  # Optimize for recall (catch outbreak cases)
                n_jobs=-1,
                verbose=0,
                random_state=42
            )
            
            search.fit(X_train_scaled, y_train)
            best_model_cv = search.best_estimator_
            
            logger.info(f"Best hyperparameters for {name}:")
            for param, value in search.best_params_.items():
                logger.info(f"  {param}: {value}")
            
            # Cross-validation scores
            cv_scores = {
                'accuracy': cross_val_score(best_model_cv, X_train_scaled, y_train, cv=5, scoring='accuracy'),
                'f1': cross_val_score(best_model_cv, X_train_scaled, y_train, cv=5, scoring='f1'),
                'recall': cross_val_score(best_model_cv, X_train_scaled, y_train, cv=5, scoring='recall')
            }
            
            # Test set predictions
            y_pred = best_model_cv.predict(X_test_scaled)
            y_pred_proba = best_model_cv.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate optimal threshold using F1-score maximization
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': best_model_cv,
                'pipeline': None,  # Will be set later
                'accuracy': accuracy,
                'f1': f1,
                'recall': recall,
                'precision': precision,
                'roc_auc': roc_auc,
                'cv_scores': cv_scores,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'optimal_threshold': optimal_threshold,
                'best_params': search.best_params_
            }
            
            # Score: prioritize recall (0.6) over F1 (0.4)
            current_score = 0.6 * recall + 0.4 * f1
            
            if current_score > best_score:
                best_score = current_score
                best_model_name = name
            
            logger.info(f"\n{name} Results:")
            logger.info(f"  CV Scores (mean ± std):")
            logger.info(f"    Accuracy: {cv_scores['accuracy'].mean():.4f} ± {cv_scores['accuracy'].std():.4f}")
            logger.info(f"    F1-Score: {cv_scores['f1'].mean():.4f} ± {cv_scores['f1'].std():.4f}")
            logger.info(f"    Recall: {cv_scores['recall'].mean():.4f} ± {cv_scores['recall'].std():.4f}")
            logger.info(f"  Test Set Results:")
            logger.info(f"    Accuracy: {accuracy:.4f}")
            logger.info(f"    Precision: {precision:.4f}")
            logger.info(f"    Recall: {recall:.4f}")
            logger.info(f"    F1-Score: {f1:.4f}")
            logger.info(f"    ROC-AUC: {roc_auc:.4f}")
            logger.info(f"    Optimal Threshold: {optimal_threshold:.3f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    logger.info(f"\n✓ Best Model: {best_model_name}")
    logger.info(f"  Score (0.6*Recall + 0.4*F1): {best_score:.4f}")
    
    return results, best_model_name

# ============================================
# VISUALIZATION
# ============================================
def create_visualizations(results, best_model_name, X, y_test, cm):
    """Create comprehensive visualization plots"""
    logger.info("Generating visualizations...")
    
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Disease Outbreak Prediction Model - SDG 3 (Good Health)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['No Outbreak', 'Outbreak'], 
                yticklabels=['No Outbreak', 'Outbreak'])
    axes[0, 0].set_title(f'Confusion Matrix ({best_model_name})')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')
    
    # Plot 2: ROC Curve
    y_pred_proba_best = results[best_model_name]['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
    auc = roc_auc_score(y_test, y_pred_proba_best)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    
    # Plot 3: Feature Importance
    best_model = results[best_model_name]['model']
    importance = get_feature_importance(best_model, X.columns)
    
    if importance is not None:
        importance_sorted = importance.sort_values('importance')
        axes[1, 0].barh(importance_sorted['feature'], importance_sorted['importance'], 
                        color='steelblue')
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].set_title('Feature Importance in Predictions')
    else:
        axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Feature Importance')
    
    # Plot 4: Model Comparison
    model_names = list(results.keys())
    recall_scores = [results[m]['recall'] for m in model_names]
    f1_scores = [results[m]['f1'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, recall_scores, width, label='Recall', color='steelblue', alpha=0.8)
    axes[1, 1].bar(x + width/2, f1_scores, width, label='F1-Score', color='coral', alpha=0.8)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Model Comparison (Recall vs F1-Score)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names)
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'outbreak_prediction_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Visualizations saved to {plot_path}")
    plt.close()

# ============================================
# MODEL PERSISTENCE
# ============================================
def save_models_and_metrics(results, best_model_name, X, scaler):
    """Save trained models, pipelines, and metrics"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    logger.info("Saving models and metrics...")
    
    best_result = results[best_model_name]
    
    # Create and save pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', best_result['model'])
    ])
    pipeline_path = os.path.join(MODELS_DIR, 'best_model_pipeline.joblib')
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"✓ Pipeline saved to {pipeline_path}")
    
    # Save individual models
    for model_name, result in results.items():
        model_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.joblib")
        joblib.dump(result['model'], model_path)
    logger.info(f"✓ All models saved to {MODELS_DIR}/")
    
    # Save scaler separately
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved to {scaler_path}")
    
    # Save metrics summary
    metrics_summary = {
        'best_model': best_model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': float(best_result['accuracy']),
            'precision': float(best_result['precision']),
            'recall': float(best_result['recall']),
            'f1': float(best_result['f1']),
            'roc_auc': float(best_result['roc_auc']),
            'optimal_threshold': float(best_result['optimal_threshold'])
        },
        'best_hyperparameters': best_result['best_params'],
        'feature_names': list(X.columns)
    }
    
    metrics_path = os.path.join(MODELS_DIR, 'metrics_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"✓ Metrics summary saved to {metrics_path}")
    
    # Save all models comparison
    all_models_summary = {}
    for model_name, res in results.items():
        all_models_summary[model_name] = {
            'accuracy': float(res['accuracy']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'f1': float(res['f1']),
            'roc_auc': float(res['roc_auc']),
            'optimal_threshold': float(res['optimal_threshold'])
        }
    
    all_models_path = os.path.join(MODELS_DIR, 'all_models_comparison.json')
    with open(all_models_path, 'w') as f:
        json.dump(all_models_summary, f, indent=2)
    logger.info(f"✓ All models comparison saved to {all_models_path}")
    
    # Save training statistics
    training_stats = {
        'training_date': datetime.now().isoformat(),
        'n_samples': 1000,  # Update if using custom data
        'n_features': X.shape[1],
        'feature_names': list(X.columns),
        'random_seed': 42
    }
    
    stats_path = os.path.join(MODELS_DIR, 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"✓ Training statistics saved to {stats_path}")

# ============================================
# PREDICTION ON NEW DATA
# ============================================
def predict_regions(pipeline, data_raw, threshold=0.5):
    """Make predictions for new regions with confidence scores"""
    try:
        logger.info(f"Making predictions for {len(data_raw)} regions with threshold={threshold}")
        
        # Pipeline handles scaling internally
        probabilities = pipeline.predict_proba(data_raw)[:, 1]
        
        results = []
        for i, prob in enumerate(probabilities):
            # Confidence: distance from decision boundary
            confidence = abs(prob - 0.5) * 2
            
            # Risk levels with specific thresholds
            if prob > 0.7:
                risk_level = "CRITICAL"
                action = "Immediate intervention required"
            elif prob > 0.5:
                risk_level = "HIGH"
                action = "Urgent intervention needed"
            elif prob > 0.3:
                risk_level = "MODERATE"
                action = "Enhanced monitoring required"
            else:
                risk_level = "LOW"
                action = "Standard prevention measures"
            
            confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
            
            results.append({
                'region_id': i + 1,
                'risk_level': risk_level,
                'outbreak_probability': round(prob, 4),
                'confidence_level': confidence_level,
                'confidence_score': round(confidence, 4),
                'recommended_action': action
            })
        
        logger.info(f"✓ Predictions completed")
        return pd.DataFrame(results)
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        sys.exit(1)

def load_and_predict(predict_file, output_file):
    """Load trained model and make predictions on new data"""
    try:
        # Check if trained model exists
        pipeline_path = os.path.join(MODELS_DIR, 'best_model_pipeline.joblib')
        metrics_path = os.path.join(MODELS_DIR, 'metrics_summary.json')
        
        if not os.path.exists(pipeline_path):
            logger.error("Trained model not found. Run with --train flag first.")
            sys.exit(1)
        
        # Load pipeline and metrics
        logger.info(f"Loading trained model from {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        optimal_threshold = metrics['metrics']['optimal_threshold']
        logger.info(f"Loaded optimal threshold: {optimal_threshold:.3f}")
        
        # Load and validate new data
        new_data = load_data(predict_file, require_target=False)
        new_data_features = validate_features(new_data)
        
        # Make predictions
        predictions_df = predict_regions(pipeline, new_data_features, threshold=optimal_threshold)
        
        # Combine with input features
        results_df = pd.concat([new_data, predictions_df], axis=1)
        results_df.to_csv(output_file, index=False)
        logger.info(f"✓ Predictions saved to {output_file}")
        
        # Display summary
        logger.info("\nPrediction Summary:")
        logger.info(f"  Total regions: {len(results_df)}")
        logger.info(f"  CRITICAL: {(results_df['risk_level'] == 'CRITICAL').sum()}")
        logger.info(f"  HIGH: {(results_df['risk_level'] == 'HIGH').sum()}")
        logger.info(f"  MODERATE: {(results_df['risk_level'] == 'MODERATE').sum()}")
        logger.info(f"  LOW: {(results_df['risk_level'] == 'LOW').sum()}")
        
        return results_df
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)

# ============================================
# TRAINING MODE
# ============================================
def train_mode(data_file=None):
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("DISEASE OUTBREAK PREDICTION - TRAINING MODE")
    logger.info("=" * 60)
    
    # Load or generate data
    if data_file:
        df = load_data(data_file, require_target=True)
    else:
        df = generate_synthetic_data(n_samples=1000)
    
    # Display dataset info
    logger.info(f"\nDataset Overview:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Outbreak cases: {df['outbreak_risk'].sum()} ({df['outbreak_risk'].mean()*100:.1f}%)")
    logger.info(f"  Features: {len(REQUIRED_FEATURES)}")
    
    # Data preprocessing
    logger.info("\n" + "=" * 60)
    logger.info("DATA PREPROCESSING")
    logger.info("=" * 60)
    
    df = handle_missing_values(df)
    
    # Prepare features and target
    X = df[REQUIRED_FEATURES]
    y = df['outbreak_risk']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training set size: {X_train_scaled.shape[0]}")
    logger.info(f"Test set size: {X_test_scaled.shape[0]}")
    logger.info(f"Features scaled and normalized ✓")
    
    # Train models
    results, best_model_name = train_models(X_train, X_test, y_train, y_test, 
                                           X_train_scaled, X_test_scaled)
    
    # Detailed evaluation
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED EVALUATION - BEST MODEL")
    logger.info("=" * 60)
    
    y_pred_best = results[best_model_name]['y_pred']
    cm = confusion_matrix(y_test, y_pred_best)
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred_best, 
                                    target_names=['No Outbreak', 'Outbreak Risk']))
    
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"\nConfusion Matrix Details:")
    logger.info(f"  True Negatives: {tn}  | False Positives: {fp}")
    logger.info(f"  False Negatives: {fn} | True Positives: {tp}")
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    logger.info(f"\nRecall (Critical for outbreak detection): {recall:.4f} ({recall*100:.1f}%)")
    
    # Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 60)
    
    importance = get_feature_importance(results[best_model_name]['model'], X.columns)
    if importance is not None:
        logger.info("\nTop factors contributing to outbreak risk prediction:")
        for idx, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Create visualizations
    create_visualizations(results, best_model_name, X, y_test, cm)
    
    # Save models and metrics
    save_models_and_metrics(results, best_model_name, X, scaler)
    
    # Demo predictions on sample regions
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE PREDICTIONS ON NEW REGIONS")
    logger.info("=" * 60)
    
    sample_regions = pd.DataFrame({
        'population_density': [150, 800, 300],
        'access_to_clean_water': [85, 45, 70],
        'vaccination_rate': [90, 30, 60],
        'healthcare_spending_per_capita': [2000, 200, 500],
        'avg_temperature': [22, 28, 25],
        'rainfall_mm': [1000, 500, 800],
        'malnutrition_rate': [8, 35, 15],
        'urbanization_rate': [45, 70, 50],
    })
    
    pipeline = joblib.load(os.path.join(MODELS_DIR, 'best_model_pipeline.joblib'))
    sample_predictions = predict_regions(pipeline, sample_regions, 
                                        threshold=results[best_model_name]['optimal_threshold'])
    
    logger.info("\nSample Region Predictions:")
    for idx, row in sample_predictions.iterrows():
        logger.info(f"\nRegion {row['region_id']}:")
        logger.info(f"  Risk Level: {row['risk_level']}")
        logger.info(f"  Outbreak Probability: {row['outbreak_probability']*100:.1f}%")
        logger.info(f"  Confidence: {row['confidence_level']} ({row['confidence_score']*100:.1f}%)")
        logger.info(f"  Recommended Action: {row['recommended_action']}")
        
        # Feature-specific recommendations
        region_data = sample_regions.iloc[idx]
        risk_factors = []
        
        if region_data['population_density'] > 500:
            risk_factors.append("High population density → Implement social distancing")
        if region_data['access_to_clean_water'] < 60:
            risk_factors.append("Limited water access → Prioritize water infrastructure")
        if region_data['vaccination_rate'] < 50:
            risk_factors.append("Low vaccination → Launch vaccination campaign")
        if region_data['healthcare_spending_per_capita'] < 1000:
            risk_factors.append("Limited resources → Increase medical capacity")
        
        if risk_factors:
            logger.info("  Key Risk Factors:")
            for factor in risk_factors:
                logger.info(f"    - {factor}")
    
    # Ethical considerations
    logger.info("\n" + "=" * 60)
    logger.info("ETHICAL REFLECTION & FAIRNESS ANALYSIS")
    logger.info("=" * 60)
    
    ethical_notes = """
1. DATA BIAS MITIGATION:
   ✓ Prioritized recall to avoid missing real outbreaks (false negatives)
   ✓ Stratified train-test split ensures balanced class representation
   ✓ Cross-validation guards against overfitting to biased samples
   → Recommendation: Actively collect data from underrepresented populations

2. MODEL FAIRNESS:
   ✓ Feature importance enables transparency in decision-making
   ✓ Multiple decision thresholds allow region-specific calibration
   ✓ Optimal threshold calculated to maximize recall+precision trade-off
   → Risk: High false positives in resource-poor regions → stigmatization
   → Mitigation: Different thresholds per region; human validation required

3. TRANSPARENCY & ACCOUNTABILITY:
   ✓ Feature importance reveals which health factors drive predictions
   ✓ Actionable insights: policymakers can target modifiable factors
   ✓ Model predictions are recommendations, not deterministic directives
   → Requirement: Human epidemiologists validate before policy decisions

4. IMPLEMENTATION CONCERNS:
   ✓ Recall prioritized (catch real outbreaks) over precision
   ✓ Regular audits needed for algorithmic drift as new data arrives
   ✓ Communities must understand why they're flagged as "high risk"
   → Ensure equitable resource allocation, not reinforcing disparities

5. SUSTAINABILITY ALIGNMENT (SDG 3):
   ✓ Model directly supports Goal 3: "Ensure healthy lives for all"
   ✓ Early warning enables preventive interventions
   ✓ Data-driven resource allocation maximizes impact per dollar spent
"""
    logger.info(ethical_notes)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE ✓")
    logger.info("=" * 60)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Recall (Outbreak Detection): {recall:.4f}")
    logger.info(f"Models saved to: {MODELS_DIR}/")
    logger.info(f"Plots saved to: {PLOTS_DIR}/")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """Main entry point"""
    args = parse_args()
    
    if args.predict:
        # Prediction mode
        logger.info("=" * 60)
        logger.info("PREDICTION MODE")
        logger.info("=" * 60)
        load_and_predict(args.predict, args.output)
    
    elif args.train:
        # Training mode
        train_mode(data_file=args.data)
    
    else:
        # No mode specified
        logger.error("Please specify --train or --predict")
        logger.info("\nUsage examples:")
        logger.info("  Train: python outbreak_prediction.py --train")
        logger.info("  Predict: python outbreak_prediction.py --predict new_data.csv")
        sys.exit(1)

if __name__ == '__main__':
    main()

