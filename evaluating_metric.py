import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Create log directory and configure logging
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'evaluating_metric.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MetricEvaluator:
    """Evaluate and summarize metrics from individual CSV files"""
    
    def __init__(self, metrics_folder: Path):
        self.metrics_folder = metrics_folder
        self.combined_data = []
        
    def load_individual_metrics(self) -> pd.DataFrame:
        """Load and combine all individual CSV metric files"""
        if not self.metrics_folder.exists():
            raise ValueError(f"Metrics folder does not exist: {self.metrics_folder}")
        
        csv_files_found = 0
        
        # Walk through the metrics directory structure
        for lang_dir in self.metrics_folder.iterdir():
            if not lang_dir.is_dir():
                continue
                
            language = lang_dir.name
            logger.info(f"Processing language: {language}")
            
            for csv_file in lang_dir.glob('*.csv'):
                try:
                    # Load individual CSV
                    df = pd.read_csv(csv_file)
                    
                    if df.empty:
                        logger.warning(f"Empty CSV file: {csv_file}")
                        continue
                    
                    # Add metadata columns back
                    clean_name = csv_file.stem  # filename without extension
                    df['clean_audio_language'] = language
                    df['clean_audio_name'] = clean_name
                    
                    # Convert string columns back to numeric for calculations
                    numeric_columns = [
                        'clean_audio_duration', 'pesq_score', 'stoi_score', 'mse_score',
                        'buffer_queue_avg', 'buffer_droped_avg', 'worklet_ms_avg',
                        'worker_ms_avg', 'model1_ms_avg', 'model2_ms_avg'
                    ]
                    
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    self.combined_data.append(df)
                    csv_files_found += 1
                    logger.debug(f"Loaded: {csv_file} (Shape: {df.shape})")
                    
                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")
        
        if not self.combined_data:
            raise ValueError("No valid CSV files found in the metrics directory")
        
        # Combine all dataframes
        combined_df = pd.concat(self.combined_data, ignore_index=True)
        logger.info(f"Combined {csv_files_found} CSV files into dataset with shape: {combined_df.shape}")
        
        return combined_df
    
    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate min, max, std, mean for specified columns"""
        # Define columns to calculate statistics for
        stat_columns = [
            'clean_audio_duration',
            'pesq_score', 
            'stoi_score', 
            'mse_score',
            'buffer_queue_avg',
            'buffer_droped_avg', 
            'worklet_ms_avg',
            'worker_ms_avg',
            'model1_ms_avg',
            'model2_ms_avg'
        ]
        
        # Filter to only existing columns
        available_columns = [col for col in stat_columns if col in df.columns]
        
        if not available_columns:
            raise ValueError("No valid numeric columns found for statistics calculation")
        
        logger.info(f"Calculating statistics for columns: {available_columns}")
        
        # Calculate statistics
        statistics = []
        
        for col in available_columns:
            # Get valid (non-NaN) values
            valid_values = df[col].dropna()
            
            if len(valid_values) == 0:
                logger.warning(f"No valid values found for column: {col}")
                stats = {
                    'metric': col,
                    'count': 0,
                    'min': np.nan,
                    'max': np.nan,
                    'mean': np.nan,
                    'std': np.nan
                }
            else:
                stats = {
                    'metric': col,
                    'count': len(valid_values),
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std())
                }
                
                logger.debug(f"{col}: count={stats['count']}, min={stats['min']:.6f}, "
                           f"max={stats['max']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")
            
            statistics.append(stats)
        
        # Create statistics DataFrame
        stats_df = pd.DataFrame(statistics)
        
        # Reorder columns
        column_order = ['metric', 'count', 'min', 'max', 'mean', 'std']
        stats_df = stats_df[column_order]
        
        return stats_df
    
    def save_statistics(self, stats_df: pd.DataFrame, output_path: Path) -> None:
        """Save statistics to CSV file with proper formatting"""
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format statistics with appropriate precision
        formatted_df = stats_df.copy()
        
        # Format numeric columns (except count)
        numeric_cols = ['min', 'max', 'mean', 'std']
        for col in numeric_cols:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.6f}" if pd.notna(x) else ""
            )
        
        # Save to CSV
        formatted_df.to_csv(output_path, index=False, na_rep='')
        logger.info(f"Statistics saved to: {output_path}")
        
        # Print summary
        self._print_statistics_summary(stats_df, output_path)
    
    def _print_statistics_summary(self, stats_df: pd.DataFrame, output_path: Path) -> None:
        """Print summary of calculated statistics"""
        print("\n" + "="*80)
        print("METRIC EVALUATION SUMMARY")
        print("="*80)
        
        print(f"Output file: {output_path.resolve()}")
        print(f"Total metrics evaluated: {len(stats_df)}")
        
        # Show statistics for each metric
        print(f"\nStatistical Summary:")
        print("-" * 70)
        print(f"{'Metric':<20} {'Count':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print("-" * 70)
        
        for _, row in stats_df.iterrows():
            metric_name = row['metric']
            count = int(row['count']) if pd.notna(row['count']) else 0
            
            if count > 0:
                print(f"{metric_name:<20} {count:<8} {row['min']:<12.6f} {row['max']:<12.6f} "
                      f"{row['mean']:<12.6f} {row['std']:<12.6f}")
            else:
                print(f"{metric_name:<20} {count:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        
        print("="*80)
    
    def run_evaluation(self, output_path: Path) -> None:
        """Run complete evaluation process"""
        logger.info("Starting metric evaluation...")
        
        # Load and combine individual CSV files
        combined_df = self.load_individual_metrics()
        
        # Calculate statistics
        stats_df = self.calculate_statistics(combined_df)
        
        # Save results
        self.save_statistics(stats_df, output_path)
        
        logger.info("Metric evaluation completed successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Evaluate and summarize audio quality metrics from individual CSV files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--metric', required=True,
                       help='Path to folder containing individual metric CSV files (e.g., analyses/metrics)')
    parser.add_argument('-o', '--output', default='evaluates/metrics.csv',
                       help='Output CSV file path for statistics summary')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate and resolve paths
    metrics_folder = Path(args.metric).resolve()
    output_path = Path(args.output).resolve()
    
    if not metrics_folder.exists():
        logger.error(f"Metrics folder does not exist: {metrics_folder}")
        return 1
    
    if not metrics_folder.is_dir():
        logger.error(f"Metrics path is not a directory: {metrics_folder}")
        return 1
    
    print(f"Starting metric evaluation...")
    print(f"Metrics folder: {metrics_folder}")
    print(f"Output file: {output_path}")
    
    try:
        evaluator = MetricEvaluator(metrics_folder)
        evaluator.run_evaluation(output_path)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
