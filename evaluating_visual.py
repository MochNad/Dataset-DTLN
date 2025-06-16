import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Create log directory and configure logging
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'evaluating_visual.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_plots_folder():
    """Create evaluates/visuals folder if it doesn't exist"""
    plots_dir = Path("evaluates") / "visuals"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir

def load_data(csv_path):
    """Load CSV data - can be single file or directory structure"""
    try:
        csv_path = Path(csv_path)
        
        # Check if it's a single CSV file
        if csv_path.is_file() and csv_path.suffix.lower() == '.csv':
            df = pd.read_csv(csv_path)
            logger.info(f"Single CSV loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        
        # Check if it's the analyses/metrics directory structure
        elif csv_path.is_dir() or (csv_path.name == 'metrics' and csv_path.parent.name == 'analyses'):
            # Handle both cases: direct path to metrics folder or analyses folder
            if csv_path.name == 'analyses':
                metrics_dir = csv_path / 'metrics'
            elif csv_path.name == 'metrics':
                metrics_dir = csv_path
            else:
                # Assume it's a path that contains analyses/metrics
                metrics_dir = csv_path / 'analyses' / 'metrics'
            
            if not metrics_dir.exists():
                logger.error(f"Metrics directory not found: {metrics_dir}")
                return None
            
            # Load and combine all individual CSV files
            all_dataframes = []
            csv_files_found = 0
            
            for lang_dir in metrics_dir.iterdir():
                if not lang_dir.is_dir():
                    continue
                
                language = lang_dir.name
                logger.info(f"Processing language: {language}")
                
                for csv_file in lang_dir.glob('*.csv'):
                    try:
                        # Load individual CSV
                        individual_df = pd.read_csv(csv_file)
                        
                        # Add language and clean audio name columns back
                        clean_name = csv_file.stem  # filename without extension
                        individual_df['clean_audio_language'] = language
                        individual_df['clean_audio_name'] = clean_name
                        
                        all_dataframes.append(individual_df)
                        csv_files_found += 1
                        logger.debug(f"Loaded: {csv_file} (Shape: {individual_df.shape})")
                        
                    except Exception as e:
                        logger.warning(f"Error loading {csv_file}: {e}")
            
            if not all_dataframes:
                logger.error("No valid CSV files found in the metrics directory structure")
                return None
            
            # Combine all dataframes
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Reorder columns to match expected structure
            expected_columns = [
                'clean_audio_language',
                'clean_audio_name', 
                'clean_audio_duration',
                'noise_category',
                'noise_type',
                'snr_level',
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
            
            # Reorder columns if they exist
            available_columns = [col for col in expected_columns if col in combined_df.columns]
            other_columns = [col for col in combined_df.columns if col not in expected_columns]
            final_columns = available_columns + other_columns
            
            combined_df = combined_df[final_columns]
            
            logger.info(f"Combined data loaded successfully from {csv_files_found} CSV files")
            logger.info(f"Final shape: {combined_df.shape}")
            logger.info(f"Languages: {sorted(combined_df['clean_audio_language'].unique())}")
            logger.info(f"Columns: {combined_df.columns.tolist()}")
            
            return combined_df
        
        else:
            logger.error(f"Invalid path: {csv_path}. Must be a CSV file or metrics directory")
            return None
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def plot_quality_vs_snr_linechart(df, plots_dir):
    """Line Chart: Quality vs SNR Level"""
    plt.figure(figsize=(12, 8))
    
    if 'snr_level' in df.columns and 'pesq_score' in df.columns:
        # Plot PESQ Score
        snr_pesq = df.groupby('snr_level')['pesq_score'].mean().reset_index()
        plt.plot(snr_pesq['snr_level'], snr_pesq['pesq_score'], 
                marker='o', linewidth=3, markersize=10, color='#1976D2', label='PESQ Score')
        
        # Plot STOI Score if available
        if 'stoi_score' in df.columns:
            snr_stoi = df.groupby('snr_level')['stoi_score'].mean().reset_index()
            plt.plot(snr_stoi['snr_level'], snr_stoi['stoi_score'], 
                    marker='s', linewidth=3, markersize=10, color='#42A5F5', label='STOI Score')
        
        # Add value labels on points
        for i, row in snr_pesq.iterrows():
            plt.annotate(f'{row["pesq_score"]:.3f}', 
                        (row['snr_level'], row['pesq_score']),
                        textcoords="offset points", xytext=(0,15), ha='center', fontsize=10)
        
        if 'stoi_score' in df.columns:
            for i, row in snr_stoi.iterrows():
                plt.annotate(f'{row["stoi_score"]:.3f}', 
                            (row['snr_level'], row['stoi_score']),
                            textcoords="offset points", xytext=(0,-20), ha='center', fontsize=10)
        
        plt.xlabel('SNR Level', fontsize=12, fontweight='bold')
        plt.ylabel('Average Quality Score', fontsize=12, fontweight='bold')
        plt.title('Audio Quality vs Noise Level (SNR)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'quality_vs_snr_linechart.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Quality vs SNR Line Chart saved")
    else:
        logger.warning("Missing columns for Quality vs SNR plot")

def plot_quality_by_noise_category_barchart(df, plots_dir):
    """Bar Chart: Quality by Noise Category"""
    plt.figure(figsize=(12, 8))
    
    if 'noise_category' in df.columns and 'pesq_score' in df.columns:
        category_performance = df.groupby('noise_category')['pesq_score'].mean().reset_index()
        bars = plt.bar(category_performance['noise_category'], category_performance['pesq_score'], 
                      color='#2196F3', alpha=0.8, label='Average PESQ Score')
        
        plt.xlabel('Noise Category', fontsize=12, fontweight='bold')
        plt.ylabel('Average PESQ Score', fontsize=12, fontweight='bold')
        plt.title('Model Performance by Noise Category', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, fontsize=11)
        plt.legend(fontsize=11)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'quality_by_noise_category_barchart.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Quality by Noise Category Bar Chart saved")
    else:
        logger.warning("Missing columns for Quality by Noise Category plot")

def plot_quality_vs_performance_scatterplot(df, plots_dir):
    """Scatter Plot: Quality vs Performance Trade-off"""
    plt.figure(figsize=(12, 8))
    
    performance_col = None
    for col in ['worker_ms_avg', 'worklet_ms_avg', 'model1_ms_avg', 'model2_ms_avg']:
        if col in df.columns:
            performance_col = col
            break
    
    if performance_col and 'pesq_score' in df.columns:
        # Create scatter plot with different colors for SNR levels if available
        if 'snr_level' in df.columns:
            snr_levels = df['snr_level'].unique()
            colors = ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9']
            
            for i, snr in enumerate(snr_levels):
                snr_data = df[df['snr_level'] == snr]
                plt.scatter(snr_data[performance_col], snr_data['pesq_score'], 
                           alpha=0.7, color=colors[i % len(colors)], s=60, label=f'SNR: {snr}')
        else:
            plt.scatter(df[performance_col], df['pesq_score'], alpha=0.7, 
                       color='#1976D2', s=60, label='Data Points')
        
        plt.xlabel(f'{performance_col.replace("_", " ").title()} (ms)', fontsize=12, fontweight='bold')
        plt.ylabel('PESQ Score', fontsize=12, fontweight='bold')
        plt.title('Quality vs Performance Trade-off', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        
        # Add correlation coefficient
        corr = df[performance_col].corr(df['pesq_score'])
        plt.text(0.05, 0.95, f'Overall Correlation: {corr:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8), fontsize=11)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'quality_vs_performance_scatterplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Quality vs Performance Scatter Plot saved")
    else:
        logger.warning("Missing columns for Quality vs Performance plot")

def plot_language_comparison_barchart(df, plots_dir):
    """Grouped Bar Chart: Language Comparison"""
    plt.figure(figsize=(12, 8))
    
    if 'clean_audio_language' in df.columns:
        metrics = ['pesq_score', 'stoi_score']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if available_metrics:
            lang_comparison = df.groupby('clean_audio_language')[available_metrics].mean()
            
            # Create grouped bar chart
            x = np.arange(len(lang_comparison.index))
            width = 0.35
            
            if len(available_metrics) == 2:
                bars1 = plt.bar(x - width/2, lang_comparison[available_metrics[0]], width, 
                               label='PESQ Score', color='#1976D2', alpha=0.8)
                bars2 = plt.bar(x + width/2, lang_comparison[available_metrics[1]], width,
                               label='STOI Score', color='#42A5F5', alpha=0.8)
                
                # Add value labels
                for bar in bars1:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                for bar in bars2:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                bars = plt.bar(x, lang_comparison[available_metrics[0]], width,
                              label=available_metrics[0].replace('_', ' ').title(), color='#1976D2', alpha=0.8)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.xlabel('Language', fontsize=12, fontweight='bold')
            plt.ylabel('Average Score', fontsize=12, fontweight='bold')
            plt.title('Model Performance: English vs Indonesian', fontsize=14, fontweight='bold')
            plt.xticks(x, lang_comparison.index, fontsize=11)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'language_comparison_barchart.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Language Comparison Bar Chart saved")
        else:
            logger.warning("No valid metrics found for Language Comparison")
    else:
        logger.warning("Missing clean_audio_language column for Language Comparison plot")

def plot_metrics_correlation_pairplot(df, plots_dir):
    """Pair Plot: Correlation between Quality Metrics"""
    quality_metrics = ['pesq_score', 'stoi_score', 'mse_score']
    available_metrics = [m for m in quality_metrics if m in df.columns]
    
    if len(available_metrics) >= 2:
        plt.figure(figsize=(14, 12))
        
        if len(available_metrics) >= 3:
            # Create pair plot with blue colors
            g = sns.PairGrid(df[available_metrics], height=4)
            g.map_upper(sns.scatterplot, alpha=0.7, color='#64B5F6', s=40)
            g.map_lower(sns.scatterplot, alpha=0.7, color='#1976D2', s=40)
            g.map_diag(sns.histplot, color='#0D47A1', alpha=0.8)
            
            # Add correlation values
            def add_corr(x, y, **kwargs):
                corr = x.corr(y)
                ax = plt.gca()
                ax.text(0.1, 0.9, f'r = {corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                       fontsize=12, fontweight='bold')
            
            g.map_upper(add_corr)
            
            # Set titles and labels
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                g.axes[i, j].set_title(f'{available_metrics[j]} vs {available_metrics[i]}', fontsize=11)
            
        else:
            # Simple scatter plot for 2 metrics
            plt.scatter(df[available_metrics[0]], df[available_metrics[1]], 
                       alpha=0.7, color='#1976D2', s=60, label='Data Points')
            plt.xlabel(available_metrics[0].replace('_', ' ').title(), fontsize=12, fontweight='bold')
            plt.ylabel(available_metrics[1].replace('_', ' ').title(), fontsize=12, fontweight='bold')
            plt.title('Quality Metrics Correlation', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            
            # Add correlation coefficient
            corr = df[available_metrics[0]].corr(df[available_metrics[1]])
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                    fontsize=12, fontweight='bold')
            
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'metrics_correlation_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Metrics Correlation Pair Plot saved")
    else:
        logger.warning("Not enough quality metrics for correlation plot")

def main():
    parser = argparse.ArgumentParser(description='Generate audio quality analysis plots')
    parser.add_argument('--metric', required=True, 
                       help='Path to CSV file or analyses/metrics directory containing individual CSV files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create evaluates/visuals directory
    plots_dir = create_plots_folder()
    logger.info(f"Plots will be saved to: {plots_dir.absolute()}")
    
    # Load data (supports both single CSV and directory structure)
    logger.info(f"Loading data from: {Path(args.metric).absolute()}")
    df = load_data(args.metric)
    if df is None:
        logger.error("Failed to load data. Please check the path and try again.")
        return 1
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("Blues")
    
    # Count total plots to generate
    total_plots = 5
    
    # Generate all plots with progress bar
    logger.info(f"Generating {total_plots} visualization plots")
    
    plot_functions = [
        (plot_quality_vs_snr_linechart, "Quality vs SNR Line Chart"),
        (plot_quality_by_noise_category_barchart, "Quality by Noise Category Bar Chart"),
        (plot_quality_vs_performance_scatterplot, "Quality vs Performance Scatter Plot"),
        (plot_language_comparison_barchart, "Language Comparison Bar Chart"),
        (plot_metrics_correlation_pairplot, "Metrics Correlation Pair Plot")
    ]
    
    success_count = 0
    current_plot = 1
    
    with tqdm(total=total_plots, desc="Generating Charts", 
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
             ncols=120) as pbar:
        
        for plot_func, plot_name in plot_functions:
            # Format progress description to match analyzing_visual.py style
            progress_desc = f"[{current_plot:4d}/{total_plots}] {plot_name[:50]}"
            pbar.set_description(progress_desc)
            
            try:
                plot_func(df, plots_dir)
                success_count += 1
                pbar.set_postfix_str(f"{success_count}/{current_plot}")
            except Exception as e:
                logger.error(f"Error generating {plot_name}: {e}")
                pbar.set_postfix_str(f"Failed: {current_plot - success_count}")
            
            pbar.update(1)
            current_plot += 1
    
    # Final summary matching analyzing_visual.py style
    failure_count = total_plots - success_count
    success_rate = (success_count / total_plots * 100) if total_plots > 0 else 0
    
    logger.info("Plot generation completed!")
    logger.info(f"Success: {success_count}/{total_plots} ({success_rate:.1f}%)")
    if failure_count > 0:
        logger.warning(f"Failures: {failure_count} plots could not be generated")
    
    logger.info(f"All plots generated successfully in '{plots_dir}' folder!")
    return 0

if __name__ == "__main__":
    exit(main())
