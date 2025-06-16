import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import librosa
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

# Constants
TARGET_SAMPLE_RATE = 16000
SNR_LEVELS = ['-5 dB', '0 dB', '5 dB', '10 dB']
AUDIO_EXTENSION = '.wav'  # Only process .wav files

# Create log directory and configure logging
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'analyzing_metric.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioMetrics:
    """Audio quality metrics calculator for WAV files only"""
    
    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE):
        self.target_sr = target_sr
    
    @staticmethod
    def is_wav_file(file_path: Path) -> bool:
        """Check if file is a WAV file"""
        return file_path.suffix.lower() == AUDIO_EXTENSION
    
    def load_audio(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load WAV file and convert to target sample rate"""
        if not self.is_wav_file(file_path):
            logger.warning(f"Skipping non-WAV file: {file_path}")
            return None, None
        
        try:
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            
            if y is None or len(y) == 0:
                logger.warning(f"Empty audio loaded from {file_path}")
                return None, None
            
            # Validate and clean audio data
            if not np.isfinite(y).all():
                logger.warning(f"Audio contains non-finite values: {file_path}")
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check for silent audio
            y_abs_max = np.max(np.abs(y))
            if y_abs_max == 0:
                logger.warning(f"Audio is completely silent: {file_path}")
                return None, None
            
            # Don't normalize here - let metric calculations handle normalization
            return y, sr
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def align_audio_lengths(self, clean: np.ndarray, processed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align audio lengths by trimming to the shorter one"""
        min_len = min(len(clean), len(processed))
        if min_len == 0:
            raise ValueError("One or both audio signals are empty")
        return clean[:min_len], processed[:min_len]
    
    def calculate_pesq(self, clean: np.ndarray, processed: np.ndarray) -> Optional[float]:
        """Calculate PESQ score according to ITU-T P.862 standard"""
        try:
            clean_aligned, processed_aligned = self.align_audio_lengths(clean, processed)
            
            # PESQ requires minimum 0.25 seconds of audio
            min_samples = int(0.25 * self.target_sr)
            if len(clean_aligned) < min_samples:
                logger.debug("Audio too short for PESQ calculation")
                return None
            
            # Convert to float64 for calculations
            clean_f64 = clean_aligned.astype(np.float64)
            processed_f64 = processed_aligned.astype(np.float64)
            
            # Check for valid signals
            if not (np.isfinite(clean_f64).all() and np.isfinite(processed_f64).all()):
                logger.debug("Non-finite values in audio signals")
                return None
            
            # Standard PESQ preprocessing - normalize to prevent saturation
            # Scale to appropriate level for PESQ (around -26 dBov to -16 dBov)
            clean_rms = np.sqrt(np.mean(clean_f64 ** 2))
            processed_rms = np.sqrt(np.mean(processed_f64 ** 2))
            
            if clean_rms < 1e-8 or processed_rms < 1e-8:
                logger.debug("Signal RMS too low")
                return None
            
            # Normalize to -20 dBov (0.1 RMS for full scale)
            target_level = 0.1
            clean_scaled = clean_f64 * (target_level / clean_rms)
            processed_scaled = processed_f64 * (target_level / processed_rms)
            
            # Ensure no clipping (PESQ is sensitive to this)
            clean_peak = np.max(np.abs(clean_scaled))
            processed_peak = np.max(np.abs(processed_scaled))
            
            if clean_peak > 0.99 or processed_peak > 0.99:
                scale_factor = 0.99 / max(clean_peak, processed_peak)
                clean_scaled *= scale_factor
                processed_scaled *= scale_factor
            
            # Convert to float32 for PESQ library
            clean_final = clean_scaled.astype(np.float32)
            processed_final = processed_scaled.astype(np.float32)
            
            # Calculate PESQ (wideband mode for 16kHz)
            pesq_score = pesq(self.target_sr, clean_final, processed_final, 'wb')
            
            # Validate PESQ result (valid range: -0.5 to 4.5)
            if not np.isfinite(pesq_score):
                logger.debug("PESQ returned non-finite value")
                return None
            
            if pesq_score < -0.5 or pesq_score > 4.5:
                logger.debug(f"PESQ score out of valid range: {pesq_score}")
                return None
            
            return float(pesq_score)
            
        except Exception as e:
            logger.debug(f"PESQ calculation error: {e}")
            return None
    
    def calculate_stoi(self, clean: np.ndarray, processed: np.ndarray) -> Optional[float]:
        """Calculate STOI score according to IEEE standard"""
        try:
            clean_aligned, processed_aligned = self.align_audio_lengths(clean, processed)
            
            # STOI requires minimum 0.25 seconds
            min_samples = int(0.25 * self.target_sr)
            if len(clean_aligned) < min_samples:
                logger.debug("Audio too short for STOI calculation")
                return None
            
            # Convert to float64 for calculations
            clean_f64 = clean_aligned.astype(np.float64)
            processed_f64 = processed_aligned.astype(np.float64)
            
            # Check for valid signals
            if not (np.isfinite(clean_f64).all() and np.isfinite(processed_f64).all()):
                logger.debug("Non-finite values in audio signals")
                return None
            
            # STOI preprocessing - remove DC and normalize
            # Remove DC component
            clean_dc_removed = clean_f64 - np.mean(clean_f64)
            processed_dc_removed = processed_f64 - np.mean(processed_f64)
            
            # Check RMS levels
            clean_rms = np.sqrt(np.mean(clean_dc_removed ** 2))
            processed_rms = np.sqrt(np.mean(processed_dc_removed ** 2))
            
            if clean_rms < 1e-8 or processed_rms < 1e-8:
                logger.debug("Signal RMS too low for STOI")
                return None
            
            # Normalize RMS to same level (standard STOI preprocessing)
            target_rms = 0.1
            clean_norm = clean_dc_removed * (target_rms / clean_rms)
            processed_norm = processed_dc_removed * (target_rms / processed_rms)
            
            # Ensure signals are within [-1, 1] range
            clean_final = np.clip(clean_norm, -0.99, 0.99)
            processed_final = np.clip(processed_norm, -0.99, 0.99)
            
            # Calculate STOI (standard version, not extended)
            stoi_score = stoi(clean_final, processed_final, self.target_sr, extended=False)
            
            # Validate STOI result (valid range: 0 to 1)
            if not np.isfinite(stoi_score):
                logger.debug("STOI returned non-finite value")
                return None
            
            # Ensure within valid range
            stoi_score = float(np.clip(stoi_score, 0.0, 1.0))
            
            return stoi_score
            
        except Exception as e:
            logger.debug(f"STOI calculation error: {e}")
            return None
    
    def calculate_mse(self, clean: np.ndarray, processed: np.ndarray) -> Optional[float]:
        """Calculate normalized Mean Squared Error"""
        try:
            clean_aligned, processed_aligned = self.align_audio_lengths(clean, processed)
            
            # Convert to float64 for calculations
            clean_f64 = clean_aligned.astype(np.float64)
            processed_f64 = processed_aligned.astype(np.float64)
            
            # Check for valid signals
            if not (np.isfinite(clean_f64).all() and np.isfinite(processed_f64).all()):
                logger.debug("Non-finite values in audio signals")
                return None
            
            # Standard MSE calculation with proper normalization
            # Remove DC component
            clean_dc_removed = clean_f64 - np.mean(clean_f64)
            processed_dc_removed = processed_f64 - np.mean(processed_f64)
            
            # Check for non-zero signals
            clean_energy = np.sum(clean_dc_removed ** 2)
            processed_energy = np.sum(processed_dc_removed ** 2)
            
            if clean_energy < 1e-12 or processed_energy < 1e-12:
                logger.debug("Signal energy too low for MSE")
                return None
            
            # Normalize both signals to unit energy for fair comparison
            clean_norm = clean_dc_removed / np.sqrt(clean_energy)
            processed_norm = processed_dc_removed / np.sqrt(processed_energy)
            
            # Calculate MSE
            mse = np.mean((clean_norm - processed_norm) ** 2)
            
            # Validate result
            if not np.isfinite(mse) or mse < 0:
                logger.debug("Invalid MSE result")
                return None
            
            return float(mse)
            
        except Exception as e:
            logger.debug(f"MSE calculation error: {e}")
            return None

class PerformanceMetricsExtractor:
    """Extract performance metrics from CSV files"""
    
    @staticmethod
    def find_csv_file(audio_folder: Path) -> Optional[Path]:
        """Find CSV file in the same folder as processed audio"""
        csv_files = list(audio_folder.glob('*.csv'))
        if csv_files:
            return csv_files[0]  # Return first CSV found
        return None
    
    @staticmethod
    def extract_metrics_from_csv(csv_path: Path) -> Dict[str, float]:
        """Extract performance metrics from CSV file and calculate averages"""
        default_metrics = {
            'buffer_queue_avg': np.nan,
            'buffer_droped_avg': np.nan,
            'worklet_ms_avg': np.nan,
            'worker_ms_avg': np.nan,
            'model1_ms_avg': np.nan,
            'model2_ms_avg': np.nan
        }
        
        try:
            # Read CSV with proper handling of different separators and encodings
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='latin-1')
            
            if df.empty:
                logger.debug(f"Empty CSV file: {csv_path}")
                return default_metrics
            
            # Define flexible column mappings (handle different naming conventions)
            column_mappings = {
                # Exact matches
                'buffer_queue': 'buffer_queue_avg',
                'buffer_droped': 'buffer_droped_avg', 
                'buffer_dropped': 'buffer_droped_avg',  # Alternative spelling
                'worklet_ms': 'worklet_ms_avg',
                'worker_ms': 'worker_ms_avg',
                'model1_ms': 'model1_ms_avg',
                'model2_ms': 'model2_ms_avg',
                # Case variations
                'Buffer_Queue': 'buffer_queue_avg',
                'Buffer_Droped': 'buffer_droped_avg',
                'Buffer_Dropped': 'buffer_droped_avg',
                'Worklet_ms': 'worklet_ms_avg',
                'Worker_ms': 'worker_ms_avg',
                'Model1_ms': 'model1_ms_avg',
                'Model2_ms': 'model2_ms_avg'
            }
            
            # Calculate averages for each metric
            for csv_col, output_col in column_mappings.items():
                if csv_col in df.columns:
                    # Convert to numeric, handling various formats
                    values = pd.to_numeric(df[csv_col], errors='coerce').dropna()
                    
                    if len(values) > 0:
                        # Calculate mean and round to reasonable precision
                        avg_value = float(values.mean())
                        default_metrics[output_col] = round(avg_value, 6)
                        logger.debug(f"Calculated {output_col}: {default_metrics[output_col]:.6f} from {len(values)} values")
                    else:
                        logger.debug(f"No valid numeric values found for column: {csv_col}")
            
            return default_metrics
            
        except Exception as e:
            logger.warning(f"Error reading CSV {csv_path}: {e}")
            return default_metrics

class DatasetAnalyzer:
    """Analyze dataset and experiment folders for audio quality metrics"""
    
    def __init__(self, dataset_folder: Path, experiment_folder: Path):
        self.dataset_folder = dataset_folder
        self.experiment_folder = experiment_folder
        self.metrics_calculator = AudioMetrics()
        self.performance_extractor = PerformanceMetricsExtractor()
        self.results = []
    
    def find_clean_audio_files(self) -> Dict[str, Dict[str, Path]]:
        """Find clean WAV files in dataset folder"""
        clean_files = {}
        
        for lang_folder in self.dataset_folder.iterdir():
            if lang_folder.is_dir():
                clean_files[lang_folder.name] = {}
                
                for clean_name_folder in lang_folder.iterdir():
                    if clean_name_folder.is_dir():
                        clean_audio_file = clean_name_folder / f"{clean_name_folder.name}.wav"
                        if (clean_audio_file.exists() and 
                            self.metrics_calculator.is_wav_file(clean_audio_file)):
                            clean_files[lang_folder.name][clean_name_folder.name] = clean_audio_file
        
        return clean_files
    
    def find_processed_audio_files(self, lang: str, clean_name: str) -> Dict[str, Dict[str, Dict[str, Path]]]:
        """Find processed WAV files in experiment folder"""
        processed_files = {}
        exp_clean_folder = self.experiment_folder / lang / clean_name
        
        if not exp_clean_folder.exists():
            return processed_files
        
        for category_folder in exp_clean_folder.iterdir():
            if category_folder.is_dir():
                processed_files[category_folder.name] = {}
                
                for noise_folder in category_folder.iterdir():
                    if noise_folder.is_dir():
                        processed_files[category_folder.name][noise_folder.name] = {}
                        
                        for snr_folder in noise_folder.iterdir():
                            if snr_folder.is_dir() and snr_folder.name in SNR_LEVELS:
                                for audio_file in snr_folder.iterdir():
                                    if (audio_file.is_file() and 
                                        self.metrics_calculator.is_wav_file(audio_file)):
                                        processed_files[category_folder.name][noise_folder.name][snr_folder.name] = audio_file
                                        break
        
        return processed_files
    
    def analyze_audio_pair(self, clean_path: Path, processed_path: Path, 
                          lang: str, clean_name: str, category: str, 
                          noise_type: str, snr_level: str) -> None:
        """Analyze a pair of clean and processed audio files"""
        try:
            # Load audio files
            clean_audio, _ = self.metrics_calculator.load_audio(clean_path)
            processed_audio, _ = self.metrics_calculator.load_audio(processed_path)
            
            if clean_audio is None or processed_audio is None:
                return
            
            # Calculate audio quality metrics
            pesq_score = self.metrics_calculator.calculate_pesq(clean_audio, processed_audio)
            stoi_score = self.metrics_calculator.calculate_stoi(clean_audio, processed_audio)
            mse_score = self.metrics_calculator.calculate_mse(clean_audio, processed_audio)
            
            # Calculate clean audio duration
            clean_duration = len(clean_audio) / TARGET_SAMPLE_RATE
            
            # Extract performance metrics from CSV file in the same folder
            audio_folder = processed_path.parent
            csv_file = self.performance_extractor.find_csv_file(audio_folder)
            
            if csv_file:
                performance_metrics = self.performance_extractor.extract_metrics_from_csv(csv_file)
                logger.debug(f"Found performance CSV: {csv_file}")
            else:
                performance_metrics = {
                    'buffer_queue_avg': np.nan,
                    'buffer_droped_avg': np.nan,
                    'worklet_ms_avg': np.nan,
                    'worker_ms_avg': np.nan,
                    'model1_ms_avg': np.nan,
                    'model2_ms_avg': np.nan
                }
                logger.debug(f"No CSV found in {audio_folder}")
            
            # Store results in required format
            result = {
                'clean_audio_language': lang,
                'clean_audio_name': clean_name,
                'clean_audio_duration': clean_duration,
                'noise_category': category,
                'noise_type': noise_type,
                'snr_level': snr_level,
                'pesq_score': pesq_score,
                'stoi_score': stoi_score,
                'mse_score': mse_score,
                **performance_metrics  # Unpack performance metrics
            }
            
            self.results.append(result)
                        
        except Exception as e:
            logger.error(f"Error analyzing audio pair {clean_path} - {processed_path}: {e}")
    
    def analyze_dataset(self) -> None:
        """Analyze entire dataset"""
        clean_files = self.find_clean_audio_files()
        
        if not clean_files:
            logger.error("No clean WAV files found in dataset folder")
            return
        
        # Count total operations
        total_operations = 0
        for lang, files in clean_files.items():
            for clean_name in files.keys():
                processed_files = self.find_processed_audio_files(lang, clean_name)
                for category in processed_files.values():
                    for noise_types in category.values():
                        total_operations += len(noise_types)
        
        logger.info(f"Found {total_operations} audio pairs to analyze")
        
        # Process with progress bar
        with tqdm(total=total_operations, desc="Analyzing audio pairs") as pbar:
            for lang, files in clean_files.items():
                for clean_name, clean_path in files.items():
                    processed_files = self.find_processed_audio_files(lang, clean_name)
                    
                    for category, noise_types in processed_files.items():
                        for noise_type, snr_files in noise_types.items():
                            for snr_level, processed_path in snr_files.items():
                                self.analyze_audio_pair(
                                    clean_path, processed_path,
                                    lang, clean_name, category,
                                    noise_type, snr_level
                                )
                                
                                pbar.update(1)
                                pbar.set_postfix({
                                    'lang': lang,
                                    'category': category,
                                    'snr': snr_level
                                })
    
    def save_individual_csv_files(self, df: pd.DataFrame, base_output_dir: Path) -> None:
        """Save individual CSV files for each clean audio in organized folder structure"""
        if df.empty:
            logger.warning("No data to save individual CSV files")
            return
        
        # Create base directory structure
        base_dir = base_output_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by language and clean audio name
        grouped = df.groupby(['clean_audio_language', 'clean_audio_name'])
        
        logger.info(f"Creating individual CSV files for {len(grouped)} clean audio files...")
        
        # Define columns for individual CSV (excluding language and clean audio name)
        individual_columns = [
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
        
        for (language, clean_name), group_df in grouped:
            try:
                # Create language directory
                lang_dir = base_dir / language
                lang_dir.mkdir(exist_ok=True)
                
                # Create CSV filename
                csv_filename = f"{clean_name}.csv"
                csv_path = lang_dir / csv_filename
                
                # Select only the required columns
                individual_df = group_df[individual_columns].copy()
                
                # Sort by category, noise type, SNR level
                individual_df = individual_df.sort_values([
                    'noise_category',
                    'noise_type',
                    'snr_level'
                ])
                
                # Apply formatting (same as combined CSV)
                # Duration: 3 decimal places
                individual_df.loc[:, 'clean_audio_duration'] = individual_df['clean_audio_duration'].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )
                
                # Audio quality metrics: 6 decimal places
                audio_metrics = ['pesq_score', 'stoi_score', 'mse_score']
                for col in audio_metrics:
                    individual_df.loc[:, col] = individual_df[col].apply(
                        lambda x: f"{x:.6f}" if pd.notna(x) else ""
                    )
                
                # Performance metrics: 6 decimal places
                perf_metrics = [
                    'buffer_queue_avg', 'buffer_droped_avg', 'worklet_ms_avg',
                    'worker_ms_avg', 'model1_ms_avg', 'model2_ms_avg'
                ]
                for col in perf_metrics:
                    individual_df.loc[:, col] = individual_df[col].apply(
                        lambda x: f"{x:.6f}" if pd.notna(x) else ""
                    )
                
                # Save individual CSV file
                individual_df.to_csv(csv_path, index=False, na_rep='')
                logger.debug(f"Saved individual CSV: {csv_path}")
                
            except Exception as e:
                logger.error(f"Error saving individual CSV for {language}/{clean_name}: {e}")
        
        logger.info(f"Individual CSV files saved in: {base_dir}")
        
        # Print directory structure summary
        self._print_directory_summary(base_dir)

    def save_results_to_csv(self, output_path: Path) -> None:
        """Save analysis results to individual CSV files only"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Save individual CSV files per clean audio only
        self.save_individual_csv_files(pd.DataFrame(self.results), output_path)
        
        # Convert back to numeric for summary statistics
        self._print_summary(pd.DataFrame(self.results))

    def _print_directory_summary(self, base_dir: Path) -> None:
        """Print summary of saved CSV files and folder structure"""
        print("\n" + "="*80)
        print("SAVED CSV FILES SUMMARY")
        print("="*80)
        
        # Walk the directory structure and list saved CSV files
        for lang_folder in base_dir.iterdir():
            if lang_folder.is_dir():
                print(f"Language: {lang_folder.name}")
                
                for clean_folder in lang_folder.iterdir():
                    if clean_folder.is_dir():
                        print(f"  Clean Audio: {clean_folder.name}")
                        
                        # List CSV files in this folder
                        csv_files = list(clean_folder.glob('*_metrics.csv'))
                        for csv_file in csv_files:
                            print(f"    - {csv_file.name}")
        
        print("="*80)

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics with proper formatting"""
        print("\n" + "="*80)
        print("AUDIO ANALYSIS SUMMARY")
        print("="*80)
        
        # Basic statistics
        print(f"Total audio pairs analyzed: {len(df):,}")
        print(f"Languages: {', '.join(sorted(df['clean_audio_language'].unique()))}")
        print(f"Noise categories: {', '.join(sorted(df['noise_category'].unique()))}")
        print(f"SNR levels: {', '.join(sorted(df['snr_level'].unique()))}")
        
        # Audio duration statistics
        duration_stats = df['clean_audio_duration'].describe()
        print(f"\nAudio Duration Statistics:")
        print(f"  Mean: {duration_stats['mean']:.3f}s")
        print(f"  Min:  {duration_stats['min']:.3f}s") 
        print(f"  Max:  {duration_stats['max']:.3f}s")
        print(f"  Std:  {duration_stats['std']:.3f}s")
        
        # Audio quality metrics statistics  
        print(f"\nAudio Quality Metrics:")
        print("-" * 50)
        
        audio_metrics = [
            ('pesq_score', 'PESQ', 3),
            ('stoi_score', 'STOI', 3), 
            ('mse_score', 'MSE', 6)
        ]
        
        for col, name, decimals in audio_metrics:
            valid_scores = df[col].dropna()
            if len(valid_scores) > 0:
                print(f"  {name:4s} ({len(valid_scores):3d}/{len(df):3d} valid) - "
                      f"Mean: {valid_scores.mean():.{decimals}f}, "
                      f"Std: {valid_scores.std():.{decimals}f}, "
                      f"Min: {valid_scores.min():.{decimals}f}, "
                      f"Max: {valid_scores.max():.{decimals}f}")
            else:
                print(f"  {name:4s} - No valid scores calculated")
        
        # Performance metrics statistics
        print(f"\nPerformance Metrics:")
        print("-" * 50)
        
        perf_metrics = [
            ('buffer_queue_avg', 'Buffer Queue'),
            ('buffer_droped_avg', 'Buffer Dropped'),
            ('worklet_ms_avg', 'Worklet ms'),
            ('worker_ms_avg', 'Worker ms'),
            ('model1_ms_avg', 'Model1 ms'),
            ('model2_ms_avg', 'Model2 ms')
        ]
        
        for col, name in perf_metrics:
            valid_scores = df[col].dropna()
            if len(valid_scores) > 0:
                print(f"  {name:14s} ({len(valid_scores):3d}/{len(df):3d} valid) - "
                      f"Mean: {valid_scores.mean():8.3f}, "
                      f"Std: {valid_scores.std():8.3f}")
            else:
                print(f"  {name:14s} - No data available")
        
        print("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Analyze WAV audio quality metrics and performance data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', required=True,
                       help='Path to dataset folder containing clean WAV files')
    parser.add_argument('--experiment', required=True,
                       help='Path to experiment folder containing processed WAV files and CSV data')
    parser.add_argument('-o', '--output', default='analyses/metrics',
                       help='Output directory for individual CSV files')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate paths
    dataset_folder = Path(args.dataset).resolve()
    experiment_folder = Path(args.experiment).resolve()
    output_path = Path(args.output).resolve()
    
    if not dataset_folder.exists():
        logger.error(f"Dataset folder does not exist: {dataset_folder}")
        return 1
    
    if not experiment_folder.exists():
        logger.error(f"Experiment folder does not exist: {experiment_folder}")
        return 1
    
    print(f"Starting WAV audio analysis with performance metrics...")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Experiment folder: {experiment_folder}")
    print(f"Individual CSV files will be saved to: {output_path}")
    
    try:
        analyzer = DatasetAnalyzer(dataset_folder, experiment_folder)
        analyzer.analyze_dataset()
        analyzer.save_results_to_csv(output_path)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
