import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Constants
AUDIO_EXTENSIONS = {'.wav'}
TARGET_SAMPLE_RATE = 16000
SNR_LEVELS = ['-5 dB', '0 dB', '5 dB', '10 dB']
FIGURE_SIZE = (32, 8)  # Changed for single row layout
DPI = 100

# Create log directory and configure logging
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'analyzing_visual.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioVisualizer:
    """Audio visualization class for generating spectrograms"""
    
    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE):
        self.target_sr = target_sr
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
    
    @staticmethod
    def is_audio_file(file_path: Path) -> bool:
        """Check if file is an audio file based on extension"""
        return file_path.suffix.lower() in AUDIO_EXTENSIONS
    
    def load_audio(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file and convert to target sample rate"""
        try:
            if not file_path.exists():
                logger.warning(f"Audio file does not exist: {file_path}")
                return None, None
            
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def create_spectrogram(self, y: np.ndarray, sr: int, title: str, ax: plt.Axes) -> None:
        """Create mel spectrogram visualization"""
        try:
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                n_mels=self.n_mels
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Create spectrogram plot
            img = librosa.display.specshow(
                S_dB, sr=sr, 
                hop_length=self.hop_length,
                x_axis='time', 
                y_axis='mel',
                ax=ax,
                cmap='plasma'  # Changed to plasma colormap
            )
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Mel Frequency', fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
            cbar.set_label('Power (dB)', fontsize=10)
            
        except Exception as e:
            logger.error(f"Error creating spectrogram for {title}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{title} (Error)", fontsize=12, color='red')
    
    def create_empty_spectrogram(self, title: str, ax: plt.Axes, reason: str = "File not found") -> None:
        """Create empty spectrogram placeholder"""
        ax.text(0.5, 0.5, f'{reason}', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=14, color='gray')
        ax.set_title(f"{title} (Not Available)", fontsize=12, color='gray')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Mel Frequency', fontsize=10)

class AudioPathFinder:
    """Find audio files in dataset and experiment structures"""
    
    def __init__(self, dataset_folder: Path, experiment_folder: Path):
        self.dataset_folder = dataset_folder
        self.experiment_folder = experiment_folder
    
    def find_clean_audio(self, lang: str, clean_name: str) -> Optional[Path]:
        """Find clean audio file in dataset"""
        clean_folder = self.dataset_folder / lang / clean_name
        if clean_folder.exists():
            for file_path in clean_folder.iterdir():
                if AudioVisualizer.is_audio_file(file_path):
                    return file_path
        return None
    
    def find_noise_audio(self, lang: str, clean_name: str, category: str, noise_type: str) -> Optional[Path]:
        """Find noise audio file (ch01.wav) in dataset"""
        noise_folder = self.dataset_folder / lang / clean_name / category / noise_type
        noise_file = noise_folder / "ch01.wav"
        return noise_file if noise_file.exists() else None
    
    def find_snr_audio(self, lang: str, clean_name: str, category: str, noise_type: str, snr_level: str) -> Optional[Path]:
        """Find SNR audio file in dataset"""
        snr_folder = self.dataset_folder / lang / clean_name / category / noise_type
        snr_file = snr_folder / f"{snr_level}.wav"
        return snr_file if snr_file.exists() else None
    
    def find_processed_audio(self, lang: str, clean_name: str, category: str, noise_type: str, snr_level: str) -> Optional[Path]:
        """Find processed audio file in experiment"""
        processed_folder = self.experiment_folder / lang / clean_name / category / noise_type / snr_level
        if processed_folder.exists():
            for file_path in processed_folder.iterdir():
                if AudioVisualizer.is_audio_file(file_path):
                    return file_path
        return None

class VisualizationGenerator:
    """Generate visualization plots for audio analyses"""
    
    def __init__(self, dataset_folder: Path, experiment_folder: Path, output_folder: Path):
        self.dataset_folder = dataset_folder
        self.experiment_folder = experiment_folder
        self.output_folder = output_folder
        self.visualizer = AudioVisualizer()
        self.path_finder = AudioPathFinder(dataset_folder, experiment_folder)
    
    def create_comparison_plot(self, lang: str, clean_name: str, category: str, 
                             noise_type: str, snr_level: str, output_path: Path) -> bool:
        """Create 4-panel comparison plot"""
        try:
            fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE)  # Changed to single row
            fig.suptitle(f'Audio Analyses: {clean_name} + {noise_type} ({snr_level})', 
                        fontsize=16, fontweight='bold')
            
            # Find audio files
            clean_path = self.path_finder.find_clean_audio(lang, clean_name)
            noise_path = self.path_finder.find_noise_audio(lang, clean_name, category, noise_type)
            snr_path = self.path_finder.find_snr_audio(lang, clean_name, category, noise_type, snr_level)
            processed_path = self.path_finder.find_processed_audio(lang, clean_name, category, noise_type, snr_level)
            
            # Load and plot clean audio
            if clean_path:
                clean_audio, clean_sr = self.visualizer.load_audio(clean_path)
                if clean_audio is not None:
                    self.visualizer.create_spectrogram(
                        clean_audio, clean_sr, 
                        f"Clean Audio: {clean_path.name}", 
                        axes[0]  # Changed indexing for single row
                    )
                else:
                    self.visualizer.create_empty_spectrogram(
                        f"Clean Audio: {clean_name}", axes[0], "Load Error"
                    )
            else:
                self.visualizer.create_empty_spectrogram(
                    f"Clean Audio: {clean_name}", axes[0]
                )
            
            # Load and plot noise audio
            if noise_path:
                noise_audio, noise_sr = self.visualizer.load_audio(noise_path)
                if noise_audio is not None:
                    self.visualizer.create_spectrogram(
                        noise_audio, noise_sr, 
                        f"Noise Audio: {noise_type}", 
                        axes[1]  # Changed indexing for single row
                    )
                else:
                    self.visualizer.create_empty_spectrogram(
                        f"Noise Audio: {noise_type}", axes[1], "Load Error"
                    )
            else:
                self.visualizer.create_empty_spectrogram(
                    f"Noise Audio: {noise_type}", axes[1]
                )
            
            # Load and plot SNR audio
            if snr_path:
                snr_audio, snr_sr = self.visualizer.load_audio(snr_path)
                if snr_audio is not None:
                    self.visualizer.create_spectrogram(
                        snr_audio, snr_sr, 
                        f"Mixed Audio: {snr_level}", 
                        axes[2]  # Changed indexing for single row
                    )
                else:
                    self.visualizer.create_empty_spectrogram(
                        f"Mixed Audio: {snr_level}", axes[2], "Load Error"
                    )
            else:
                self.visualizer.create_empty_spectrogram(
                    f"Mixed Audio: {snr_level}", axes[2]
                )
            
            # Load and plot processed audio
            if processed_path:
                processed_audio, processed_sr = self.visualizer.load_audio(processed_path)
                if processed_audio is not None:
                    self.visualizer.create_spectrogram(
                        processed_audio, processed_sr, 
                        f"Processed Audio: {processed_path.name}", 
                        axes[3]  # Changed indexing for single row
                    )
                else:
                    self.visualizer.create_empty_spectrogram(
                        f"Processed Audio", axes[3], "Load Error"
                    )
            else:
                self.visualizer.create_empty_spectrogram(
                    "Processed Audio", axes[3]
                )
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")
            plt.close('all')
            return False
    
    def generate_all_visualizations(self):
        """Generate all visualization plots based on experiment structure"""
        if not self.experiment_folder.exists():
            raise ValueError(f"Experiment folder does not exist: {self.experiment_folder}")
        
        # Count total operations
        total_ops = self._count_visualizations()
        logger.info(f"Generating {total_ops} visualization plots")
        
        # Create base analyses directory structure
        base_output_folder = Path('analyses') / 'visuals'
        base_output_folder.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        current_file = 1
        
        with tqdm(total=total_ops, desc="Generating Spectrograms", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                 ncols=120) as pbar:
            
            for lang_folder in self.experiment_folder.iterdir():
                if not lang_folder.is_dir():
                    continue
                
                lang = lang_folder.name
                # Create language folder in analyses/visuals/
                output_lang_folder = base_output_folder / lang
                output_lang_folder.mkdir(parents=True, exist_ok=True)
                
                for clean_folder in lang_folder.iterdir():
                    if not clean_folder.is_dir():
                        continue
                    
                    clean_name = clean_folder.name
                    # Create clean name folder
                    output_clean_folder = output_lang_folder / clean_name
                    output_clean_folder.mkdir(parents=True, exist_ok=True)
                    
                    for category_folder in clean_folder.iterdir():
                        if not category_folder.is_dir():
                            continue
                        
                        category = category_folder.name
                        # Create category folder with "_noise" suffix
                        category_noise_name = f"{category}"
                        output_category_folder = output_clean_folder / category_noise_name
                        output_category_folder.mkdir(parents=True, exist_ok=True)
                        
                        for noise_folder in category_folder.iterdir():
                            if not noise_folder.is_dir():
                                continue
                            
                            noise_type = noise_folder.name
                            # Create noise type folder
                            output_noise_folder = output_category_folder / noise_type
                            output_noise_folder.mkdir(parents=True, exist_ok=True)
                            
                            for snr_folder in noise_folder.iterdir():
                                if not snr_folder.is_dir() or snr_folder.name not in SNR_LEVELS:
                                    continue
                                
                                snr_level = snr_folder.name
                                
                                # Generate visualization with new file path structure
                                # analyses/visuals/{language}/{clean_name}/{category}/{type_noise}/{snr_level}.png
                                output_file = output_noise_folder / f"{snr_level}.png"
                                
                                # Fixed progress description - ensure noise type is shown
                                progress_desc = f"[{current_file:4d}/{total_ops}] {lang.upper()}|{clean_name[:10]}|{category[:6]}|{noise_type[:10]}|{snr_level}"
                                pbar.set_description(progress_desc)
                                
                                if self.create_comparison_plot(
                                    lang, clean_name, category, noise_type, snr_level, output_file
                                ):
                                    success_count += 1
                                    pbar.set_postfix_str(f"✓ {success_count}/{current_file}")
                                else:
                                    pbar.set_postfix_str(f"✗ Failed: {current_file - success_count}")
                                
                                pbar.update(1)
                                current_file += 1
        
        # Final summary
        failure_count = total_ops - success_count
        success_rate = (success_count / total_ops * 100) if total_ops > 0 else 0
        
        logger.info(f"Visualization generation completed!")
        logger.info(f"Success: {success_count}/{total_ops} ({success_rate:.1f}%)")
        logger.info(f"Visualizations saved to: {base_output_folder.resolve()}")
        if failure_count > 0:
            logger.warning(f"Failures: {failure_count} visualizations could not be generated")
        
        # Print directory structure summary
        self._print_directory_summary(base_output_folder)

    def _print_directory_summary(self, base_dir: Path) -> None:
        """Print summary of created directory structure"""
        print(f"\nVisualization Files Structure:")
        print("-" * 50)
        
        total_files = 0
        for lang_dir in sorted(base_dir.iterdir()):
            if lang_dir.is_dir():
                print(f"  {lang_dir.name}/")
                lang_files = 0
                
                for clean_dir in sorted(lang_dir.iterdir()):
                    if clean_dir.is_dir():
                        clean_files = sum(1 for f in clean_dir.rglob('*.png'))
                        lang_files += clean_files
                        print(f"    ├── {clean_dir.name}/ ({clean_files} visualizations)")
                        
                        # Show category structure for first few items
                        categories = list(clean_dir.iterdir())[:2]  # Show first 2 categories
                        for i, cat_dir in enumerate(categories):
                            if cat_dir.is_dir():
                                cat_files = sum(1 for f in cat_dir.rglob('*.png'))
                                prefix = "│   ├──" if i < len(categories) - 1 else "│   └──"
                                print(f"    {prefix} {cat_dir.name}/ ({cat_files} files)")
                
                total_files += lang_files
                print(f"    └── Total: {lang_files} files\n")
        
        print(f"Total visualization files created: {total_files}")
        print(f"Base directory: {base_dir.resolve()}")

    def _count_visualizations(self) -> int:
        """Count total number of visualizations to generate"""
        count = 0
        for lang_folder in self.experiment_folder.iterdir():
            if not lang_folder.is_dir():
                continue
            for clean_folder in lang_folder.iterdir():
                if not clean_folder.is_dir():
                    continue
                for category_folder in clean_folder.iterdir():
                    if not category_folder.is_dir():
                        continue
                    for noise_folder in category_folder.iterdir():
                        if not noise_folder.is_dir():
                            continue
                        for snr_folder in noise_folder.iterdir():
                            if snr_folder.is_dir() and snr_folder.name in SNR_LEVELS:
                                count += 1
        return count

def validate_folders(dataset_folder: Path, experiment_folder: Path) -> bool:
    """Validate input folders exist and have proper structure"""
    if not dataset_folder.exists():
        logger.error(f"Dataset folder does not exist: {dataset_folder}")
        return False
    
    if not experiment_folder.exists():
        logger.error(f"Experiment folder does not exist: {experiment_folder}")
        return False
    
    # Check for language folders in both
    dataset_langs = [f.name for f in dataset_folder.iterdir() if f.is_dir()]
    experiment_langs = [f.name for f in experiment_folder.iterdir() if f.is_dir()]
    
    if not dataset_langs:
        logger.error("No language folders found in dataset")
        return False
    
    if not experiment_langs:
        logger.error("No language folders found in experiment")
        return False
    
    logger.info(f"Dataset languages: {dataset_langs}")
    logger.info(f"Experiment languages: {experiment_langs}")
    return True

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description='Generate visual analyses with spectrograms comparing dataset and experiment audio',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', required=True,
                       help='Dataset folder path containing original audio files')
    parser.add_argument('--experiment', required=True,
                       help='Experiment folder path containing processed audio files')
    parser.add_argument('-o', '--output', default='analyses/visuals',
                       help='Output folder for visualization plots')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dpi', type=int, default=DPI,
                       help=f'DPI for output images (default: {DPI})')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate and resolve paths
    dataset_folder = Path(args.dataset).resolve()
    experiment_folder = Path(args.experiment).resolve()
    output_folder = Path(args.output).resolve()
    
    logger.info(f"Dataset folder: {dataset_folder}")
    logger.info(f"Experiment folder: {experiment_folder}")
    logger.info(f"Output folder: {output_folder}")
    
    # Validate input folders
    if not validate_folders(dataset_folder, experiment_folder):
        return 1
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate visualizations
        generator = VisualizationGenerator(dataset_folder, experiment_folder, output_folder)
        generator.generate_all_visualizations()
        
        logger.info("Visual analyses generation completed successfully!")
        logger.info(f"Visualization plots saved to: {output_folder}")
        
    except Exception as e:
        logger.error(f"Error during visualization generation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
