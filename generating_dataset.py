import os
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Constants
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
TARGET_SAMPLE_RATE = 16000
LANGUAGES = ['en', 'id']
SNR_LEVELS = [-5, 0, 5, 10]  # dB
MAX_CLEAN_FILES = 30  # Default value

# Noise category mapping
NOISE_CATEGORIES = {
    'domestic': 'domestic',
    'nature': 'nature', 
    'office': 'office',
    'public': 'public',
    'street': 'street',
    'transportation': 'transportation'
}

# Create log directory and configure logging
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'generating_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioMixer:
    """Audio mixing class for generating noisy datasets"""
    
    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE):
        self.target_sr = target_sr
    
    @staticmethod
    def is_audio_file(file_path: Path) -> bool:
        """Check if file is an audio file based on extension"""
        return file_path.suffix.lower() in AUDIO_EXTENSIONS
    
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to target sample rate"""
        try:
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def adjust_noise_duration(self, noise: np.ndarray, target_duration: int) -> np.ndarray:
        """Adjust noise duration to match clean audio"""
        noise_len = len(noise)
        
        if noise_len == target_duration:
            return noise
        elif noise_len > target_duration:
            # Trim noise
            start_idx = random.randint(0, noise_len - target_duration)
            return noise[start_idx:start_idx + target_duration]
        else:
            # Repeat noise to match duration
            repeats = (target_duration // noise_len) + 1
            extended_noise = np.tile(noise, repeats)
            return extended_noise[:target_duration]
    
    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) of audio signal"""
        return np.sqrt(np.mean(audio ** 2))
    
    def mix_audio_snr(self, clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """Mix clean audio with noise at specified SNR level"""
        # Adjust noise duration to match clean audio
        noise_adjusted = self.adjust_noise_duration(noise, len(clean))
        
        # Calculate RMS values
        clean_rms = self.calculate_rms(clean)
        noise_rms = self.calculate_rms(noise_adjusted)
        
        # Calculate noise scaling factor for desired SNR
        snr_linear = 10 ** (snr_db / 20)
        noise_scale = clean_rms / (noise_rms * snr_linear)
        
        # Scale noise and mix with clean audio
        scaled_noise = noise_adjusted * noise_scale
        mixed = clean + scaled_noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * (0.95 / max_val)
        
        return mixed

def get_clean_audio_files(clean_folder: Path) -> Dict[str, List[Path]]:
    """Get clean audio files organized by language"""
    clean_files = {}
    
    for lang in LANGUAGES:
        lang_folder = clean_folder / lang
        if lang_folder.exists():
            files = []
            for file_path in lang_folder.iterdir():
                if file_path.is_file() and AudioMixer.is_audio_file(file_path):
                    files.append(file_path)
            
            # Sort files by name
            files.sort(key=lambda x: x.name)
            clean_files[lang] = files
            logger.info(f"Found {len(files)} clean audio files for language: {lang}")
    
    return clean_files

def get_noise_files(noise_folder: Path) -> Dict[str, Dict[str, Path]]:
    """Get noise files organized by category and noise type"""
    noise_files = {}
    
    # Only check for categorized structure
    for category_dir in noise_folder.iterdir():
        if category_dir.is_dir() and category_dir.name.lower() in NOISE_CATEGORIES:
            category = category_dir.name.lower()
            noise_files[category] = {}
            
            # Find noise files in category subfolder
            for noise_dir in category_dir.iterdir():
                if noise_dir.is_dir():
                    # Find audio files in noise directory
                    for file_path in noise_dir.iterdir():
                        if (file_path.is_file() and 
                            AudioMixer.is_audio_file(file_path)):
                            noise_type = noise_dir.name.lower()
                            noise_files[category][noise_type] = file_path
                            break
    
    # Log found noise files
    total_noise_types = sum(len(types) for types in noise_files.values())
    logger.info(f"Found {total_noise_types} noise types in {len(noise_files)} categories")
    for category, types in noise_files.items():
        logger.info(f"  {category}: {list(types.keys())}")
    
    return noise_files

def process_language(clean_files: List[Path], noise_files: Dict[str, Dict[str, Path]], 
                    output_folder: Path, lang: str, max_files: int = MAX_CLEAN_FILES) -> None:
    """Process all versions for a specific language"""
    mixer = AudioMixer()
    
    # Load all noise files by category
    loaded_noises = {}
    for category, noise_types in noise_files.items():
        loaded_noises[category] = {}
        for noise_type, noise_path in noise_types.items():
            noise_audio, _ = mixer.load_audio(noise_path)
            if noise_audio is not None:
                loaded_noises[category][noise_type] = noise_audio
    
    logger.info(f"Processing language: {lang}")
    
    # Create language folder
    lang_folder = output_folder / lang
    lang_folder.mkdir(parents=True, exist_ok=True)
    
    # Process first N clean audio files based on max_files parameter
    selected_clean = clean_files[:min(max_files, len(clean_files))]
    
    for clean_path in tqdm(selected_clean, desc=f"Processing {lang} files"):
        # Load clean audio
        clean_audio, _ = mixer.load_audio(clean_path)
        if clean_audio is None:
            continue
        
        # Create folder named after clean audio file (without extension)
        clean_folder = lang_folder / clean_path.stem
        clean_folder.mkdir(exist_ok=True)
        
        # Save clean audio in its own folder
        clean_output = clean_folder / clean_path.name
        sf.write(clean_output, clean_audio, TARGET_SAMPLE_RATE, subtype='PCM_16')
        
        # Process each noise category
        for category, noise_types in loaded_noises.items():
            # Create category folder
            category_folder = clean_folder / category
            category_folder.mkdir(exist_ok=True)
            
            # Process each noise type in category
            for noise_type, noise_audio in noise_types.items():
                # Create noise scenario folder
                noise_scenario_folder = category_folder / noise_type
                noise_scenario_folder.mkdir(exist_ok=True)
                
                # Adjust noise duration to match clean audio
                adjusted_noise = mixer.adjust_noise_duration(noise_audio, len(clean_audio))
                
                # Save adjusted noise as ch01.wav
                noise_output = noise_scenario_folder / "ch01.wav"
                sf.write(noise_output, adjusted_noise, TARGET_SAMPLE_RATE, subtype='PCM_16')
                
                # Generate mixed audio at different SNR levels
                for snr in SNR_LEVELS:
                    mixed_audio = mixer.mix_audio_snr(clean_audio, noise_audio, snr)
                    snr_filename = f"{snr} dB.wav"
                    snr_output = noise_scenario_folder / snr_filename
                    sf.write(snr_output, mixed_audio, TARGET_SAMPLE_RATE, subtype='PCM_16')

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description='Generate mixed audio dataset with clean, noise, and SNR variations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--clean', required=True,
                       help='Clean audio folder containing en/ and id/ subfolders')
    parser.add_argument('--noise', required=True,
                       help='Noise folder containing various noise types')
    parser.add_argument('-o', '--output', default='generates/datasets',
                       help='Output dataset folder')
    parser.add_argument('--max', type=int, default=MAX_CLEAN_FILES,
                       help=f'Maximum number of clean files to process per language (default: {MAX_CLEAN_FILES})')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate paths
    clean_folder = Path(args.clean).resolve()
    noise_folder = Path(args.noise).resolve()
    output_folder = Path(args.output).resolve()
    
    if not clean_folder.exists():
        logger.error(f"Clean folder '{clean_folder}' does not exist")
        return 1
    
    if not noise_folder.exists():
        logger.error(f"Noise folder '{noise_folder}' does not exist")
        return 1
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get clean audio files
    clean_files = get_clean_audio_files(clean_folder)
    if not clean_files:
        logger.error("No clean audio files found")
        return 1
    
    # Get noise files
    noise_files = get_noise_files(noise_folder)
    if not noise_files:
        logger.error("No noise files found")
        return 1
    
    # Process each language
    for lang, files in clean_files.items():
        if len(files) < args.max:
            logger.warning(f"Language {lang}: only {len(files)} files available, need {args.max}")
        
        process_language(files, noise_files, output_folder, lang, args.max)
    
    logger.info("Dataset generation complete!")
    return 0

if __name__ == "__main__":
    exit(main())
