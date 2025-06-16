import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Dict
import librosa
import soundfile as sf
from tqdm import tqdm

# Constants
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
TARGET_SAMPLE_RATE = 16000
CHANNEL_IDENTIFIER = 'ch01'

# Noise category mapping
NOISE_CATEGORIES = {
    'D': 'domestic',
    'N': 'nature', 
    'O': 'office',
    'P': 'public',
    'S': 'street',
    'T': 'transportation'
}

# Create log directory and configure logging
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'formating_noise.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NoiseAudioProcessor:
    """Audio processor for noise datasets with folder renaming"""
    
    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE):
        self.target_sr = target_sr
    
    @staticmethod
    def is_audio_file(file_path: Path) -> bool:
        """Check if file is an audio file based on extension"""
        return file_path.suffix.lower() in AUDIO_EXTENSIONS
    
    @staticmethod
    def is_ch01_file(file_path: Path) -> bool:
        """Check if file contains ch01 identifier"""
        return CHANNEL_IDENTIFIER in file_path.stem.lower()
    
    @staticmethod
    def format_folder_name(original_name: str) -> tuple:
        """
        Format folder name and determine category based on first character
        Returns (category, formatted_name)
        Example: TMETRO_16k -> ('transportation', 'metro')
        """
        name = original_name
        
        # Get first character and determine category
        if len(name) > 0:
            first_char = name[0].upper()
            category = NOISE_CATEGORIES.get(first_char, 'unknown')
            # Remove first character
            name = name[1:]
        else:
            category = 'unknown'
        
        # Special case for psquare -> square (after removing first character from spsquare)
        if name.lower().startswith('psquare'):
            name = 'square' + name[7:]  # Keep anything after 'psquare'
        
        # Remove _16k suffix (case insensitive)
        if name.lower().endswith('_16k'):
            name = name[:-4]
        
        return category, name.lower()
    
    def process_audio_file(self, input_path: Path, output_path: Path) -> bool:
        """Process and copy ch01 audio file"""
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None, mono=False)
            
            # Convert to mono if needed
            if y.ndim > 1:
                y = librosa.to_mono(y)
            
            # Resample if needed
            if sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            
            # Save as WAV
            sf.write(output_path, y, self.target_sr, subtype='PCM_16')
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False

def find_noise_folders(root_folder: Path) -> Dict[Path, List[Path]]:
    """
    Find all folders containing ch01 audio files
    Returns dict with folder_path -> list of ch01 files
    """
    noise_folders = {}
    
    for folder_path in root_folder.rglob('*'):
        if folder_path.is_dir():
            ch01_files = []
            for file_path in folder_path.iterdir():
                if (file_path.is_file() and 
                    NoiseAudioProcessor.is_audio_file(file_path) and 
                    NoiseAudioProcessor.is_ch01_file(file_path)):
                    ch01_files.append(file_path)
            
            if ch01_files:
                noise_folders[folder_path] = ch01_files
    
    return noise_folders

def setup_output_folder(output_folder: Path) -> bool:
    """Create output folder and validate permissions"""
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = output_folder / '.test_write'
        test_file.touch()
        test_file.unlink()
        return True
    except Exception as e:
        logger.error(f"Cannot create or write to output folder: {e}")
        return False

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description='Process noise audio files: extract ch01 files and rename folders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True,
                       help='Root folder containing nested folders with audio files')
    parser.add_argument('-o', '--output', default='formats/noises',
                       help='Output folder for processed noise files')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate paths
    input_folder = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()
    
    if not input_folder.exists():
        logger.error(f"Input folder '{input_folder}' does not exist")
        return 1
    
    if not setup_output_folder(output_folder):
        return 1
    
    # Find noise folders
    logger.info(f"Scanning for noise folders in: {input_folder}")
    noise_folders = find_noise_folders(input_folder)
    
    if not noise_folders:
        logger.warning("No folders with ch01 audio files found")
        return 0
    
    logger.info(f"Found {len(noise_folders)} folders with ch01 files")
    
    # Process each folder
    processor = NoiseAudioProcessor()
    processed_folders = 0
    processed_files = 0
    
    for folder_path, ch01_files in tqdm(noise_folders.items(), desc="Processing folders"):
        # Format output folder name and get category
        original_folder_name = folder_path.name
        category, formatted_folder_name = NoiseAudioProcessor.format_folder_name(original_folder_name)
        
        # Create category folder first
        category_folder = output_folder / category
        category_folder.mkdir(exist_ok=True)
        
        # Create output subfolder inside category
        output_subfolder = category_folder / formatted_folder_name
        output_subfolder.mkdir(exist_ok=True)
        
        logger.info(f"Processing folder: {original_folder_name} -> {category}/{formatted_folder_name}")
        
        # Process each ch01 file in the folder
        folder_file_count = 0
        for ch01_file in ch01_files:
            # Create output filename
            output_filename = ch01_file.name
            output_path = output_subfolder / output_filename
            
            # Avoid overwriting
            counter = 1
            while output_path.exists():
                stem = ch01_file.stem
                ext = ch01_file.suffix
                output_filename = f"{stem}_{counter}{ext}"
                output_path = output_subfolder / output_filename
                counter += 1
            
            # Process the file
            if processor.process_audio_file(ch01_file, output_path):
                folder_file_count += 1
                processed_files += 1
                logger.debug(f"Processed: {ch01_file.name} -> {category}/{formatted_folder_name}/{output_filename}")
            else:
                logger.warning(f"Failed to process: {ch01_file}")
        
        if folder_file_count > 0:
            processed_folders += 1
            logger.info(f"Folder '{category}/{formatted_folder_name}': {folder_file_count} files processed")
    
    # Summary
    logger.info(f"Processing complete:")
    logger.info(f"  - Folders processed: {processed_folders}/{len(noise_folders)}")
    logger.info(f"  - Files processed: {processed_files}")
    
    return 0

if __name__ == "__main__":
    exit(main())
