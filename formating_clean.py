import os
import argparse
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
import librosa
import soundfile as sf
from tqdm import tqdm

# Constants
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
TARGET_SAMPLE_RATE = 16000
MIN_DURATION = 5.0
MAX_DURATION = 6.0

# Create log directory and configure logging
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'formating_clean.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing class with optimized methods"""
    
    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE):
        self.target_sr = target_sr
    
    @staticmethod
    def is_audio_file(file_path: Path) -> bool:
        """Check if file is an audio file based on extension"""
        return file_path.suffix.lower() in AUDIO_EXTENSIONS
    
    def validate_duration(self, duration: float) -> bool:
        """Validate if audio duration is within acceptable range"""
        return MIN_DURATION <= duration < MAX_DURATION
    
    def process_single_file(self, input_path: Path, output_folder: Path) -> Tuple[bool, str]:
        """
        Process single audio file if it meets duration requirements
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Get audio info without loading entire file first
            info = sf.info(str(input_path))
            duration = info.duration
            
            if not self.validate_duration(duration):
                return False, f"Duration {duration:.2f}s not in range {MIN_DURATION}-{MAX_DURATION}s"
            
            # Load and process audio
            y, sr = librosa.load(input_path, sr=None, mono=False)
            
            # Convert to mono if needed
            if y.ndim > 1:
                y = librosa.to_mono(y)
            
            # Resample if needed
            if sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            
            # Create output path
            output_filename = input_path.stem + '.wav'
            output_path = output_folder / output_filename
            
            # Avoid overwriting existing files
            counter = 1
            while output_path.exists():
                output_filename = f"{input_path.stem}_{counter}.wav"
                output_path = output_folder / output_filename
                counter += 1
            
            # Save as WAV with optimal settings
            sf.write(output_path, y, self.target_sr, subtype='PCM_16')
            
            return True, f"Converted: {input_path.name} -> {output_filename}"
            
        except Exception as e:
            error_msg = f"Error processing {input_path.name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

def process_file_wrapper(args: Tuple[Path, Path]) -> Tuple[bool, str]:
    """Wrapper function for multiprocessing"""
    input_path, output_folder = args
    processor = AudioProcessor()
    return processor.process_single_file(input_path, output_folder)

def find_audio_files(input_folder: Path) -> List[Path]:
    """Find all audio files in input folder and subdirectories"""
    audio_files = []
    for file_path in input_folder.rglob('*'):
        if file_path.is_file() and AudioProcessor.is_audio_file(file_path):
            audio_files.append(file_path)
    return audio_files

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
        description=f'Convert audio files ({MIN_DURATION}-{MAX_DURATION}s) to WAV mono {TARGET_SAMPLE_RATE}Hz',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True, 
                       help='Input folder containing audio files')
    parser.add_argument('-o', '--output', default='formats/cleans',
                       help='Output folder for converted WAV files')
    parser.add_argument('-j', '--jobs', type=int, default=mp.cpu_count(),
                       help='Number of parallel processes')
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
    
    # Find audio files
    logger.info(f"Scanning for audio files in: {input_folder}")
    audio_files = find_audio_files(input_folder)
    
    if not audio_files:
        logger.warning("No audio files found in input folder")
        return 0
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process files with multiprocessing
    max_workers = min(args.jobs, len(audio_files))
    converted_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file_wrapper, (audio_file, output_folder)): audio_file
            for audio_file in audio_files
        }
        
        # Process results with progress bar
        with tqdm(total=len(audio_files), desc="Processing audio files") as pbar:
            for future in as_completed(future_to_file):
                success, message = future.result()
                
                if success:
                    converted_count += 1
                    logger.debug(message)
                else:
                    failed_count += 1
                    if args.verbose:
                        logger.warning(message)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Converted': converted_count,
                    'Failed': failed_count
                })
    
    # Summary
    logger.info(f"Processing complete: {converted_count}/{len(audio_files)} files converted")
    if failed_count > 0:
        logger.warning(f"{failed_count} files failed to process")
    
    return 0

if __name__ == "__main__":
    exit(main())
