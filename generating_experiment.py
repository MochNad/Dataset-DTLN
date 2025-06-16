import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Constants
SNR_LEVELS = ['-5 dB', '0 dB', '5 dB', '10 dB']

# Create log directory and configure logging
log_dir = Path('log')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'generating_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentStructureBuilder:
    """Build experiment folder structure with SNR subfolders but no audio files"""
    
    def __init__(self, output_folder: Path):
        self.output_folder = output_folder
    
    def create_snr_folder_structure(self, noise_folder_path: Path) -> None:
        """Create empty SNR folders for experiment structure"""
        noise_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create all SNR level folders (empty)
        for snr_level in SNR_LEVELS:
            snr_folder = noise_folder_path / snr_level
            snr_folder.mkdir(exist_ok=True)
            logger.debug(f"Created SNR folder: {snr_folder}")
    
    def build_structure_from_dataset(self, dataset_folder: Path) -> None:
        """Build experiment folder structure based on dataset folder structure"""
        if not dataset_folder.exists():
            raise ValueError(f"Dataset folder does not exist: {dataset_folder}")
        
        # Count total operations for progress tracking
        total_operations = self._count_noise_folders(dataset_folder)
        logger.info(f"Creating structure for {total_operations} noise scenarios")
        
        # Process with progress bar
        with tqdm(total=total_operations, desc="Creating experiment structure") as pbar:
            self._process_dataset_structure(dataset_folder, pbar)
    
    def _count_noise_folders(self, dataset_folder: Path) -> int:
        """Count total number of noise folders to process"""
        count = 0
        for lang_folder in dataset_folder.iterdir():
            if not lang_folder.is_dir():
                continue
            for clean_folder in lang_folder.iterdir():
                if not clean_folder.is_dir():
                    continue
                for category_folder in clean_folder.iterdir():
                    if not category_folder.is_dir():
                        continue
                    for noise_folder in category_folder.iterdir():
                        if noise_folder.is_dir():
                            count += 1
        return count
    
    def _process_dataset_structure(self, dataset_folder: Path, pbar: tqdm) -> None:
        """Process dataset structure and create corresponding experiment structure"""
        for lang_folder in dataset_folder.iterdir():
            if not lang_folder.is_dir():
                continue
            
            output_lang_folder = self.output_folder / lang_folder.name
            
            for clean_folder in lang_folder.iterdir():
                if not clean_folder.is_dir():
                    continue
                
                output_clean_folder = output_lang_folder / clean_folder.name
                
                for category_folder in clean_folder.iterdir():
                    if not category_folder.is_dir():
                        continue
                    
                    output_category_folder = output_clean_folder / category_folder.name
                    
                    for noise_folder in category_folder.iterdir():
                        if not noise_folder.is_dir():
                            continue
                        
                        output_noise_folder = output_category_folder / noise_folder.name
                        
                        # Create SNR folder structure (no files)
                        self.create_snr_folder_structure(output_noise_folder)
                        
                        pbar.update(1)
                        pbar.set_postfix(
                            lang=lang_folder.name,
                            clean=clean_folder.name[:10] + "..." if len(clean_folder.name) > 10 else clean_folder.name,
                            category=category_folder.name,
                            noise=noise_folder.name[:8] + "..." if len(noise_folder.name) > 8 else noise_folder.name
                        )

def validate_dataset_structure(dataset_folder: Path) -> bool:
    """Validate that the dataset has the expected structure"""
    if not dataset_folder.exists():
        logger.error(f"Dataset folder does not exist: {dataset_folder}")
        return False
    
    # Check for language folders
    lang_folders = [f for f in dataset_folder.iterdir() if f.is_dir()]
    if not lang_folders:
        logger.error("No language folders found in dataset")
        return False
    
    # Validate structure depth
    structure_valid = False
    for lang_folder in lang_folders:
        for clean_folder in lang_folder.iterdir():
            if clean_folder.is_dir():
                for category_folder in clean_folder.iterdir():
                    if category_folder.is_dir():
                        for noise_folder in category_folder.iterdir():
                            if noise_folder.is_dir():
                                structure_valid = True
                                break
                        if structure_valid:
                            break
                if structure_valid:
                    break
        if structure_valid:
            break
    
    if not structure_valid:
        logger.error("Dataset structure validation failed - expected lang/clean/category/noise hierarchy")
        return False
    
    logger.info(f"Dataset structure validation passed: {len(lang_folders)} language folders found")
    return True

def clean_existing_structure(output_folder: Path, force: bool = False) -> None:
    """Clean existing experiment structure if needed"""
    if output_folder.exists():
        if force:
            import shutil
            logger.warning(f"Removing existing experiment folder: {output_folder}")
            shutil.rmtree(output_folder)
        else:
            logger.info(f"Experiment folder exists: {output_folder} (will update structure)")

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description='Create experiment folder structure with SNR subfolders (no audio files)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True,
                       help='Input dataset folder path')
    parser.add_argument('-o', '--output', default='generates/experiments',
                       help='Output experiment folder path')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--force', action='store_true',
                       help='Remove existing experiment folder before creating new structure')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually creating folders')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Validate and resolve paths
    dataset_folder = Path(args.input).resolve()
    output_folder = Path(args.output).resolve()
    
    logger.info(f"Input dataset: {dataset_folder}")
    logger.info(f"Output experiment: {output_folder}")
    
    # Validate dataset structure
    if not validate_dataset_structure(dataset_folder):
        logger.error("Dataset structure validation failed")
        return 1
    
    # Handle dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - No folders will be created")
        logger.info(f"Would create experiment structure based on: {dataset_folder}")
        logger.info(f"Target output folder: {output_folder}")
        logger.info(f"SNR levels to create: {SNR_LEVELS}")
        return 0
    
    # Clean existing structure if requested
    clean_existing_structure(output_folder, args.force)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        # Build experiment structure
        builder = ExperimentStructureBuilder(output_folder)
        builder.build_structure_from_dataset(dataset_folder)
        
        logger.info("Experiment structure creation completed successfully!")
        logger.info(f"Created folder structure with SNR levels: {SNR_LEVELS}")
        logger.info(f"No audio files were copied - structure only")
        
        # Summary statistics
        total_snr_folders = sum(1 for _ in output_folder.rglob('*') if _.is_dir() and _.name in SNR_LEVELS)
        logger.info(f"Total SNR folders created: {total_snr_folders}")
        
    except Exception as e:
        logger.error(f"Error during structure creation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
