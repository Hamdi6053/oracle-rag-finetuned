#!/usr/bin/env python3
"""
PDF to Markdown Converter using Marker
=====================================

This script converts PDF files to markdown format using the Marker library,
specifically designed for RAG (Retrieval Augmented Generation) applications.

Features:
- Properly formatted header levels
- Accurate table extraction
- Image extraction and local storage
- Comprehensive metadata generation in JSON format
- Batch processing of multiple PDFs
- Organized output structure

Requirements:
- marker-pdf library
- Python 3.9+
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import uuid

def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def create_output_directories(base_output_dir: str) -> Dict[str, str]:
    """Create organized output directory structure."""
    base_path = Path(base_output_dir)
    
    directories = {
        'base': str(base_path),
        'markdown': str(base_path / 'markdown'),
        'images': str(base_path / 'images'),
        'metadata': str(base_path / 'metadata'),
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

class PDFToMarkdownConverter:
    """
    Enhanced PDF to Markdown converter using Marker with comprehensive features.
    """
    
    def __init__(self, output_dirs: Dict[str, str], logger: logging.Logger, force_ocr: bool = False):
        self.output_dirs = output_dirs
        self.logger = logger
        self.conversion_stats = {
            'total_files': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_pages': 0,
            'total_images': 0,
            'total_tables': 0
        }
        
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            self.logger.info("Loading Marker models...")
            self.model_artifacts = create_model_dict()
            self.converter = PdfConverter(artifact_dict=self.model_artifacts)
            self.text_from_rendered = text_from_rendered
            self.logger.info("Marker models loaded successfully.")
        except ImportError:
            self.logger.error("Failed to import Marker components. Your marker-pdf version might be incompatible.")
            self.logger.error("Please try updating to the latest version from GitHub:")
            self.logger.error("pip install --upgrade git+https://github.com/VikParuchuri/marker.git")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Marker: {e}")
            raise
    
    def extract_title_from_context(self, markdown_content: str, image_position: int) -> str:
        """Extract the previous closest title/header before the image position in the markdown."""
        lines = markdown_content.split('\n')
        title = "Untitled"
        found_headers = []
        
        # Look backwards from image position to find headers
        for i in range(min(image_position, len(lines) - 1), -1, -1):
            line = lines[i].strip()
            if line.startswith('#'):
                # Extract header text, prioritize higher level headers
                if line.startswith('# '):
                    found_headers.append(line.strip('# '))
                elif line.startswith('## '):
                    found_headers.append(line.strip('## '))
                elif line.startswith('### '):
                    found_headers.append(line.strip('### '))
        
        # Return the second closest header if available, otherwise the closest one
        if len(found_headers) >= 2:
            title = found_headers[1]  # Previous closest (second in the list)
        elif len(found_headers) == 1:
            title = found_headers[0]  # Only one header found
        
        return title

    def process_images(self, images: Dict, pdf_name: str, markdown_content: str, rendered_data) -> Dict[str, Dict]:
        """Process and save extracted images with individual metadata for each image."""
        image_metadata_list = {}
        
        if not images:
            self.logger.info(f"No images found in {pdf_name}")
            return image_metadata_list
        
        pdf_image_dir = Path(self.output_dirs['images']) / pdf_name
        pdf_image_dir.mkdir(exist_ok=True)
        
        # Try to get page information from rendered data
        page_info = {}
        if hasattr(rendered_data, 'children'):
            for page_idx, page in enumerate(rendered_data.children):
                if hasattr(page, 'images'):
                    for img_id in page.images.keys():
                        page_info[img_id] = page_idx + 1
        
        for idx, (image_id, image_data) in enumerate(images.items()):
            try:
                image_filename = f"{pdf_name}_image_{idx+1}.png"
                image_path = pdf_image_dir / image_filename
                
                # Save the image temporarily to check size
                image_data.save(image_path)
                
                # Check image size to filter out small footer/header images
                file_size_kb = image_path.stat().st_size / 1024
                
                # Skip small images (likely footer/header logos) - adjust threshold as needed
                if file_size_kb < 10:  # Less than 10KB
                    self.logger.debug(f"Skipping small image: {image_filename} ({file_size_kb:.1f}KB)")
                    image_path.unlink()  # Delete the small image
                    continue
                
                # Get page number (fallback to estimated position)
                page_number = page_info.get(image_id, idx + 1)
                
                # Find the position of this image in markdown (approximate)
                image_position_in_markdown = idx * 50  # Rough estimate
                
                # Extract title from context (previous closest)
                title = self.extract_title_from_context(markdown_content, image_position_in_markdown)
                
                # Create simplified metadata for this image
                image_metadata = {
                    'id': str(uuid.uuid4()),
                    'image_name': str(Path('images') / pdf_name / image_filename),
                    'page_number': page_number,
                    'title': title
                }
                
                image_metadata_list[image_id] = image_metadata
                
                self.logger.debug(f"Saved image: {image_filename} with metadata")
                
            except Exception as e:
                self.logger.error(f"Failed to save image {image_id}: {e}")
        
        # Save individual image metadata files
        for image_id, img_meta in image_metadata_list.items():
            image_meta_filename = f"{img_meta['image_name'].split('/')[-1][:-4]}_metadata.json"
            image_meta_path = pdf_image_dir / image_meta_filename
            
            with open(image_meta_path, 'w', encoding='utf-8') as f:
                json.dump(img_meta, f, indent=2, ensure_ascii=False)
        
        self.conversion_stats['total_images'] += len(image_metadata_list)
        return image_metadata_list
    
    def enhance_markdown_headers(self, markdown_content: str) -> str:
        """Enhance markdown headers for better RAG chunking."""
        lines = markdown_content.split('\n')
        enhanced_lines = []
        for line in lines:
            if line.strip().startswith('#'):
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.strip('#').strip()
                enhanced_header = '#' * header_level + ' ' + header_text
                enhanced_lines.append(enhanced_header)
            else:
                enhanced_lines.append(line)
        return '\n'.join(enhanced_lines)

    def extract_metadata_from_content(self, markdown_content: str, rendered_data, image_info: Dict) -> Dict:
        """Extract comprehensive metadata from the converted content."""
        base_meta = rendered_data.metadata
        
        # Find first H1 for title
        title = "Untitled"
        for line in markdown_content.split('\n'):
            if line.startswith('# '):
                title = line.strip('# ')
                break

        metadata = {
            'id': str(uuid.uuid4()),
            'title': title,
            'source_pages': base_meta.get("pages"),
            'conversion_timestamp': datetime.now().isoformat(),
            'images': image_info
        }
        return metadata
    
    def convert_pdf(self, pdf_path: str) -> Tuple[bool, Dict]:
        """Convert a single PDF file to markdown with comprehensive processing."""
        pdf_file = Path(pdf_path)
        pdf_name = pdf_file.stem
        self.logger.info(f"Starting conversion of: {pdf_file.name}")
        
        try:
            self.logger.info("Running marker conversion...")
            rendered_result = self.converter(str(pdf_path))
            markdown_content, _, images = self.text_from_rendered(rendered_result)
            
            enhanced_markdown = self.enhance_markdown_headers(markdown_content)
            image_info = self.process_images(images, pdf_name, markdown_content, rendered_result)
            
            metadata = self.extract_metadata_from_content(enhanced_markdown, rendered_result, image_info)
            metadata['source_file'] = str(pdf_file)
            metadata['output_files'] = {
                'markdown': str(Path('markdown') / f"{pdf_name}.md"),
                'metadata': str(Path('metadata') / f"{pdf_name}_metadata.json"),
                'images_directory': str(Path('images') / pdf_name) if image_info else None
            }
            metadata['images'] = image_info
            
            markdown_path = Path(self.output_dirs['markdown']) / f"{pdf_name}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_markdown)
            
            metadata_path = Path(self.output_dirs['metadata']) / f"{pdf_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.conversion_stats['successful_conversions'] += 1
            self.conversion_stats['total_pages'] += (metadata.get('source_pages') or 0)
            self.conversion_stats['total_images'] += len(image_info)
            
            self.logger.info(f"✅ Successfully converted {pdf_file.name}")
            return True, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert {pdf_file.name}: {e}", exc_info=True)
            self.conversion_stats['failed_conversions'] += 1
            return False, {'error': str(e), 'source_file': str(pdf_file)}
    
    def convert_multiple_pdfs(self, pdf_paths: List[str]) -> Dict:
        """Convert multiple PDF files."""
        self.conversion_stats['total_files'] = len(pdf_paths)
        results = {}
        
        for pdf_path in pdf_paths:
            success, metadata = self.convert_pdf(pdf_path)
            results[pdf_path] = {'success': success, 'metadata': metadata}
        
        batch_results_path = Path(self.output_dirs['metadata']) / 'batch_conversion_results.json'
        with open(batch_results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'conversion_stats': self.conversion_stats,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        return results

def find_pdf_files(input_path: str) -> List[str]:
    """Find all PDF files in the given path."""
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == '.pdf':
        return [str(path)]
    elif path.is_dir():
        return sorted([str(p) for p in path.rglob('*.pdf')])
    return []

def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to markdown using Marker for RAG applications.",
        epilog="Examples:\n"
               "  python run_marker.py -i file.pdf\n"
               "  python run_marker.py -i ./docs/ -o ./output --force-ocr"
    )
    parser.add_argument('-i', '--input', default='.', help='Input PDF file or directory.')
    parser.add_argument('-o', '--output', default='marker-output', help='Output directory.')
    parser.add_argument('--force-ocr', action='store_true', help='Force OCR on all pages.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    args = parser.parse_args()
    
    logger = setup_logging(args.debug)
    
    try:
        output_dirs = create_output_directories(args.output)
        pdf_files = find_pdf_files(args.input)
        
        if not pdf_files:
            logger.error(f"No PDF files found in: {args.input}")
            return 1
        
        logger.info(f"Found {len(pdf_files)} PDF(s) to convert.")
        
        converter = PDFToMarkdownConverter(output_dirs, logger, force_ocr=args.force_ocr)
        converter.convert_multiple_pdfs(pdf_files)
        
        stats = converter.conversion_stats
        logger.info("\n" + "="*50 + "\nCONVERSION SUMMARY\n" + "="*50)
        logger.info(f"Total files processed: {stats['total_files']}")
        logger.info(f"Successful: {stats['successful_conversions']}, Failed: {stats['failed_conversions']}")
        logger.info(f"Output directory: {args.output}")
        
        return 0 if stats['failed_conversions'] == 0 else 1
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 