"""
Docling Document Processing Pipeline for Contextual RAG
Handles PDF and DOCX processing with intelligent chunking
Location: src/contextual_rag/document_processing/docling_processor.py
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

@dataclass
class DocumentChunk:
    """Represents a processed document chunk"""
    id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    chunk_type: str = "text"

@dataclass
class ProcessedDocument:
    """Represents a fully processed document"""
    id: str
    filename: str
    file_type: str
    title: str
    total_pages: int
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    processing_timestamp: datetime

class DoclingProcessor:
    """Advanced document processing using Docling pipeline"""
    
    def __init__(self, 
                 raw_data_path: str = "data/raw",
                 processed_data_path: str = "data/processed",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Ensure directories exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docling converter
        self.converter = self._setup_docling_converter()
        
        # Supported file types
        self.supported_extensions = {'.pdf', '.docx', '.doc'}
    
    def _setup_docling_converter(self) -> DocumentConverter:
        """Setup Docling converter with basic configuration"""
        
        # Use basic converter without advanced options to avoid compatibility issues
        converter = DocumentConverter()
        return converter
    
    def _generate_document_id(self, filepath: Path) -> str:
        """Generate unique document ID based on filename and size"""
        # Use filename + file size for consistent ID generation
        file_info = f"{filepath.name}_{filepath.stat().st_size}"
        file_hash = hashlib.md5(file_info.encode()).hexdigest()
        return f"doc_{file_hash[:12]}"
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        return f"{document_id}_chunk_{chunk_index:04d}"
    
    def process_single_document(self, filepath: Path) -> Optional[ProcessedDocument]:
        """Process a single document using Docling"""
        
        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}")
            return None
        
        if filepath.suffix.lower() not in self.supported_extensions:
            self.logger.warning(f"Unsupported file type: {filepath.suffix}")
            return None
        
        self.logger.info(f"📄 Processing document: {filepath.name}")
        
        try:
            # Generate document ID
            doc_id = self._generate_document_id(filepath)
            
            # Convert document using Docling
            self.logger.info(f"🔄 Converting {filepath.name} with Docling...")
            result = self.converter.convert(str(filepath))
            doc = result.document
            
            # Extract document metadata
            metadata = {
                "source_file": str(filepath),
                "file_size": filepath.stat().st_size,
                "file_extension": filepath.suffix.lower(),
                "processing_method": "docling",
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Extract title
            title = self._extract_title(doc, filepath)
            
            # Extract content
            full_text = self._extract_full_text(doc)
            
            if not full_text.strip():
                self.logger.warning(f"No text content extracted from {filepath.name}")
                return None
            
            # Create chunks
            chunks = self._create_chunks(full_text, doc_id)
            
            # Create processed document
            processed_doc = ProcessedDocument(
                id=doc_id,
                filename=filepath.name,
                file_type=filepath.suffix.lower(),
                title=title,
                total_pages=1,  # Will be updated if we can extract page info
                chunks=chunks,
                metadata=metadata,
                processing_timestamp=datetime.now()
            )
            
            self.logger.info(f"✅ Processed {filepath.name}: {len(chunks)} chunks, {len(full_text)} characters")
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {filepath.name}: {e}")
            return None
    
    def _extract_title(self, doc, filepath: Path) -> str:
        """Extract document title"""
        
        # Try to get title from document metadata
        if hasattr(doc, 'meta') and hasattr(doc.meta, 'title') and doc.meta.title:
            return doc.meta.title.strip()
        
        # Fallback to filename
        return filepath.stem
    
    def _extract_full_text(self, doc) -> str:
        """Extract all text content from the document"""
        
        # Use the method that worked in debug script
        try:
            if hasattr(doc, 'export_to_text'):
                text_content = doc.export_to_text()
                if text_content and text_content.strip():
                    return text_content
        except Exception as e:
            self.logger.debug(f"export_to_text failed: {e}")
        
        # Fallback methods
        text_parts = []
        
        try:
            if hasattr(doc, 'texts') and doc.texts:
                for text_element in doc.texts:
                    if hasattr(text_element, 'text'):
                        text_content = str(text_element.text).strip()
                        if text_content:
                            text_parts.append(text_content)
        except Exception as e:
            self.logger.debug(f"Text elements extraction failed: {e}")
        
        return '\n\n'.join(text_parts) if text_parts else ""
    
    def _create_chunks(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Create chunks from text with overlap"""
        
        chunks = []
        
        # Simple paragraph-based chunking
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = DocumentChunk(
                    id=self._generate_chunk_id(document_id, chunk_index),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=current_chunk.strip(),
                    metadata={
                        'word_count': len(current_chunk.split()),
                        'char_count': len(current_chunk),
                        'chunk_method': 'paragraph_based'
                    },
                    chunk_type='text'
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-50:] if len(words) > 50 else words  # 50 word overlap
                current_chunk = ' '.join(overlap_words) + '\n\n' + paragraph
                chunk_index += 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                id=self._generate_chunk_id(document_id, chunk_index),
                document_id=document_id,
                chunk_index=chunk_index,
                content=current_chunk.strip(),
                metadata={
                    'word_count': len(current_chunk.split()),
                    'char_count': len(current_chunk),
                    'chunk_method': 'paragraph_based',
                    'final_chunk': True
                },
                chunk_type='text'
            )
            chunks.append(chunk)
        
        return chunks
    
    def save_processed_document(self, processed_doc: ProcessedDocument) -> bool:
        """Save processed document to JSON file"""
        
        try:
            output_file = self.processed_data_path / f"{processed_doc.id}.json"
            
            # Convert to serializable format
            doc_data = {
                'id': processed_doc.id,
                'filename': processed_doc.filename,
                'file_type': processed_doc.file_type,
                'title': processed_doc.title,
                'total_pages': processed_doc.total_pages,
                'metadata': processed_doc.metadata,
                'processing_timestamp': processed_doc.processing_timestamp.isoformat(),
                'chunk_count': len(processed_doc.chunks),
                'chunks': []
            }
            
            # Add chunks
            for chunk in processed_doc.chunks:
                chunk_data = {
                    'id': chunk.id,
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'page_number': chunk.page_number,
                    'chunk_type': chunk.chunk_type
                }
                doc_data['chunks'].append(chunk_data)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved processed document: {output_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save processed document: {e}")
            return False
    
    def process_all_documents(self) -> List[ProcessedDocument]:
        """Process all documents in the raw data directory"""
        
        self.logger.info(f"🚀 Starting batch document processing from: {self.raw_data_path}")
        
        processed_docs = []
        
        # Find all supported files
        files_to_process = []
        for ext in self.supported_extensions:
            files_to_process.extend(self.raw_data_path.glob(f"*{ext}"))
        
        if not files_to_process:
            self.logger.warning(f"⚠️ No supported files found in {self.raw_data_path}")
            print(f"⚠️ No PDF or DOCX files found in {self.raw_data_path}")
            print(f"   Please add your documents to this directory and try again.")
            return processed_docs
        
        self.logger.info(f"📋 Found {len(files_to_process)} files to process")
        print(f"📋 Found {len(files_to_process)} files to process:")
        for f in files_to_process:
            print(f"   📄 {f.name}")
        
        # Process each file
        for filepath in files_to_process:
            print(f"\n🔄 Processing: {filepath.name}")
            
            processed_doc = self.process_single_document(filepath)
            
            if processed_doc:
                # Save processed document
                if self.save_processed_document(processed_doc):
                    processed_docs.append(processed_doc)
                    print(f"✅ Successfully processed: {filepath.name}")
                else:
                    print(f"❌ Failed to save: {filepath.name}")
            else:
                print(f"❌ Failed to process: {filepath.name}")
        
        print(f"\n🎉 Batch processing complete!")
        print(f"✅ Successfully processed: {len(processed_docs)} documents")
        total_chunks = sum(len(doc.chunks) for doc in processed_docs)
        print(f"📊 Total chunks generated: {total_chunks}")
        
        return processed_docs
    
    def get_processing_summary(self, processed_docs: List[ProcessedDocument]) -> Dict[str, Any]:
        """Generate processing summary statistics"""
        
        if not processed_docs:
            return {"status": "no_documents"}
        
        total_chunks = sum(len(doc.chunks) for doc in processed_docs)
        total_chars = sum(sum(len(chunk.content) for chunk in doc.chunks) for doc in processed_docs)
        
        summary = {
            "total_documents": len(processed_docs),
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "average_chunks_per_doc": total_chunks / len(processed_docs),
            "average_chunk_size": total_chars / total_chunks if total_chunks > 0 else 0,
            "documents": []
        }
        
        for doc in processed_docs:
            doc_chars = sum(len(chunk.content) for chunk in doc.chunks)
            doc_summary = {
                "filename": doc.filename,
                "title": doc.title,
                "chunks": len(doc.chunks),
                "characters": doc_chars,
                "file_type": doc.file_type,
                "processing_time": doc.processing_timestamp.isoformat()
            }
            summary["documents"].append(doc_summary)
        
        return summary

def setup_docling_processor() -> DoclingProcessor:
    """Initialize and setup Docling processor"""
    
    print("🚀 Setting up Docling Document Processor...")
    
    try:
        processor = DoclingProcessor()
        print("✅ Docling processor initialized successfully")
        return processor
    except Exception as e:
        print(f"❌ Failed to initialize Docling processor: {e}")
        return None

if __name__ == "__main__":
    # Test the processor
    processor = setup_docling_processor()
    
    if processor:
        processed_docs = processor.process_all_documents()
        
        if processed_docs:
            summary = processor.get_processing_summary(processed_docs)
            print(f"\n📊 Processing Summary:")
            print(f"   Documents: {summary['total_documents']}")
            print(f"   Chunks: {summary['total_chunks']}")
            print(f"   Avg chunks/doc: {summary['average_chunks_per_doc']:.1f}")
        else:
            print("❌ No documents processed successfully")
    else:
        print("❌ Failed to setup processor")