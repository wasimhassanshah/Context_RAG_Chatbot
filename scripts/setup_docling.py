"""
Setup Script for Docling Document Processing
Location: scripts/setup_docling.py
"""

import os
import sys
from pathlib import Path
import subprocess

def check_directory_structure():
    """Check and create required directories"""
    print("📁 Checking directory structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/embeddings",
        "src/contextual_rag/document_processing"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_path}")
        else:
            print(f"   📁 Exists: {dir_path}")
    
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔧 Checking Docling dependencies...")
    
    try:
        import docling
        # Try to get version, fallback if not available
        try:
            version = docling.__version__
            print(f"   ✅ docling: {version}")
        except AttributeError:
            print("   ✅ docling: installed (version info not available)")
    except ImportError:
        print("   ❌ docling not found")
        return False
    
    try:
        from docling.document_converter import DocumentConverter
        print("   ✅ DocumentConverter available")
    except ImportError:
        print("   ❌ DocumentConverter not available")
        return False
    
    try:
        from docling.datamodel.base_models import InputFormat
        print("   ✅ InputFormat available")
    except ImportError:
        print("   ❌ InputFormat not available")
        return False
    
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        print("   ✅ PdfPipelineOptions available")
    except ImportError:
        print("   ❌ PdfPipelineOptions not available")
        return False
    
    return True

def install_dependencies():
    """Install missing dependencies"""
    print("\n📦 Installing Docling dependencies...")
    
    packages = [
        "docling",
        "docling-core", 
        "docling-ibm-models",
        "docling-parse"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            result = subprocess.run([
                "uv", "add", package
            ], capture_output=True, text=True, check=True)
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            print(f"   📋 Output: {e.stdout}")
            print(f"   📋 Error: {e.stderr}")
            return False
        except FileNotFoundError:
            # Fallback to pip if uv not found
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, check=True)
                print(f"   ✅ {package} installed successfully (via pip)")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e}")
                return False
    
    return True

def check_documents():
    """Check for documents in data/raw"""
    print("\n📄 Checking for documents...")
    
    raw_path = Path("data/raw")
    supported_extensions = ['.pdf', '.docx', '.doc']
    
    files_found = []
    for ext in supported_extensions:
        files = list(raw_path.glob(f"*{ext}"))
        files_found.extend(files)
    
    if files_found:
        print(f"   📋 Found {len(files_found)} documents:")
        for file in files_found:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"      📄 {file.name} ({size_mb:.2f} MB)")
    else:
        print("   ⚠️ No PDF or DOCX files found in data/raw/")
        print("   💡 Please add your documents to data/raw/ before testing")
    
    return len(files_found) > 0

def create_test_files():
    """Create required module files if they don't exist"""
    print("\n📝 Checking module files...")
    
    # Create __init__.py files
    init_files = [
        "src/contextual_rag/document_processing/__init__.py"
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        if not path.exists():
            path.touch()
            print(f"   ✅ Created: {init_file}")
        else:
            print(f"   📁 Exists: {init_file}")
    
    return True

def run_quick_test():
    """Run a quick test of Docling functionality"""
    print("\n🧪 Running quick Docling test...")
    
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        
        # Create converter
        converter = DocumentConverter()
        print("   ✅ DocumentConverter created successfully")
        
        # Test basic configuration
        pdf_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        print("   ✅ PdfPipelineOptions configured successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 DOCLING DOCUMENT PROCESSING SETUP")
    print("=" * 60)
    
    # Step 1: Check directory structure
    if not check_directory_structure():
        print("❌ Directory setup failed")
        return False
    
    # Step 2: Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n💡 Installing missing dependencies...")
        if not install_dependencies():
            print("❌ Dependency installation failed")
            return False
        
        # Recheck after installation
        if not check_dependencies():
            print("❌ Dependencies still missing after installation")
            return False
    
    # Step 3: Create module files
    if not create_test_files():
        print("❌ Module file creation failed")
        return False
    
    # Step 4: Quick test
    if not run_quick_test():
        print("❌ Quick test failed")
        return False
    
    # Step 5: Check for documents
    has_docs = check_documents()
    
    print("\n🎉 DOCLING SETUP COMPLETE!")
    print("=" * 40)
    print("✅ Directory structure ready")
    print("✅ Dependencies installed")
    print("✅ Module files created")
    print("✅ Docling functionality tested")
    
    if has_docs:
        print("✅ Documents found for processing")
        print("\n🚀 Next steps:")
        print("1. Run: python tests/test_docling_processor.py")
        print("2. If tests pass, documents will be processed automatically")
    else:
        print("⚠️ No documents found")
        print("\n📋 Next steps:")
        print("1. Add your PDF/DOCX files to data/raw/")
        print("2. Run: python tests/test_docling_processor.py")
    
    return True

if __name__ == "__main__":
    main()