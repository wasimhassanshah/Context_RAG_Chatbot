"""
Debug script to test Docling with your PDF files
Location: scripts/debug_docling.py
"""

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

def test_basic_conversion():
    """Test basic Docling conversion without options"""
    
    print("🧪 Testing Basic Docling Conversion")
    print("=" * 50)
    
    # Create basic converter
    converter = DocumentConverter()
    
    # Test with first PDF
    pdf_files = list(Path("data/raw").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    test_file = pdf_files[0]
    print(f"📄 Testing with: {test_file.name}")
    
    try:
        # Basic conversion
        result = converter.convert(str(test_file))
        doc = result.document
        
        print(f"✅ Conversion successful!")
        print(f"📋 Document type: {type(doc)}")
        
        # Check available attributes
        attrs = [attr for attr in dir(doc) if not attr.startswith('_')]
        print(f"📋 Available attributes: {attrs[:10]}...")  # Show first 10
        
        # Try different text extraction methods
        text_methods = [
            ('export_to_markdown', 'markdown'),
            ('export_to_text', 'text'), 
            ('body', 'body'),
            ('texts', 'texts')
        ]
        
        for method, name in text_methods:
            try:
                if hasattr(doc, method):
                    if method == 'texts':
                        content = doc.texts
                        if content:
                            text_preview = str(content[0].text)[:100] if hasattr(content[0], 'text') else str(content[0])[:100]
                            print(f"✅ {name}: Found {len(content)} text elements - {text_preview}...")
                        else:
                            print(f"⚠️ {name}: Empty")
                    else:
                        content = getattr(doc, method)
                        if callable(content):
                            content = content()
                        if content:
                            preview = str(content)[:100]
                            print(f"✅ {name}: {preview}...")
                        else:
                            print(f"⚠️ {name}: Empty")
                else:
                    print(f"❌ {name}: Method not available")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def test_all_formats():
    """Test conversion with different file types"""
    
    print("\n🧪 Testing All Document Formats")
    print("=" * 50)
    
    converter = DocumentConverter()
    
    # Test each file type
    test_files = {
        'PDF': list(Path("data/raw").glob("*.pdf"))[:2],  # Test first 2 PDFs
        'DOCX': list(Path("data/raw").glob("*.docx"))
    }
    
    for file_type, files in test_files.items():
        print(f"\n📋 Testing {file_type} files:")
        
        for file_path in files:
            print(f"   📄 {file_path.name}...")
            
            try:
                result = converter.convert(str(file_path))
                doc = result.document
                
                # Try to get some text
                text_found = False
                
                if hasattr(doc, 'export_to_text'):
                    try:
                        text = doc.export_to_text()
                        if text and text.strip():
                            print(f"      ✅ Text extracted: {len(text)} chars")
                            text_found = True
                    except:
                        pass
                
                if not text_found and hasattr(doc, 'texts'):
                    try:
                        texts = doc.texts
                        if texts:
                            total_chars = sum(len(str(t.text)) for t in texts if hasattr(t, 'text'))
                            print(f"      ✅ Text elements: {len(texts)} elements, {total_chars} chars")
                            text_found = True
                    except:
                        pass
                
                if not text_found:
                    print(f"      ⚠️ No text extracted")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")

if __name__ == "__main__":
    test_basic_conversion()
    test_all_formats()