import re
import os
from typing import List, Dict, Any

try:
    from langchain.schema import Document
except ImportError:
    print("HATA: LangChain kütüphanesi bulunamadı. Lütfen 'pip install langchain' komutu ile kurun.")
    exit()
def clean_text(text: str) -> str:
    """
    Verilen metindeki istenmeyen HTML etiketlerini ve Markdown formatlama
    karakterlerini temizler.
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'(\*\*|__|\*|_)', '', text)
    return text.strip()
def get_content_type(block: str) -> str:
    stripped_block = block.strip()
    
    if stripped_block.startswith('!['):
        return 'image'
    if stripped_block.startswith(('* ', '- ', '• ')):
        return 'bullet_list'
    if stripped_block.startswith('|') and stripped_block.endswith('|'):
        return 'table'
    if re.match(r'^\d+\.\s', stripped_block):
        return 'numbered_list'
    if stripped_block.startswith('```') and stripped_block.endswith('```'):
        return 'code_block'
    if stripped_block.lower().startswith('note:'):
        return 'note'
        
    return 'text'
def process_markdown_to_documents(md_text: str, source_file: str) -> List[Document]:
    blocks = md_text.split('\n\n')
    documents = []
    current_headers: Dict[str, str] = {}
    
    for block in blocks:
        clean_block = block.strip()
        if not clean_block or clean_block.startswith('!['):
            continue

        header_match = re.match(r'^(#+)\s(.*)', clean_block)
        if header_match and '\n' not in clean_block:
            level = len(header_match.group(1))
            title = clean_text(header_match.group(2))
            current_headers[f'h{level}'] = title
            for i in range(level + 1, 7):
                if f'h{i}' in current_headers:
                    del current_headers[f'h{i}']
            continue

        internal_links = re.findall(r'\[.*?\]\(#(.*?)\)', clean_block)
        external_links = re.findall(r'\[.*?\]\((https?://.*?)\)', clean_block)
        content_type = get_content_type(clean_block)
        
        header_str = " > ".join(current_headers.values())

        doc_metadata = {
            "source": source_file,
            "header_hierarchy": header_str,
            "content_type": content_type,
            "references_internal": ", ".join(internal_links),
            "references_external": ", ".join(external_links)
        }
        
        doc = Document(page_content=block, metadata=doc_metadata)
        documents.append(doc)
        
    return documents
if __name__ == "__main__":
    markdown_file_path = "C:/Users/hamdi/Desktop/parser_rag/using-manufacturing.md" # <--- DEĞİŞTİR

    try:
        with open(markdown_file_path, "r", encoding='utf-8') as f:
            markdown_content = f.read()
        
        source_filename = os.path.basename(markdown_file_path)
        
        processed_docs = process_markdown_to_documents(markdown_content, source_file=source_filename)

        print(f"BAŞARILI: '{source_filename}' dosyasından toplam {len(processed_docs)} adet metadata'lı doküman oluşturuldu.\n")
        
        print("--- DOKÜMAN 25-40 ARASI ÖNİZLEME ---")
        for i, doc in enumerate(processed_docs[24:39]): # index 24'ten başlar
            print(f"\n--- DOKÜMAN {i+25} ---")
            print(f"METADATA: {doc.metadata}")
            print("İÇERİK:")
            print(doc.page_content)
            print("-" * 25)

    except FileNotFoundError:
        print(f"HATA: Dosya bulunamadı! Lütfen yolu kontrol edin: {markdown_file_path}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")