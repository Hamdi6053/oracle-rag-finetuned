import os
import re
import pickle
import uuid
from typing import List, Dict, Any, Tuple

# LangChain'in en güncel ve doğru import yolları
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# =================================================================
# BÖLÜM 1: TEMİZLEME FONKSİYONLARI
# =================================================================

def clean_markdown_content(text: str) -> str:
    """
    PDF'den dönüştürülmüş Markdown içeriğini temizler:
    - Aşırı boşlukları kaldırır
    - Bozuk tablo formatlarını düzeltir  
    - Resim referanslarını temizler
    - Gereksiz satır sonlarını düzenler
    """
    
    # 1. Resim referanslarını kaldır
    text = re.sub(r'!\[\]\([^)]*\)', '', text)
    text = re.sub(r'!\[.*?\]\([^)]*\)', '', text)
    
    # 2. Span etiketlerini temizle ama içeriği koru
    text = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', text)
    
    # 3. Aşırı boşlukları temizle (5+ boşluk → 1 boşluk)
    text = re.sub(r' {5,}', ' ', text)
    
    # 4. Çoklu newline'ları düzenle (3+ → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 5. Satır sonundaki boşlukları temizle
    text = re.sub(r' +\n', '\n', text)
    
    # 6. Bozuk tablo satırlarını temizle
    text = re.sub(r'\|\s*\n\s*\|', '|', text)
    
    # 7. Boş tablo hücrelerini düzenle
    text = re.sub(r'\|\s+\|', '| |', text)
    
    # 8. Gereksiz **Contents** tablolarını kaldır (içerik listesi genelde gereksiz)
    text = re.sub(r'\*\*Contents\*\*.*?\n\|.*?\n(\|.*?\n)*', '', text, flags=re.DOTALL)
    
    return text.strip()

def is_content_meaningful(content: str) -> bool:
    """
    Gelişmiş içerik anlamlılık kontrolü - Web araştırması sonucu iyileştirildi.
    
    Kontrol edilenler:
    - Minimum uzunluk (100 karakter)
    - Cümle sayısı (en az 1 tam cümle)
    - Anlamlı kelime oranı
    - Header-only detection (gelişmiş)
    - İçerik çeşitliliği
    - Tablo separator kontrolü
    """
    content = content.strip()
    
    # 1. ⚡ Minimum uzunluk kontrolü (50 → 100 karakter)
    if len(content) < 100:
        return False
    
    # 2. 🚫 Sadece tablo separator'ları
    if re.match(r'^[\|\-\s\+\=]+$', content):
        return False
    
    # 3. 🚫 Gelişmiş header-only detection
    # Sadece markdown header'ları (#, ##, ###) ve minimal içerik
    header_pattern = r'^[#\s\*\-\_\[\]]+[\w\s\*\-\_\[\]]*[#\s\*\-\_\[\]]*$'
    if re.match(header_pattern, content):
        return False
    
    # 4. 📝 Cümle sayısı kontrolü
    # En az 1 tam cümle olmalı (nokta, ünlem, soru işareti ile biten)
    sentences = re.findall(r'[.!?]+', content)
    if len(sentences) == 0:
        return False
    
    # 5. 🔤 Anlamlı kelime kontrolü
    # Sadece harflerden oluşan kelimeleri say
    words = re.findall(r'\b[a-zA-ZçğıöşüÇĞIİÖŞÜ]+\b', content)
    if len(words) < 8:  # En az 8 anlamlı kelime
        return False
    
    # 6. 📊 İçerik çeşitliliği kontrolü
    # Sadece tekrarlanan kelimelerden oluşuyorsa red
    unique_words = set(word.lower() for word in words)
    if len(unique_words) < 5:  # En az 5 farklı kelime
        return False
    
    # 7. 🎯 Alfanumerik karakter oranı kontrolü
    # İçeriğin en az %40'ı alfanumerik olmalı
    alphanum_chars = len(re.findall(r'[a-zA-Z0-9çğıöşüÇĞIİÖŞÜ]', content))
    total_chars = len(content)
    if alphanum_chars / total_chars < 0.4:
        return False
    
    # 8. 🚫 Sadece liste öğeleri kontrolü
    # Sadece "- item", "* item", "1. item" formatında olan içerikleri red
    list_only_pattern = r'^[\s\-\*\d\.\)]+[\w\s\-\*\d\.\)]*$'
    if re.match(list_only_pattern, content) and len(sentences) == 0:
        return False
    
    # 9. ✅ Paragraph varlığı kontrolü
    # En az bir paragraph (çoklu cümle) olmalı
    paragraphs = content.split('\n\n')
    meaningful_paragraphs = [p for p in paragraphs if len(p.strip()) > 30 and '.' in p]
    if len(meaningful_paragraphs) == 0:
        return False
    
    return True

# =================================================================
# BÖLÜM 2: PICKLE STORE SINIFI (değişiklik yok)
# =================================================================
class SimplePickleStore:
    """Basit pickle tabanlı kalıcı depolama"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.store_file = os.path.join(storage_path, "documents.pkl")
        self._store = self._load_store()
    
    def _load_store(self) -> Dict:
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, 'rb') as f:
                    return pickle.load(f)
            except:
                print("Store dosyası okunamadı, yeni store oluşturuluyor.")
        return {}
    
    def _save_store(self):
        with open(self.store_file, 'wb') as f:
            pickle.dump(self._store, f)
    
    def mget(self, keys: List[str]) -> List[Document]:
        return [self._store.get(key) for key in keys]
    
    def mset(self, key_value_pairs: List[tuple]):
        for key, value in key_value_pairs:
            self._store[key] = value
        self._save_store()
    
    def mdelete(self, keys: List[str]):
        for key in keys:
            self._store.pop(key, None)
        self._save_store()

# =================================================================
# BÖLÜM 3: TEMİZLENMİŞ CHUNKING STRATEJİSİ
# =================================================================

def clean_text(text: str) -> str:
    """Başlık metinlerini temizler."""
    text = re.sub(r'<.*?>', '', text)  # HTML tagları
    text = re.sub(r'(\*\*|__|\*|_)', '', text)  # Markdown formatları
    text = re.sub(r'\s+', ' ', text)  # Çoklu boşlukları tek boşluğa çevir
    return text.strip()

def create_cleaned_parent_child_sections(
    md_text: str,
    source_file: str,
    max_child_length: int = 1200,  # Biraz daha küçük chunk'lar
) -> Tuple[List[Tuple[str, Document]], List[Document]]:
    """
    Temizlenmiş ve optimize edilmiş parent-child chunking.
    """
    
    print("🧹 Markdown içeriği temizleniyor...")
    cleaned_text = clean_markdown_content(md_text)
    print(f"✅ İçerik temizlendi: {len(md_text)} → {len(cleaned_text)} karakter")
    
    # MarkdownHeaderTextSplitter konfigürasyonu
    headers_to_split_on = [
        ("#", "h1"),           
        ("##", "h2"),          
        ("###", "h3"),         
        ("####", "h4"),        
    ]
    
    print("🔄 MarkdownHeaderTextSplitter ile bölme...")
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  
        return_each_line=False
    )
    
    md_header_splits = markdown_splitter.split_text(cleaned_text)
    print(f"✅ Başlık bazlı {len(md_header_splits)} bölüm oluşturuldu")
    
    # Anlamlı olmayan bölümleri filtrele
    meaningful_splits = []
    filtered_count = 0
    
    for doc in md_header_splits:
        if is_content_meaningful(doc.page_content):
            meaningful_splits.append(doc)
        else:
            filtered_count += 1
    
    print(f"🚮 {filtered_count} anlamsız bölüm filtrelendi")
    print(f"✅ {len(meaningful_splits)} anlamlı bölüm kaldı")
    
    if not meaningful_splits:
        print("⚠️ Hiç anlamlı bölüm bulunamadı, fallback stratejisi...")
        return _fallback_single_parent(cleaned_text, source_file, max_child_length)
    
    parents_for_store: List[Tuple[str, Document]] = []
    all_child_docs: List[Document] = []
    
    # H1 ve H2 bazlı parent grouping (iyileştirilmiş)
    current_parent_content = ""
    current_parent_title = "Başlangıç Bölümü"
    current_parent_level = "h1"
    parent_docs_buffer = []
    
    for doc in meaningful_splits:
        doc_metadata = doc.metadata
        doc_content = doc.page_content.strip()
        
        # En yüksek seviye başlığı bul
        header_level = _get_highest_header_level(doc_metadata)
        header_title = _get_primary_header_title(doc_metadata)
        
        # H1 veya H2 seviyesinde yeni parent başlatılıyor mu?
        if header_level in ['h1', 'h2']:
            # Önceki parent'ı kaydet (eğer anlamlıysa)
            if current_parent_content.strip() and is_content_meaningful(current_parent_content):
                parent_id = str(uuid.uuid4())
                parent_doc = Document(
                    page_content=current_parent_content.strip(),
                    metadata={
                        "source": source_file,
                        "header_hierarchy": current_parent_title,
                        "section_level": current_parent_level,
                        "content_length": len(current_parent_content.strip()),
                        "is_cleaned": True,  # Temizlenmiş olduğunu işaretle
                    }
                )
                parents_for_store.append((parent_id, parent_doc))
                
                # Bu parent'ın child'larını oluştur
                child_docs = _create_cleaned_child_chunks(
                    parent_docs_buffer, current_parent_title, parent_id, source_file, max_child_length
                )
                all_child_docs.extend(child_docs)
            
            # Yeni parent başlat
            current_parent_content = doc_content
            current_parent_title = header_title or f"Bölüm {len(parents_for_store) + 1}"
            current_parent_level = header_level
            parent_docs_buffer = [doc]
            
        else:
            # Mevcut parent'a ekle (H3+ seviyesi)
            current_parent_content += "\n\n" + doc_content
            parent_docs_buffer.append(doc)
    
    # Son parent'ı kaydet
    if current_parent_content.strip() and is_content_meaningful(current_parent_content):
        parent_id = str(uuid.uuid4())
        parent_doc = Document(
            page_content=current_parent_content.strip(),
            metadata={
                "source": source_file,
                "header_hierarchy": current_parent_title,
                "section_level": current_parent_level,
                "content_length": len(current_parent_content.strip()),
                "is_cleaned": True,
            }
        )
        parents_for_store.append((parent_id, parent_doc))
        
        child_docs = _create_cleaned_child_chunks(
            parent_docs_buffer, current_parent_title, parent_id, source_file, max_child_length
        )
        all_child_docs.extend(child_docs)
    
    print(f"✅ {len(parents_for_store)} parent ve {len(all_child_docs)} child doküman oluşturuldu")
    return parents_for_store, all_child_docs

def _get_highest_header_level(metadata: Dict) -> str:
    """Metadata'dan en yüksek seviye başlığı bul"""
    for level in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        if level in metadata:
            return level
    return 'h0'

def _get_primary_header_title(metadata: Dict) -> str:
    """Metadata'dan birincil başlık metnini al"""
    for level in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        if level in metadata and metadata[level]:
            return clean_text(metadata[level])
    return "Başlıksız Bölüm"

def _create_cleaned_child_chunks(
    parent_sections: List[Document],
    parent_title: str,
    parent_id: str, 
    source_file: str,
    max_length: int
) -> List[Document]:
    """
    Temizlenmiş child chunk'lar oluşturur.
    """
    child_docs = []
    
    for section_idx, section in enumerate(parent_sections):
        section_content = section.page_content.strip()
        section_metadata = section.metadata
        
        # Anlamlı içerik kontrolü
        if not is_content_meaningful(section_content):
            continue
        
        section_title = _get_primary_header_title(section_metadata)
        section_level = _get_highest_header_level(section_metadata)
        
        # İçerik bölme (temizlenmiş versiyon)
        if len(section_content) <= max_length:
            chunks = [section_content]
        else:
            chunks = _split_content_cleaned(section_content, max_length)
        
        # Her chunk için child doküman oluştur
        for chunk_idx, chunk in enumerate(chunks):
            # Son bir kez anlamlılık kontrolü
            if not is_content_meaningful(chunk):
                continue
                
            hierarchy = f"{parent_title} > {section_title}" if section_title != parent_title else parent_title
            
            child_meta = {
                "source": source_file,
                "header_hierarchy": hierarchy,
                "section_level": section_level,
                "parent_id": parent_id,
                "chunk_index": chunk_idx,
                "subsection_title": section_title,
                "content_length": len(chunk),
                "is_cleaned": True,
                **{k: v for k, v in section_metadata.items() if k not in ['source', 'header_hierarchy']}
            }
            
            child_docs.append(Document(page_content=chunk, metadata=child_meta))
    
    print(f"   📝 {parent_title}: {len(child_docs)} anlamlı child chunk oluşturuldu")
    return child_docs

def _split_content_cleaned(content: str, max_length: int) -> List[str]:
    """
    Temizlenmiş içerik bölme stratejisi.
    """
    if len(content) <= max_length:
        return [content]
    
    # RecursiveCharacterTextSplitter ile daha dikkatli bölme
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=150,  # Bağlam korunması için overlap
        length_function=len,
        separators=[
            "\n\n",      # Paragraf arası
            "\n",        # Satır sonu
            ". ",        # Cümle sonu
            "! ",        # Ünlem
            "? ",        # Soru
            "; ",        # Noktalı virgül
            " ",         # Boşluk
            ""           # Karakter bazlı
        ]
    )
    
    chunks = text_splitter.split_text(content)
    
    # Anlamlı olmayan chunk'ları filtrele
    meaningful_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if is_content_meaningful(chunk):
            meaningful_chunks.append(chunk)
    
    return meaningful_chunks

def _fallback_single_parent(md_text: str, source_file: str, max_child_length: int) -> Tuple[List[Tuple[str, Document]], List[Document]]:
    """Başlık bulunamadığında fallback stratejisi (temizlenmiş)"""
    cleaned_text = clean_markdown_content(md_text)
    
    parent_id = str(uuid.uuid4())
    parent_doc = Document(
        page_content=cleaned_text.strip(),
        metadata={
            "source": source_file,
            "header_hierarchy": "Tam Doküman",
            "section_level": "h0",
            "content_length": len(cleaned_text.strip()),
            "is_cleaned": True,
        }
    )
    
    # Tüm dokümanı anlamlı child chunk'lara böl
    chunks = _split_content_cleaned(cleaned_text, max_child_length)
    child_docs = []
    
    for idx, chunk in enumerate(chunks):
        child_meta = {
            "source": source_file,
            "header_hierarchy": "Tam Doküman > Bölüm",
            "section_level": "chunk",
            "parent_id": parent_id,
            "chunk_index": idx,
            "content_length": len(chunk),
            "is_cleaned": True,
        }
        child_docs.append(Document(page_content=chunk, metadata=child_meta))
    
    return [(parent_id, parent_doc)], child_docs

# =================================================================
# ANA PROGRAM AKIŞI
# =================================================================

# 1. VERİYİ YÜKLEME
markdown_file_path = "./using-manufacturing.md"
markdown_content = ""
source_filename = ""

try:
    with open(markdown_file_path, "r", encoding='utf-8') as f:
        markdown_content = f.read()
    source_filename = os.path.basename(markdown_file_path)
    print(f"✅ '{source_filename}' dosyası başarıyla okundu ({len(markdown_content)} karakter)")
except FileNotFoundError:
    print(f"❌ HATA: Dosya bulunamadı! Lütfen yolu kontrol et: {markdown_file_path}")
    exit()
except Exception as e:
    print(f"❌ Dosya okuma hatası: {e}")
    exit()

# 2. TEMİZLENMİŞ İNDEKSLEME
if markdown_content:
    print("\n🚀 Temizlenmiş Markdown Chunking Başlıyor")
    print("=" * 60)

    # Embedding Modelini Yükle
    print("📥 Embedding modeli (BAAI/bge-base-en) yükleniyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embedding modeli yüklendi")

    # Kalıcı Depolama Alanlarını Ayarla
    print("💾 Depolama alanları yapılandırılıyor...")
    vectorstore = Chroma(
        collection_name="oracle_md_enhanced_v2",  # Gelişmiş filtreleme ile temizlenmiş koleksiyon
        embedding_function=embeddings,
        persist_directory="./oracle_vector_db_md_enhanced"
    )
    pickle_store = SimplePickleStore("./parent_document_store_md_enhanced")
    print("✅ Depolama alanları hazır")

    # TEMİZLENMİŞ CHUNKING
    print("\n🧹 Temizlenmiş MarkdownHeaderTextSplitter ile bölme başlıyor...")
    
    parent_docs_for_store, child_docs_for_embedding = create_cleaned_parent_child_sections(
        markdown_content, 
        source_filename,
        max_child_length=1200  # Biraz daha küçük chunk'lar
    )
    
    print(f"\n📊 TEMİZLENMİŞ SONUÇLAR:")
    print(f"   📚 Parent dokümanlar: {len(parent_docs_for_store)} adet")
    print(f"   📄 Child dokümanlar: {len(child_docs_for_embedding)} adet")
    
    # Kalite kontrol
    if parent_docs_for_store:
        total_parent_length = sum(len(doc.page_content) for _, doc in parent_docs_for_store)
        avg_parent_length = total_parent_length / len(parent_docs_for_store)
        print(f"   📏 Ortalama parent uzunluğu: {avg_parent_length:.0f} karakter")
        
    if child_docs_for_embedding:
        total_child_length = sum(len(doc.page_content) for doc in child_docs_for_embedding)
        avg_child_length = total_child_length / len(child_docs_for_embedding)
        print(f"   📏 Ortalama child uzunluğu: {avg_child_length:.0f} karakter")

    # Kalite kontrol: En kısa ve en uzun chunk'ları göster
    if child_docs_for_embedding:
        lengths = [len(doc.page_content) for doc in child_docs_for_embedding]
        print(f"   📐 En kısa chunk: {min(lengths)} karakter")
        print(f"   📐 En uzun chunk: {max(lengths)} karakter")

    # Dokümanları Kalıcı Olarak Kaydet
    if child_docs_for_embedding and parent_docs_for_store:
        print("\n💾 Temizlenmiş dokümantlar kaydediliyor...")
        
        pickle_store.mset(parent_docs_for_store)
        print(f"✅ {len(parent_docs_for_store)} parent doküman kaydedildi")
        
        vectorstore.add_documents(child_docs_for_embedding)
        print(f"✅ {len(child_docs_for_embedding)} child doküman vektörleştirildi")
        
        print("\n🎉 Temizlenmiş indeksleme tamamlandı!")
        print("💡 Yeni koleksiyon: oracle_md_cleaned_v1")
        print("💡 2_query.py'de COLLECTION_NAME'i güncelleyebilirsiniz")
        
    else:
        print("❌ İşlenecek doküman bulunamadı!")
else:
    print("❌ Markdown içerik boş, indeksleme atlandı")
