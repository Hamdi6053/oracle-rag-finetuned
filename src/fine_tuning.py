!pip uninstall -y bitsandbytes
!pip install -U bitsandbytes
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.26" trl peft accelerate

print("Kütüphaneler yüklendi. Şimdi 'Runtime' -> 'Restart runtime' yapın ve sonraki hücreyi çalıştırın.")
import os
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import json
import re
try:
    import bitsandbytes as bnb
    print(f"✓ Bitsandbytes sürümü: {bnb.__version__}")
except ImportError:
    print("❌ Bitsandbytes kurulu değil! Lütfen oturumu yeniden başlatın.")
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
DATASET_FILE_PATH = "/kaggle/input/questionanswer/test_qa.json"
OUTPUT_ADAPTER_DIR = "dora-finetuned-qwen2-1.5b-oracle"
MAX_SEQ_LENGTH = 2048
DORA_RANK = 16
DORA_ALPHA = 32
def load_data_from_custom_format(filepath):
    """
    JSON dosyasını okur ve question/answer formatını input/output formatına dönüştürür.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"HATA: Veri seti dosyası bulunamadı: {filepath}")
        return []

    data = []
    success_count = 0
    error_count = 0
    
    try:
        parsed_data = json.loads(content)
        if isinstance(parsed_data, list):
            for item in parsed_data:
                if 'question' in item and 'answer' in item:
                    data.append({
                        'input': item['question'],
                        'output': item['answer']
                    })
                    success_count += 1
                else:
                    error_count += 1
    except json.JSONDecodeError:
        print("JSON array formatı değil, JSONL formatı deneniyor...")
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if 'question' in parsed and 'answer' in parsed:
                    data.append({
                        'input': parsed['question'],
                        'output': parsed['answer']
                    })
                    success_count += 1
                else:
                    error_count += 1
                    if error_count <= 5:
                        print(f"Satır {line_num}: 'question' ve 'answer' alanları bulunamadı")
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 5:
                    print(f"Satır {line_num} ayrıştırılamadı: {str(e)[:100]}...")
        
        if not data:
            print("JSONL formatı da çalışmadı, ```json blokları deneniyor...")
            json_blocks = re.findall(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
            
            for i, block in enumerate(json_blocks, 1):
                try:
                    cleaned_block = block.strip()
                    parsed = json.loads(cleaned_block)
                    if 'question' in parsed and 'answer' in parsed:
                        data.append({
                            'input': parsed['question'],
                            'output': parsed['answer']
                        })
                        success_count += 1
                    else:
                        error_count += 1
                        if error_count <= 5:
                            print(f"Blok {i}: 'question' ve 'answer' alanları bulunamadı")
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"Blok {i} ayrıştırılamadı: {str(e)[:100]}...")
                
    print(f"\n✓ Başarılı: {success_count} örnek")
    print(f"❌ Hatalı: {error_count} örnek")
    
    if not data:
        raise ValueError("HATA: Hiçbir geçerli örnek yüklenemedi!")
        
    return data
print("Veri seti yükleniyor...")
json_data = load_data_from_custom_format(DATASET_FILE_PATH)
dataset = Dataset.from_list(json_data)

print(f"\n📊 Veri seti özeti:")
print(f"Toplam örnek sayısı: {len(dataset)}")
print(f"İlk örnek: {dataset[0]}")
print(f"\n🤖 Model yükleniyor: {MODEL_NAME}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Mevcut VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    print("✓ Model başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Model yükleme hatası: {e}")
    print("Alternatif model denenecek...")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3.2-3b-instruct-bnb-4bit",  # Alternatif model
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        print("✓ Alternatif model başarıyla yüklendi!")
    except Exception as e2:
        print(f"❌ Alternatif model de yüklenemedi: {e2}")
        raise
print("\n🔧 DoRA adaptörleri ekleniyor...")

model = FastLanguageModel.get_peft_model(
    model,
    r=DORA_RANK,
    lora_alpha=DORA_ALPHA,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    use_dora=True,
)

print("✓ DoRA adaptörleri eklendi!")
model.print_trainable_parameters()
def formatting_prompts_func(examples):
    """
    Oracle-specific formatting function.
    input/output alanlarını kullanarak chat formatına dönüştürür.
    """
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for input_text, output_text in zip(inputs, outputs):
        messages = [
            {"role": "system", "content": "You are an Oracle Fusion Cloud SCM expert assistant. Provide accurate and helpful information about Oracle manufacturing, supply chain management, and related processes."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        
        try:
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(formatted)
        except Exception as e:
            print(f"Chat template hatası, basit format kullanılıyor: {e}")
            simple_format = f"<|system|>You are an Oracle Fusion Cloud SCM expert assistant. Provide accurate and helpful information about Oracle manufacturing, supply chain management, and related processes.<|end|>\n<|user|>{input_text}<|end|>\n<|assistant|>{output_text}<|end|>"
            texts.append(simple_format)
    
    return {"text": texts}
print("\n📝 Veri seti formatlanıyor...")
dataset = dataset.map(formatting_prompts_func, batched=True)
print("✓ Veri seti formatlandı!")
print(f"\nFormatlanmış örnek:\n{dataset[0]['text'][:500]}...")
print("\n🚀 Eğitim başlatılıyor...")
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    num_train_epochs=2,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    output_dir="outputs",
    report_to="none",
    save_steps=100,
    save_total_limit=3,
    dataloader_pin_memory=False,
    group_by_length=True,
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    packing=False,
    dataset_num_proc=2,
)
print("=" * 50)
print("EĞİTİM BAŞLIYOR...")
print("=" * 50)

trainer_stats = trainer.train()

print("=" * 50)
print("EĞİTİM TAMAMLANDI!")
print("=" * 50)
print(f"Eğitim istatistikleri: {trainer_stats}")
print(f"\n💾 Model kaydediliyor: {OUTPUT_ADAPTER_DIR}")

model.save_pretrained(OUTPUT_ADAPTER_DIR)
tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

print(f"✅ İşlem tamamlandı!")
print(f"📁 Adaptörler '/kaggle/working/{OUTPUT_ADAPTER_DIR}' klasörüne kaydedildi.")
print("\n🧪 Hızlı test:")
FastLanguageModel.for_inference(model)
test_questions = [
    "What is the purpose of Oracle Fusion Cloud SCM Using Manufacturing?",
    "How do you implement resource allocation in Oracle Fusion Cloud SCM?",
    "What are the key features of Oracle Fusion Cloud SCM in relation to supply chain optimization?"
]

for test_question in test_questions[:2]:  # İlk 2 soruyu test et
    test_messages = [
        {"role": "system", "content": "You are an Oracle Fusion Cloud SCM expert assistant."},
        {"role": "user", "content": test_question},
    ]

    try:
        test_prompt = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True 
        )
    except:
        test_prompt = f"<|system|>You are an Oracle Fusion Cloud SCM expert assistant.<|end|>\n<|user|>{test_question}<|end|>\n<|assistant|>"

    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            use_cache=True,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🔍 Soru: {test_question}")

    # Cevabı temizle
    assistant_marker = 'assistant'
    if assistant_marker in response:
        clean_response = response.split(assistant_marker)[-1].strip()
        if clean_response.startswith('>'):
            clean_response = clean_response[1:].strip()
    else:
        clean_response = response.split(test_question)[-1].strip()

    print(f"🤖 Cevap: {clean_response}")
    print("-" * 80)

print("\n🎉 Tüm işlemler başarıyla tamamlandı!")
print(f"📈 Toplam {len(dataset)} örnek ile {training_args.num_train_epochs} epoch eğitim tamamlandı.")
print(f"💾 Model '/kaggle/working/{OUTPUT_ADAPTER_DIR}' klasörüne kaydedildi.")

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel
import os
import gc
base_model_name = "Qwen/Qwen2-1.5B-Instruct"
adapter_path = "/kaggle/working/dora-finetuned-qwen2-1.5b-oracle"
merged_model_path = "/kaggle/working/oracle-qwen2-1.5b-merged-final"
hub_repo_name = "ozkurt7/oracle-qwen2-1.5b-merged-final"
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
print("="*60 + "\nMODEL BİRLEŞTİRME (CPU)\n" + "="*60)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=False,
    device_map="cpu",
)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
print("✅ Model başarıyla birleştirildi!")
gc.collect()
print("\n" + "="*60 + "\nBİRLEŞTİRİLMİŞ MODELİ HUB'A YÜKLEME\n" + "="*60)
try:
    print(f"💾 Model '{merged_model_path}' yoluna kaydediliyor...")
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"✅ Model yerel olarak kaydedildi.")
    
    print(f"🌐 Model Hub'a yükleniyor: {hub_repo_name}")
    tokenizer.push_to_hub(hub_repo_name, token=hf_token)
    model.push_to_hub(hub_repo_name, token=hf_token)
    print(f"✅ Model başarıyla Hub'a yüklendi: https://huggingface.co/{hub_repo_name}")

except Exception as e:
    print(f"❌ Model yükleme hatası: {e}")

print("\n" + "="*60 + "\n🎉 İŞLEM TAMAMLANDI!\n" + "="*60)