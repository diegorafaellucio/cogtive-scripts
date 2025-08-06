import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ⚙️ CONFIGURAÇÃO
image_dir = "/home/diego/Downloads/frames_validation (1)/frames_validation"
upload_url = "https://internal-label.cogtive.com/api/projects/24/import"
auth_token = "a27c0b74353fe5f041d9d54d8323d8dfb3457c64"
batch_size = 50
max_threads = 10  # número de uploads em paralelo

headers = {
    "Authorization": f"Token {auth_token}"
}

def upload_image(filepath):
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "rb") as f:
            files = {"file": (filename, f, "image/jpeg")}
            response = requests.post(upload_url, headers=headers, files=files, timeout=60)
        if response.status_code in [200, 201]:
            return True
        else:
            print(f"❌ Erro ao enviar {filename}: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Falha ao enviar {filename}: {e}")
        return False

# 🔍 Lista de imagens
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
print(f"🔍 Total de imagens: {len(images)}")

# 🚀 Upload em lotes com paralelismo
for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    print(f"\n📦 Enviando lote {i//batch_size + 1} com {len(batch)} imagens...")
    
    with tqdm(total=len(batch), desc="Progresso", unit="img") as pbar:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(upload_image, img): img for img in batch}
            
            success_count = 0
            for future in as_completed(futures):
                img_path = futures[future]
                filename = os.path.basename(img_path)
                result = future.result()
                
                if result:
                    success_count += 1
                
                pbar.update(1)
                
            pbar.set_postfix({"sucesso": f"{success_count}/{len(batch)}"})
