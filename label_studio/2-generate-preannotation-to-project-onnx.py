import requests
import time
import json
from tqdm import tqdm
import os
import io
import numpy as np
import tempfile
from PIL import Image
import onnxruntime as ort
import cv2
from ultralytics import YOLO

# Modo de debug - definir como True para processar apenas a primeira imagem
DEBUG_MODE = False

# Configuração da API
project_id = 25
api_base = f"https://internal-label.cogtive.com/api/projects/{project_id}/tasks/"
headers = {
    "Authorization": "Token a27c0b74353fe5f041d9d54d8323d8dfb3457c64"
}


# Conjunto para verificar IDs de tarefas já processadas
processed_ids = set()

# Definir o tamanho da página e quantidade de tarefas a baixar com base no modo
if DEBUG_MODE:
    print("\n=== MODO DEBUG ATIVADO - Baixando apenas uma tarefa ===\n")
    page_size = 1
    max_tasks = 1
else:
    page_size = 100
    max_tasks = None  # Sem limite

page = 1
all_tasks = []
page_count = 0

# Primeira requisição para descobrir o número total de tarefas
print(f"Obtendo informações sobre o projeto {project_id}...")

# Configurar parâmetros conforme a documentação da API
params = {
    'page': 1,
    'page_size': 1,  # Apenas para obter o count total
}

response = requests.get(api_base, headers=headers, params=params)
response.raise_for_status()
data = response.json()

# Verificar resposta e obter contagem total se disponível
if isinstance(data, dict):
    total_count = data.get("count")
    if total_count:
        print(f"Total oficial de tarefas: {total_count}")
    else:
        total_count = 13052  # Valor informado pelo usuário
        print(f"Usando contagem estimada: {total_count}")
else:
    total_count = 13052  # Valor informado pelo usuário
    print(f"Usando contagem estimada: {total_count}")

# Configurar barra de progresso
if DEBUG_MODE:
    pbar = tqdm(total=1, desc="Baixando tarefas")
else:
    pbar = tqdm(total=total_count, desc="Baixando tarefas")

# Começar a paginação para obter todas as tarefas
while True:
    page_count += 1
    
    # Configurar parâmetros para esta página
    params = {
        'page': page,
        'page_size': page_size,
    }
    
    try:
        print(f"Requisitando página {page} (tamanho: {page_size})")
        response = requests.get(api_base, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Extrair tarefas da resposta
        if isinstance(data, list):
            tasks = data
            has_next = bool(tasks)  # Se tem tarefas, assumimos que pode ter próxima página
        else:
            tasks = data.get("results", [])
            # Verificar se tem próxima página
            has_next = data.get("next") is not None
        
        # Se não há tarefas, terminar
        if not tasks:
            print("Nenhuma tarefa retornada nesta página. Finalizando.")
            break
            
        # Processar tarefas e verificar duplicatas
        new_tasks_count = 0
        for task in tasks:
            task_id = task.get('id')
            if task_id not in processed_ids:
                processed_ids.add(task_id)
                all_tasks.append(task)
                new_tasks_count += 1
                
        print(f"Página {page}: {new_tasks_count} novas tarefas adicionadas")
        
        # Atualizar barra de progresso
        pbar.update(new_tasks_count)
        
        # Verificar se já baixamos o número máximo de tarefas (modo debug)
        if DEBUG_MODE or (max_tasks and len(all_tasks) >= max_tasks):
            print("Número máximo de tarefas atingido. Finalizando.")
            break
        
        # Verificar se devemos continuar para a próxima página
        if not has_next or new_tasks_count == 0:
            print("Não há mais páginas para buscar. Finalizando.")
            break
            
        # Avançar para a próxima página
        page += 1
        
        # Pequena pausa para não sobrecarregar a API
        time.sleep(0.1)
        
    except requests.exceptions.RequestException as e:
        print(f"\nErro ao fazer requisição: {e}")
        # Se for um erro temporário, esperar e tentar novamente
        if response.status_code in (429, 500, 502, 503, 504):
            wait_time = 5
            print(f"Erro temporário. Tentando novamente em {wait_time} segundos...")
            time.sleep(wait_time)
            continue
        else:
            break

pbar.close()

# Mostrar resultados finais
print(f"\nTotal de tarefas coletadas: {len(all_tasks)}")
print(f"Total de tarefas únicas (por ID): {len(processed_ids)}")
print(f"Total de páginas acessadas: {page_count}")

if len(all_tasks) > 0:
    print("\nDetalhes estatísticos:")
    
    # Estatísticas sobre predições e anotações
    with_predictions = sum(1 for task in all_tasks if task.get("predictions"))
    with_annotations = sum(1 for task in all_tasks if task.get("annotations"))
    
    print(f"Tarefas com predições: {with_predictions} ({with_predictions/len(all_tasks)*100:.1f}%)")
    print(f"Tarefas com anotações: {with_annotations} ({with_annotations/len(all_tasks)*100:.1f}%)")
    
    print("\nDetalhes das primeiras 5 tarefas:")
    print("-" * 80)
    
    for i, task in enumerate(all_tasks[:min(5, len(all_tasks))], 1):
        has_predictions = "Sim" if task.get("predictions") else "Não"
        has_annotations = "Sim" if task.get("annotations") else "Não"
        image_path = task['data'].get('image', 'N/A')
        
        print(f"{i}. Task ID: {task['id']}")
        print(f"   Imagem: {image_path}")
        print(f"   Tem predições: {has_predictions}")
        print(f"   Tem anotações: {has_annotations}")
        print("-" * 80)

# ========== ADICIONAR PREDIÇÕES COM MODELO YOLO STANDARD ==========

# Caminho para o modelo YOLO
model_path = "/home/diego/Downloads/yolov5s-kavak.onnx"

# Label mapping para o modelo Kavak (baseado na transcripção fornecida)
KAVAK_LABELS = {
    1: "license_plate",
    2: "person", 
    3: "prisma",
    4: "car"
}

# Função para pré-processamento da imagem para YOLOv5
def preprocess_image(image_path, input_size=640):
    """Pré-processa imagem para inferência YOLOv5 ONNX"""
    # Carregar imagem
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    
    # Redimensionar mantendo aspect ratio
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Redimensionar
    resized = cv2.resize(image, (new_width, new_height))
    
    # Padding para chegar ao tamanho exato
    pad_width = input_size - new_width
    pad_height = input_size - new_height
    
    # Adicionar padding
    padded = np.pad(resized, ((0, pad_height), (0, pad_width), (0, 0)), constant_values=114)
    
    # Normalizar e converter para formato NCHW
    input_tensor = padded.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC para CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Adicionar batch dimension
    
    return input_tensor, original_width, original_height, scale

# Função para pós-processamento das saídas YOLOv5
def postprocess_yolo_output(output, original_width, original_height, scale, conf_threshold=0.25, iou_threshold=0.45):
    """Pós-processa saídas YOLOv5 ONNX para extrair detecções"""
    detections = []
    
    # YOLOv5 output shape: (1, 25200, 9) onde 9 = 4 (bbox) + 1 (objectness) + 4 (classes)
    predictions = output[0]  # Remove batch dimension
    
    for prediction in predictions:
        # prediction = [cx, cy, w, h, objectness, class0, class1, class2, class3]
        cx, cy, w, h, objectness = prediction[:5]
        class_scores = prediction[5:]
        
        # Filtrar por objectness score
        if objectness < conf_threshold:
            continue
            
        # Encontrar classe com maior score
        class_id = np.argmax(class_scores) + 1  # +1 porque nossos IDs começam em 1
        class_confidence = class_scores[class_id - 1] * objectness
        
        # Filtrar por confidence da classe
        if class_confidence < conf_threshold:
            continue
        
        # Converter coordenadas do centro para x1, y1, x2, y2
        # Ajustar coordenadas para a imagem original
        x1 = (cx - w/2) / scale
        y1 = (cy - h/2) / scale
        x2 = (cx + w/2) / scale  
        y2 = (cy + h/2) / scale
        
        # Garantir que coordenadas estejam dentro dos limites da imagem
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        detections.append({
            'box': [x1, y1, x2, y2],
            'confidence': float(class_confidence),
            'class_id': int(class_id),
            'class_name': KAVAK_LABELS.get(class_id, 'unknown')
        })
    
    # Aplicar Non-Maximum Suppression
    if detections:
        detections = apply_nms(detections, iou_threshold)
    
    return detections

# Função para aplicar Non-Maximum Suppression
def apply_nms(detections, iou_threshold):
    """Aplica NMS nas detecções"""
    if not detections:
        return []
    
    # Ordenar por confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        
        # Remover detecções com alta sobreposição
        detections = [det for det in detections if calculate_iou(current['box'], det['box']) < iou_threshold]
    
    return keep

# Função para calcular IoU
def calculate_iou(box1, box2):
    """Calcula IoU entre duas boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calcular interseção
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calcular união
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# Função para enviar predições para o Label Studio
def send_prediction_to_labelstudio(task_id, predictions, headers):
    url = f"https://internal-label.cogtive.com/api/predictions/"
    data = {
        "task": task_id,
        "model_version": "yolov5_onnx_kavak",
        "result": predictions
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code >= 200 and response.status_code < 300:
        return True, response.json()
    else:
        return False, f"Erro {response.status_code}: {response.text}"

# Verificar se o modelo existe
if not os.path.exists(model_path):
    print(f"Erro: Modelo não encontrado no caminho {model_path}")
    exit(1)

# Carregar o modelo ONNX
ort_session = ort.InferenceSession(model_path)

# Filtrar tarefas sem predições
tasks_without_predictions = [t for t in all_tasks if not t.get("predictions")]
print(f"Encontradas {len(tasks_without_predictions)} tarefas sem predições.")

if tasks_without_predictions:
    # Perguntar se deseja processar as tarefas
    if True:
        # Processar tarefas sem predições
        success_count = 0
        failed_count = 0
        
        # Em modo debug, processar apenas a primeira tarefa
        if DEBUG_MODE:
            tasks_to_process = tasks_without_predictions[:1]
            print("\n=== MODO DEBUG ATIVADO - Processando apenas a primeira tarefa ===\n")
        else:
            tasks_to_process = tasks_without_predictions
            pbar_predict = tqdm(total=len(tasks_to_process), desc="Processando predições")

        for task in tasks_to_process:
            task_id = task["id"]
            image_url = task.get("data", {}).get("image")
            
            print(f"\nProcessando tarefa {task_id}...")
            print(f"URL da imagem: {image_url}")
            
            if not image_url:
                print(f"Task {task_id}: Imagem não encontrada.")
                failed_count += 1
                if not DEBUG_MODE:
                    pbar_predict.update(1)
                continue
            
            # Se a URL for relativa, adicionar o domínio
            if not image_url.startswith("http"):
                image_url = f"https://internal-label.cogtive.com{image_url}"
            
            try:
                print(f"Baixando imagem de {image_url}...")
                # Baixar a imagem
                response = requests.get(image_url, headers=headers)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                
                # Salvar temporariamente a imagem para processamento
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path)
                
                # Pré-processar a imagem
                input_tensor, original_width, original_height, scale = preprocess_image(temp_path, input_size=640)
                
                # Fazer inferência com o modelo ONNX
                print(f"Executando inferência ONNX...")
                outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
                
                # Remover arquivo temporário
                os.unlink(temp_path)
                
                # Pós-processar as saídas
                print("Processando detecções...")
                detections = postprocess_yolo_output(outputs[0], original_width, original_height, scale)
                
                print(f"Encontradas {len(detections)} detecções")
                
                # Converter resultados para formato Label Studio
                annotations = []
                for i, detection in enumerate(detections):
                    x1, y1, x2, y2 = detection['box']
                    class_name = detection['class_name'].upper()  # Maiúsculo para Label Studio
                    confidence = detection['confidence']
                    
                    # Calcular coordenadas percentuais para Label Studio
                    x_percent = (x1 / original_width) * 100
                    y_percent = (y1 / original_height) * 100
                    width_percent = ((x2 - x1) / original_width) * 100
                    height_percent = ((y2 - y1) / original_height) * 100
                    
                    print(f"Detecção {i+1}: {class_name} ({confidence:.2f}) - Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                    
                    # Formato Label Studio para bounding box
                    label_studio_format = {
                        "id": f"result_{i+1}",
                        "type": "rectanglelabels",
                        "value": {
                            "x": max(0, x_percent),
                            "y": max(0, y_percent),
                            "width": width_percent,
                            "height": height_percent,
                            "rectanglelabels": [class_name]
                        },
                        "score": confidence,
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": original_width,
                        "original_height": original_height
                    }
                    
                    annotations.append(label_studio_format)
                
                # No modo debug, mostrar as anotações formatadas
                if DEBUG_MODE:
                    print("\n==== ANOTAÇÕES FORMATADAS PARA LABEL STUDIO ====")
                    import json
                    print(json.dumps(annotations, indent=2))
                
                # Enviar predições para o Label Studio
                print(f"Enviando predições para a tarefa {task_id}...")
                success, response = send_prediction_to_labelstudio(task_id, annotations, headers)
                
                if success:
                    print(f"Task {task_id}: Predições enviadas com sucesso.")
                    success_count += 1
                else:
                    print(f"Task {task_id}: Falha ao enviar predições: {response}")
                    failed_count += 1
                
                # Pequena pausa para não sobrecarregar a API
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Task {task_id}: Erro ao processar: {str(e)}")
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()  # Mostrar traceback completo em modo debug
                failed_count += 1
            
            if not DEBUG_MODE:
                pbar_predict.update(1)
        
        if not DEBUG_MODE:
            pbar_predict.close()
            
        # Mostrar resultados finais
        print("\nResultados do processamento:")
        print(f"Total de tarefas processadas: {len(tasks_to_process)}")
        print(f"Sucesso: {success_count}")
        print(f"Falha: {failed_count}")
else:
    print("Não há tarefas sem predições para processar.")

# Opcionalmente, permitir salvar os resultados em um arquivo para análise futura
save_results = input("Deseja salvar os resultados em um arquivo JSON? (s/n): ")
if save_results.lower() == "s":
    output_file = "tasks_results.json"
    with open(output_file, "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"Resultados salvos em {output_file}")
