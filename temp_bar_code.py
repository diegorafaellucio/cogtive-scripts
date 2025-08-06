import cv2
from pdf417decoder.decoder import Decoder

# Caminho da imagem
image_path = '/home/diego/Downloads/IMG_0878.jpg'

# Carrega e converte a imagem
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Instancia o decodificador
decoder = Decoder()
results = decoder.decode(gray)

# Exibe resultados
if not results:
    print("Nenhum código PDF417 detectado.")
else:
    for i, result in enumerate(results):
        print(f"[PDF417 #{i+1}]")
        print(result['text'])
