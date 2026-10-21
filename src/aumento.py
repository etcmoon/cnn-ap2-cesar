# Aumento de dados com OpenCV: a cada lote do treino, cada rosto é sorteado com pequenas mudanças.
# Como as cópias são novas a cada época, a rede nunca vê a mesma imagem duas vezes e não decora as fotos.
import cv2
import numpy as np

from src.config import IMG_SIZE
from src.preprocessamento import aplicar_mascara_oval


def aumentar(img, gerador):
    # Recebe um rosto (64, 64, 1) com valores 0-1 e devolve uma versão modificada aleatoriamente

    # 1. Rotação (±10°), zoom (90% a 110%) e deslocamento (±4 px) numa única transformação afim
    matriz = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), gerador.uniform(-10, 10), gerador.uniform(0.9, 1.1))
    matriz[:, 2] += gerador.uniform(-4, 4, size=2)
    img = cv2.warpAffine(img[:, :, 0], matriz, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)

    # 2. Espelhamento horizontal (50% de chance): o rosto virado para o outro lado
    if gerador.random() < 0.5:
        img = img[:, ::-1]

    # 3. Contraste e brilho (iluminação diferente) e ruído (câmera de qualidade pior)
    img = (img - 0.5) * gerador.uniform(0.8, 1.2) + 0.5 + gerador.uniform(-0.1, 0.1)
    img = img + gerador.normal(0, 0.02, img.shape)

    # 4. Máscara oval de novo: o fundo volta a ser o mesmo cinza das fotos reais, depois de girar e mexer no brilho
    img = aplicar_mascara_oval(np.clip(img, 0, 1), cinza=0.5)
    return img[:, :, np.newaxis].astype(np.float32)
