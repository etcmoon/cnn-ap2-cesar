# Aumento de dados com OpenCV: a cada lote do treino, cada rosto é sorteado com mudanças.
# Como as cópias são novas a cada época, a rede nunca vê a mesma imagem duas vezes e não decora as fotos.
# As mudanças imitam o que muda de uma captura para outra (luz de um lado, câmera pior, mão no rosto),
# para a rede não usar esses detalhes como "pista" de quem é a pessoa.
import cv2
import numpy as np

from src.config import IMG_SIZE
from src.preprocessamento import aplicar_mascara_oval


def aumentar(img, gerador):
    # Recebe um rosto (64, 64, 1) com valores 0-1 e devolve uma versão modificada aleatoriamente
    t = IMG_SIZE

    # 1. Rotação (±10°), zoom (90% a 110%) e deslocamento (±4 px) numa única transformação afim
    matriz = cv2.getRotationMatrix2D((t / 2, t / 2), gerador.uniform(-10, 10), gerador.uniform(0.9, 1.1))
    matriz[:, 2] += gerador.uniform(-4, 4, size=2)
    img = cv2.warpAffine(img[:, :, 0], matriz, (t, t), borderMode=cv2.BORDER_REFLECT)

    # 2. Espelhamento horizontal (50% de chance): o rosto virado para o outro lado
    if gerador.random() < 0.5:
        img = img[:, ::-1]

    # 3. Iluminação: gama (foto mais clara ou escura), contraste, brilho e uma luz mais forte de um lado do rosto
    img = np.clip(img, 0, 1) ** gerador.uniform(0.7, 1.4)
    img = (img - 0.5) * gerador.uniform(0.8, 1.2) + 0.5 + gerador.uniform(-0.1, 0.1)
    angulo = gerador.uniform(0, 2 * np.pi)
    y, x = np.mgrid[0:t, 0:t] / (t - 1) - 0.5
    img = img + gerador.uniform(0, 0.25) * (x * np.cos(angulo) + y * np.sin(angulo))

    # 4. Câmera pior: desfoque (30% de chance) e ruído
    if gerador.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img + gerador.normal(0, 0.02, img.shape)

    # 5. Um pedaço do rosto tampado (50% de chance): a rede aprende a não depender de uma parte só
    if gerador.random() < 0.5:
        lado = int(gerador.integers(t // 6, t // 3))
        y0, x0 = gerador.integers(0, t - lado, size=2)
        img = np.array(img)
        img[y0:y0 + lado, x0:x0 + lado] = 0.5

    # 6. Máscara oval de novo: o fundo volta a ser o mesmo cinza das fotos reais
    img = aplicar_mascara_oval(np.clip(img, 0, 1), cinza=0.5)
    return img[:, :, np.newaxis].astype(np.float32)
