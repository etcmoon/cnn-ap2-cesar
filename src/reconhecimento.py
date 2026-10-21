# Funções usadas pelo servidor para reconhecer quem está na webcam, e o Grad-CAM ("onde a rede olhou").
import cv2
import numpy as np

from src.config import IMG_SIZE
from src.modelo import propagar, retropropagar, prever
from src.preprocessamento import detectar_rostos, tratar_rosto


def reconhecer_quadro(params, quadro):
    # Acha todos os rostos do quadro da webcam, aplica o MESMO tratamento do treino em cada um e passa todos
    # pela CNN de uma vez. Devolve a caixa de cada rosto e a probabilidade de cada integrante (lista vazia = ninguém).
    cinza = cv2.cvtColor(quadro, cv2.COLOR_RGB2GRAY)
    rostos = detectar_rostos(cinza)
    if not rostos:
        return []
    probs = prever(params, np.array([tratar_rosto(cinza, caixa) for caixa in rostos]))
    return [{"x": x, "y": y, "lado": lado, "probabilidades": probs[i].tolist()}
            for i, (x, y, lado, _) in enumerate(rostos)]


def grad_cam(params, img):
    # Gera um mapa de calor (64x64, valores 0-1) das regiões da foto que mais pesaram na decisão.
    # Usamos a pontuação de antes do softmax: com quase 100% de certeza, a probabilidade quase não muda
    # e o gradiente sairia zerado.
    pontuacoes, cache = propagar(params, img[np.newaxis])
    classe = int(pontuacoes[0].argmax())                         # integrante previsto
    d = np.zeros_like(pontuacoes)
    d[0, classe] = 1                                             # derivamos só a pontuação dele
    gradientes = retropropagar(params, cache, d, ate="conv3")    # quanto a nota muda com cada valor dos mapas
    mapas = cache["convs"][2][1][0]                              # os 32 mapas da conv3 (16x16)
    importancia = gradientes[0].mean(axis=(0, 1))                # importância média de cada mapa
    calor = np.maximum((mapas * importancia).sum(axis=-1), 0)    # soma ponderada (só positivos)
    calor = calor / (calor.max() + 1e-9)                         # escala para 0-1
    return cv2.resize(calor, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR), classe   # 16x16 -> 64x64
