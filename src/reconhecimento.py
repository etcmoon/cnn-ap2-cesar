# Funções usadas pelo servidor para reconhecer quem está na webcam, e o Grad-CAM ("onde a rede olhou").
import cv2
import numpy as np

from src.config import IMG_SIZE
from src.modelo import propagar, retropropagar, prever
from src.preprocessamento import detectar_rostos, tratar_rosto
from src.emocao import emocoes


def reconhecer_quadro(params, quadro):
    # Acha todos os rostos do quadro da webcam, aplica o MESMO tratamento do treino em cada um e passa todos
    # pela CNN de uma vez. Devolve, para cada rosto: a caixa, o cosseno com cada integrante e as emoções
    # (lista vazia = ninguém). A decisão "desconhecido" (cosseno abaixo do limiar) é feita no navegador.
    rostos = detectar_rostos(quadro)
    if not rostos:
        return []
    cinza = cv2.cvtColor(quadro, cv2.COLOR_RGB2GRAY)
    cossenos = prever(params, np.array([tratar_rosto(cinza, r) for r in rostos]))
    sentimentos = emocoes(quadro, rostos)
    return [{"x": int(r[0]), "y": int(r[1]), "largura": int(r[2]), "altura": int(r[3]),
             "similaridades": cossenos[i].tolist(), "emocoes": sentimentos[i]}
            for i, r in enumerate(rostos)]


def grad_cam(params, img):
    # Gera um mapa de calor (64x64, valores 0-1) das regiões da foto que mais pesaram na decisão.
    # Derivamos o cosseno do integrante escolhido em relação aos mapas da última convolução.
    cossenos, cache = propagar(params, img[np.newaxis])
    classe = int(cossenos[0].argmax())                           # integrante previsto
    d = np.zeros_like(cossenos)
    d[0, classe] = 1                                             # derivamos só o cosseno dele
    gradientes = retropropagar(params, cache, d, ate="conv3")    # quanto o cosseno muda com cada valor dos mapas
    mapas = cache["convs"][2][1][0]                              # os mapas da conv3 (16x16)
    importancia = gradientes[0].mean(axis=(0, 1))                # importância média de cada mapa
    calor = np.maximum((mapas * importancia).sum(axis=-1), 0)    # soma ponderada (só positivos)
    calor = calor / (calor.max() + 1e-9)                         # escala para 0-1
    return cv2.resize(calor, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR), classe   # 16x16 -> 64x64
