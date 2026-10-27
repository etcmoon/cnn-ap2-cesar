# Emoção de cada rosto, com um modelo pronto (biblioteca DeepFace). A identidade continua sendo da NOSSA CNN;
# a emoção é a única parte que usa uma rede de terceiros, como o trabalho permite.
import os
import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # sem as mensagens do TensorFlow no terminal
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from deepface import DeepFace                          # noqa: E402 (as variáveis acima precisam vir antes)

from src.preprocessamento import alinhar               # noqa: E402

# Nomes do DeepFace -> português, na ordem em que o site mostra
EMOCOES = {"happy": "alegre", "sad": "triste", "angry": "bravo", "fear": "com medo",
           "surprise": "surpreso", "disgust": "nojo", "neutral": "neutro"}
TAMANHO = 224     # lado do recorte colorido entregue ao DeepFace (ele reduz para 48x48 em cinza)


def emocoes(quadro, rostos):
    # Recebe a foto RGB e os rostos achados pelo YuNet e devolve, para cada rosto, a probabilidade (0-1) de cada
    # emoção, na ordem de EMOCOES. O rosto vai já recortado e alinhado ("skip"): o DeepFace não detecta de novo.
    if not rostos:
        return []
    recortes = [cv2.cvtColor(alinhar(quadro, r, TAMANHO), cv2.COLOR_RGB2BGR) for r in rostos]   # DeepFace usa BGR
    analises = DeepFace.analyze(recortes, actions=["emotion"], detector_backend="skip",
                                enforce_detection=False, silent=True)
    return [[float(a[0]["emotion"][chave]) / 100 for chave in EMOCOES] for a in analises]


def aquecer():
    # A primeira análise carrega o modelo (e baixa os pesos na 1ª vez): fazemos isso na abertura do serviço
    emocoes(np.zeros((TAMANHO, TAMANHO, 3), np.uint8), [np.array([0, 0, 224, 224, 80, 90, 144, 90, 0, 0, 0, 0, 0, 0.])])
