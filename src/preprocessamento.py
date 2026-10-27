# Pré-processamento das fotos com OpenCV. Toda foto passa pelas mesmas etapas, no treino e no ao vivo:
# detecção do rosto (YuNet) -> alinhamento pelos olhos -> 64x64 -> cinza -> CLAHE -> suavização -> máscara oval -> 0-1.
import cv2
import numpy as np

from src.config import IMG_SIZE, CAMINHO_YUNET

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # sem avisos do módulo dnn no terminal

# Detector de rostos YuNet: uma rede pequena que vem do OpenCV Zoo (arquivo .onnx em modelo/).
# Não reconhece QUEM é a pessoa; só diz ONDE há um rosto e onde estão os olhos, o nariz e a boca.
DETECTOR = cv2.FaceDetectorYN.create(CAMINHO_YUNET, "", (320, 320), score_threshold=0.8)

# CLAHE: equalização de histograma por regiões (4x4 blocos de 16 px), com limite de contraste 2.
# Ao contrário da equalização global, uma luz forte de um lado do rosto não escurece o outro lado.
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

# Onde os olhos ficam no rosto alinhado (frações do lado da imagem): sempre na mesma altura e distância
OLHO_Y, OLHO_X_ESQ, OLHO_X_DIR = 0.37, 0.35, 0.65
AMPLIADO = 4 * IMG_SIZE     # o alinhamento é feito em 256x256 e só depois reduzido (redução de qualidade, sem serrilhado)


def detectar_rostos(img):
    # Recebe a foto RGB e devolve os rostos, do maior para o menor. Cada rosto é uma lista de 14 números:
    # caixa (x, y, largura, altura) e 5 pontos (olho, olho, nariz, canto da boca, canto da boca), cada um (x, y).
    # O detector roda numa cópia com no máximo 640 px (fotos de celular têm 4000 px: mesmo resultado, bem mais rápido).
    escala = min(1.0, 640 / max(img.shape[:2]))
    menor = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else img
    DETECTOR.setInputSize((menor.shape[1], menor.shape[0]))
    _, rostos = DETECTOR.detect(cv2.cvtColor(np.ascontiguousarray(menor), cv2.COLOR_RGB2BGR))
    if rostos is None:
        return []
    lado_minimo = max(24, 0.08 * min(menor.shape[:2]))                   # ignora "rostos" minúsculos (falsos)
    rostos = [r[:14] / escala for r in rostos if min(r[2], r[3]) >= lado_minimo]
    return sorted(rostos, key=lambda r: -r[2] * r[3])


def alinhar(img, rosto, tamanho):
    # Gira, escala e recorta a foto para os dois olhos caírem sempre nos mesmos pixels (rosto "em pé" e do mesmo
    # tamanho). Funciona com foto cinza ou colorida. Devolve uma imagem tamanho x tamanho.
    olhos = sorted([rosto[4:6], rosto[6:8]], key=lambda p: p[0])         # olho da esquerda e da direita na foto
    (x1, y1), (x2, y2) = olhos
    angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))                    # inclinação da linha dos olhos
    escala = (OLHO_X_DIR - OLHO_X_ESQ) * tamanho / max(np.hypot(x2 - x1, y2 - y1), 1)
    centro = ((x1 + x2) / 2, (y1 + y2) / 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, escala)
    matriz[:, 2] += (tamanho / 2 - centro[0], OLHO_Y * tamanho - centro[1])   # leva o meio dos olhos ao lugar certo
    return cv2.warpAffine(img, matriz, (tamanho, tamanho), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def mascara_oval(tamanho, escala=1.0):
    # Peso de cada pixel de uma imagem tamanho x tamanho: 1 dentro de um oval do formato de um rosto, 0 fora,
    # com uma transição suave na borda. "escala" encolhe o oval.
    centro = (tamanho - 1) / 2
    y, x = np.mgrid[0:tamanho, 0:tamanho] - centro                         # distância de cada pixel ao centro
    r = np.sqrt((x / (0.40 * tamanho * escala)) ** 2 + (y / (0.48 * tamanho * escala)) ** 2)   # r = 1 na borda do oval
    return np.clip((1 - r) / 0.12, 0, 1)                                    # 1 até r = 0,88; cai até 0 em r = 1


def aplicar_mascara_oval(img, cinza=127.5):
    # Mantém o oval do rosto e troca os cantos (fundo, cabelo, orelhas) por um cinza neutro
    m = mascara_oval(img.shape[0])
    return img * m + cinza * (1 - m)                                        # rosto onde m = 1, cinza onde m = 0


def tratar_rosto(cinza, rosto, mostrar_etapas=False):
    # Alinha o rosto pelos olhos e aplica as correções de imagem. Devolve (64, 64, 1) entre 0 e 1.
    e1 = alinhar(cinza, rosto, AMPLIADO)                                              # 1. rosto alinhado
    e2 = cv2.resize(e1, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)           # 2. 64x64 (média da área)
    e3 = CLAHE.apply(e2)                                                              # 3. contraste local (CLAHE)
    e4 = cv2.GaussianBlur(e3, (3, 3), 0)                                              # 4. suaviza o ruído
    e5 = aplicar_mascara_oval(e4.astype(np.float32))                                  # 5. máscara oval
    final = (e5 / 255.0).reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)           # 6. escala 0-1
    if mostrar_etapas:
        return final, [e1, e2, e3, e4, e5]
    return final


def preprocessar(img, mostrar_etapas=False):
    # Foto RGB -> entrada da CNN, usando o MAIOR rosto da foto. Sem rosto, devolve None.
    # Com mostrar_etapas=True, devolve também [cinza, alinhado, 64x64, CLAHE, suavizada, com máscara].
    img = np.ascontiguousarray(img)
    cinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                                     # tons de cinza
    rostos = detectar_rostos(img)
    if not rostos:
        return (None, [cinza]) if mostrar_etapas else None
    if mostrar_etapas:
        final, etapas = tratar_rosto(cinza, rostos[0], mostrar_etapas=True)
        return final, [cinza] + etapas
    return tratar_rosto(cinza, rostos[0])
