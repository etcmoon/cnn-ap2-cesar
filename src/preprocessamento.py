# Pré-processamento das fotos com OpenCV. Toda foto passa pelas mesmas etapas, no treino e no ao vivo:
# cinza -> detecção do rosto -> recorte -> 64x64 -> CLAHE -> suavização -> máscara oval -> escala 0-1.
import cv2
import numpy as np

from src.config import IMG_SIZE, MARGEM_ROSTO

# Detector de rostos Haar (Viola-Jones, 2001): um classificador clássico que vem pronto com o OpenCV.
# Não é rede neural nem reconhece QUEM é a pessoa; só diz ONDE há um rosto.
DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# CLAHE: equalização de histograma por regiões (4x4 blocos de 16 px), com limite de contraste 2.
# Ao contrário da equalização global, uma luz forte de um lado do rosto não escurece o outro lado.
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def detectar_rostos(cinza):
    # Devolve as caixas (x, y, lado, lado) dos rostos, do maior para o menor.
    # O detector roda numa cópia com no máximo 640 px (fotos de celular têm 4000 px: mesmo resultado, bem mais rápido).
    escala = min(1.0, 640 / max(cinza.shape))
    menor = cv2.resize(cinza, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else cinza
    lado_minimo = max(24, int(0.1 * min(menor.shape)))                  # ignora "rostos" minúsculos (falsos)
    rostos = DETECTOR.detectMultiScale(menor, scaleFactor=1.1, minNeighbors=5, minSize=(lado_minimo, lado_minimo))
    caixas = [tuple(int(v / escala) for v in caixa) for caixa in rostos]
    return remover_sobrepostas(sorted(caixas, key=lambda caixa: -caixa[2]))


def remover_sobrepostas(caixas, limite=0.5):
    # O detector às vezes acha o MESMO rosto duas vezes (o rosto inteiro e uma caixa menor na metade de baixo).
    # Percorre do maior para o menor e descarta a caixa que tem mais da metade da área dentro de uma já aceita.
    # Rostos de pessoas diferentes quase não se sobrepõem, então continuam todos.
    def fracao_dentro(menor, maior):
        x1, y1, l1, _ = menor
        x2, y2, l2, _ = maior
        largura = max(0, min(x1 + l1, x2 + l2) - max(x1, x2))
        altura = max(0, min(y1 + l1, y2 + l2) - max(y1, y2))
        return largura * altura / l1 ** 2
    aceitas = []
    for caixa in caixas:
        if all(fracao_dentro(caixa, outra) <= limite for outra in aceitas):
            aceitas.append(caixa)
    return aceitas


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


def tratar_rosto(cinza, caixa, mostrar_etapas=False):
    # Recorta a caixa do detector (com margem) e aplica as correções de imagem. Devolve (64, 64, 1) entre 0 e 1.
    x, y, lado, _ = caixa
    m = int(MARGEM_ROSTO * lado)
    e1 = cinza[max(0, y - m):y + lado + m, max(0, x - m):x + lado + m]                 # 1. recorte do rosto
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
    # Com mostrar_etapas=True, devolve também [cinza, recorte, 64x64, CLAHE, suavizada, com máscara].
    cinza = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)              # tons de cinza
    rostos = detectar_rostos(cinza)
    if not rostos:
        return (None, [cinza]) if mostrar_etapas else None
    if mostrar_etapas:
        final, etapas = tratar_rosto(cinza, rostos[0], mostrar_etapas=True)
        return final, [cinza] + etapas
    return tratar_rosto(cinza, rostos[0])
