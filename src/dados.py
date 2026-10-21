# Leitura das fotos do dataset (a gravação das fotos coletadas no site é feita pelo servidor Node).
import os
import io
import base64
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from src.config import DATASET_DIR

register_heif_opener()    # ensina o PIL a abrir fotos .heic/.heif (formato padrão do iPhone)

EXTENSOES = (".jpg", ".jpeg", ".png", ".heic", ".heif")    # tipos de arquivo aceitos


def ler_imagem(caminho):
    # Abre o arquivo e devolve uma matriz (altura, largura, 3) com os valores R, G, B de cada pixel.
    # Celulares às vezes salvam a foto "deitada" e anotam a rotação certa no EXIF: o exif_transpose desvira a foto.
    return np.array(ImageOps.exif_transpose(Image.open(caminho)).convert("RGB"))


def listar_dataset(pasta=DATASET_DIR):
    # Percorre as pastas e devolve: nomes dos integrantes, caminho de cada foto e o número da pessoa (0, 1, 2...).
    # Cada subpasta é um integrante (ignoramos pastas ocultas e de sistema, que começam com "." ou "_").
    classes = sorted(p for p in os.listdir(pasta)
                     if os.path.isdir(os.path.join(pasta, p)) and p[0] not in "._")
    caminhos, rotulos = [], []
    for numero, nome in enumerate(classes):
        subpasta = os.path.join(pasta, nome)
        for arquivo in sorted(os.listdir(subpasta)):         # sorted = ordem em que as fotos foram tiradas
            if arquivo.lower().endswith(EXTENSOES):
                caminhos.append(os.path.join(subpasta, arquivo))
                rotulos.append(numero)
    return classes, caminhos, np.array(rotulos)


def imagem_de_dataurl(texto):
    # Converte o texto "data:image/jpeg;base64,..." enviado pelo navegador em uma matriz de pixels RGB
    bytes_img = base64.b64decode(texto.split(",")[1])            # tira o cabeçalho "data:image/jpeg;base64,"
    return np.array(Image.open(io.BytesIO(bytes_img)).convert("RGB"))

