# Testes do pré-processamento: formato e faixa da saída, foto sem rosto, máscara oval e aumento de dados.
# Rodar na raiz do repositório com:  pytest
import numpy as np
import pytest

from src.config import IMG_SIZE
from src.preprocessamento import preprocessar, tratar_rosto, mascara_oval, aplicar_mascara_oval
from src.aumento import aumentar


@pytest.mark.parametrize("caixa", [(100, 50, 200, 200), (0, 0, 480, 480), (600, 400, 100, 100), (5, 5, 10, 10)])
def test_tratar_rosto_devolve_64x64x1_float32_entre_0_e_1(caixa):
    # Caixas no meio, na borda e pequenas devem sair todas no formato que a CNN espera
    cinza = np.random.default_rng(0).integers(0, 256, (480, 640)).astype(np.uint8)
    final, etapas = tratar_rosto(cinza, caixa, mostrar_etapas=True)
    assert final.shape == (IMG_SIZE, IMG_SIZE, 1) and final.dtype == np.float32
    assert final.min() >= 0 and final.max() <= 1
    assert len(etapas) == 5 and etapas[1].shape == (IMG_SIZE, IMG_SIZE)


def test_foto_sem_rosto_devolve_none():
    # Uma parede lisa não tem rosto: o pré-processamento avisa com None (no ao vivo, vira "ninguém")
    parede = np.full((480, 640, 3), 128, np.uint8)
    assert preprocessar(parede) is None
    final, etapas = preprocessar(parede, mostrar_etapas=True)
    assert final is None and etapas[0].shape == (480, 640)


def test_mascara_oval_mantem_o_centro_e_apaga_os_cantos():
    m = mascara_oval(IMG_SIZE)
    assert m[IMG_SIZE // 2, IMG_SIZE // 2] == 1                 # centro (rosto) intacto
    assert m[0, 0] == m[0, -1] == m[-1, 0] == m[-1, -1] == 0    # cantos (fundo) apagados
    assert mascara_oval(IMG_SIZE, escala=0.7).sum() < m.sum()   # escala menor = oval menor
    img = np.full((IMG_SIZE, IMG_SIZE), 255.0)
    resultado = aplicar_mascara_oval(img)
    assert resultado[0, 0] == 127.5 and resultado[IMG_SIZE // 2, IMG_SIZE // 2] == 255


def test_aumentar_mantem_formato_e_o_fundo_cinza():
    img = np.random.default_rng(0).random((IMG_SIZE, IMG_SIZE, 1)).astype(np.float32)
    copia = aumentar(img, np.random.default_rng(1))
    assert copia.shape == img.shape and copia.dtype == np.float32
    assert copia.min() >= 0 and copia.max() <= 1
    assert copia[0, 0, 0] == 0.5                                # cantos continuam o cinza da máscara
