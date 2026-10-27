# Testes do pré-processamento: alinhamento pelos olhos, formato e faixa da saída, foto sem rosto, máscara oval
# e aumento de dados.
# Rodar na raiz do repositório com:  pytest
import numpy as np
import pytest

from src.config import IMG_SIZE
from src.preprocessamento import (preprocessar, tratar_rosto, alinhar, mascara_oval, aplicar_mascara_oval,
                                  OLHO_Y, OLHO_X_ESQ, OLHO_X_DIR)
from src.aumento import aumentar


def rosto_falso(olho1, olho2):
    # Rosto no formato do YuNet: caixa (x, y, largura, altura) + 5 pontos; só os olhos importam para o alinhamento
    return np.array([0, 0, 100, 100, *olho1, *olho2, 0, 0, 0, 0, 0, 0], np.float32)


def test_alinhar_leva_os_olhos_sempre_para_o_mesmo_lugar():
    # Um ponto branco em cada olho de uma foto com a cabeça inclinada: depois de alinhar, os pontos ficam na
    # altura OLHO_Y e nas colunas OLHO_X_ESQ e OLHO_X_DIR, não importa a inclinação, o tamanho ou a ordem dos olhos
    for olho1, olho2 in [((200, 150), (300, 210)), ((420, 300), (380, 260)), ((50, 60), (90, 58))]:
        img = np.zeros((480, 640), np.float32)
        for x, y in (olho1, olho2):
            img[y - 1:y + 2, x - 1:x + 2] = 1
        alinhado = alinhar(img, rosto_falso(olho1, olho2), 200)
        ys, xs = np.nonzero(alinhado > 0.3)
        esquerda, direita = xs < 100, xs >= 100
        assert abs(ys.mean() - OLHO_Y * 200) < 2
        assert abs(xs[esquerda].mean() - OLHO_X_ESQ * 200) < 2 and abs(xs[direita].mean() - OLHO_X_DIR * 200) < 2


@pytest.mark.parametrize("olhos", [((300, 200), (360, 200)), ((0, 0), (20, 5)), ((630, 470), (600, 440))])
def test_tratar_rosto_devolve_64x64x1_float32_entre_0_e_1(olhos):
    # Rostos no meio, na borda e pequenos devem sair todos no formato que a CNN espera
    cinza = np.random.default_rng(0).integers(0, 256, (480, 640)).astype(np.uint8)
    final, etapas = tratar_rosto(cinza, rosto_falso(*olhos), mostrar_etapas=True)
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
