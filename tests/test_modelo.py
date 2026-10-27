# Testes da CNN feita à mão: os gradientes da retropropagação batem com a derivada numérica?
# Rodar na raiz do repositório com:  pytest
import numpy as np

from src.config import HP
from src.modelo import (iniciar_pesos, propagar, retropropagar, cosface_entropia, passo_adam, prever,
                        calibrar_limiar, decidir, salvar_modelo, carregar_modelo)

S, M = 16.0, 0.35


def rede_pequena():
    # Rede com imagens 8x8 (3 poolings: 8 -> 1) e 3 classes, em float64 para a derivada numérica ser precisa
    gerador = np.random.default_rng(0)
    params = {k: v.astype(np.float64) for k, v in iniciar_pesos(3, gerador, {**HP, "densa": 16}, tamanho=8).items()}
    params = {k: v + gerador.normal(0, 0.1, v.shape) for k, v in params.items()}   # vieses != 0 também
    x = gerador.random((4, 8, 8, 1))
    y = np.array([0, 1, 2, 1])
    return params, x, y


def test_gradientes_batem_com_a_derivada_numerica():
    # Para alguns pesos de cada camada: (erro(w + h) - erro(w - h)) / 2h deve ser igual ao gradiente calculado.
    # Inclui a normalização do embedding e dos "rostos médios" (cabeça de cosseno do CosFace).
    params, x, y = rede_pequena()
    perda = lambda: cosface_entropia(propagar(params, x)[0], y, S, M)[0]
    cossenos, cache = propagar(params, x)
    grads = retropropagar(params, cache, cosface_entropia(cossenos, y, S, M)[1])
    gerador = np.random.default_rng(1)
    for nome, w in params.items():
        for _ in range(5):
            i = tuple(gerador.integers(0, s) for s in w.shape)
            original = w[i]
            w[i] = original + 1e-5; mais = perda()
            w[i] = original - 1e-5; menos = perda()
            w[i] = original
            np.testing.assert_allclose(grads[nome][i], (mais - menos) / 2e-5, rtol=1e-4, atol=1e-8, err_msg=nome)


def test_rede_decora_um_lote_pequeno():
    # Se forward, backward e Adam estão certos, a rede consegue decorar 4 imagens, com folga maior que a margem
    params, x, y = rede_pequena()
    estado = {}
    for _ in range(300):
        cossenos, cache = propagar(params, x)
        perda, d = cosface_entropia(cossenos, y, S, M)
        passo_adam(params, retropropagar(params, cache, d), estado, taxa=0.01)
    assert perda < 0.1
    assert (prever(params, x).argmax(axis=1) == y).all()


def test_limiar_e_decisao_do_desconhecido():
    cossenos = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.35], [0.7, 0.6]])
    y = np.array([0, 1, 1, 0])
    assert calibrar_limiar(cossenos, y, 0) == 0.35                   # percentil 0 = menor cosseno verdadeiro
    assert list(decidir(cossenos, 0.5)) == [0, 1, -1, 0]             # 0.35 não passa do limiar: desconhecido


def test_salvar_e_carregar_devolvem_a_mesma_rede(tmp_path):
    params, x, _ = rede_pequena()
    caminho = str(tmp_path / "modelo.npz")
    salvar_modelo(params, ["ana", "bia", "caio"], 0.42, caminho)
    params2, classes, limiar = carregar_modelo(caminho)
    assert classes == ["ana", "bia", "caio"] and limiar == 0.42
    np.testing.assert_allclose(prever(params2, x), prever(params, x))
