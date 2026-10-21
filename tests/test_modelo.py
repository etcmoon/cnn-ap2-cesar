# Testes da CNN feita à mão: os gradientes da retropropagação batem com a derivada numérica?
# Rodar na raiz do repositório com:  pytest
import numpy as np

from src.modelo import (iniciar_pesos, propagar, retropropagar, softmax_entropia, passo_adam, prever,
                        salvar_modelo, carregar_modelo)


def rede_pequena():
    # Rede com imagens 8x8 (3 poolings: 8 -> 1) e 3 classes, em float64 para a derivada numérica ser precisa
    gerador = np.random.default_rng(0)
    params = {k: v.astype(np.float64) for k, v in iniciar_pesos(3, gerador, tamanho=8).items()}
    params = {k: v + gerador.normal(0, 0.1, v.shape) for k, v in params.items()}   # vieses != 0 também
    x = gerador.random((4, 8, 8, 1))
    y = np.array([0, 1, 2, 1])
    return params, x, y


def test_gradientes_batem_com_a_derivada_numerica():
    # Para alguns pesos de cada camada: (erro(w + h) - erro(w - h)) / 2h deve ser igual ao gradiente calculado
    params, x, y = rede_pequena()
    perda = lambda: softmax_entropia(propagar(params, x)[0], y)[0]
    pontuacoes, cache = propagar(params, x)
    grads = retropropagar(params, cache, softmax_entropia(pontuacoes, y)[1])
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
    # Se forward, backward e Adam estão certos, a rede consegue decorar 4 imagens (o erro cai quase a zero)
    params, x, y = rede_pequena()
    estado = {}
    for _ in range(200):
        pontuacoes, cache = propagar(params, x)
        perda, d = softmax_entropia(pontuacoes, y)
        passo_adam(params, retropropagar(params, cache, d), estado, taxa=0.01)
    assert perda < 0.05
    assert (prever(params, x).argmax(axis=1) == y).all()


def test_salvar_e_carregar_devolvem_a_mesma_rede(tmp_path):
    params, x, _ = rede_pequena()
    caminho = str(tmp_path / "modelo.npz")
    salvar_modelo(params, ["ana", "bia", "caio"], caminho)
    params2, classes = carregar_modelo(caminho)
    assert classes == ["ana", "bia", "caio"]
    np.testing.assert_allclose(prever(params2, x), prever(params, x))
