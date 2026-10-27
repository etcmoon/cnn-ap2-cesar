# Testes das métricas com uma matriz de confusão pequena, conferida à mão.
# Rodar na raiz do repositório com:  pytest
import numpy as np

from src.metricas import calcular_metricas

# Matriz montada à mão (linhas = real, colunas = previsto), 16 fotos no total:
#            prev 0  prev 1  prev 2
# real 0  [    5,      1,      0  ]   -> linha soma 6
# real 1  [    2,      3,      1  ]   -> linha soma 6
# real 2  [    0,      0,      4  ]   -> linha soma 4
#   colunas somam: 7, 4, 5
MATRIZ = np.array([[5, 1, 0],
                   [2, 3, 1],
                   [0, 0, 4]])


def test_vp_fp_fn_vn():
    m = calcular_metricas(MATRIZ)
    assert list(m["VP"]) == [5, 3, 4]            # diagonal
    assert list(m["FP"]) == [2, 1, 1]            # coluna menos a diagonal: 7-5, 4-3, 5-4
    assert list(m["FN"]) == [1, 3, 0]            # linha menos a diagonal: 6-5, 6-3, 4-4
    assert list(m["VN"]) == [8, 9, 11]           # 16 - VP - FP - FN


def test_precisao_recall_f1_especificidade_acuracia():
    m = calcular_metricas(MATRIZ)
    np.testing.assert_allclose(m["precisao"], [5 / 7, 3 / 4, 4 / 5])          # VP / (VP + FP)
    np.testing.assert_allclose(m["recall"], [5 / 6, 3 / 6, 4 / 4])            # VP / (VP + FN)
    np.testing.assert_allclose(m["f1"], [10 / 13, 0.6, 8 / 9])                # 2·p·r / (p + r)
    np.testing.assert_allclose(m["especificidade"], [8 / 10, 9 / 10, 11 / 12])  # VN / (VN + FP)
    assert m["acuracia"] == 12 / 16                                            # diagonal / total


def test_integrante_sem_fotos_nao_divide_por_zero():
    # Se alguém não aparece no teste nem é previsto, as métricas dele ficam 0 (e não NaN)
    m = calcular_metricas(np.array([[3, 0], [0, 0]]))
    assert not np.isnan(m["precisao"]).any() and not np.isnan(m["f1"]).any()


def test_coluna_desconhecido_conta_como_fn_e_nao_como_fp():
    # 2 integrantes + coluna "desconhecido": ana teve 1 foto rejeitada, bia teve 2
    m = calcular_metricas(np.array([[4, 0, 1],
                                    [1, 2, 2]]))
    assert list(m["VP"]) == [4, 2] and list(m["FP"]) == [1, 0] and list(m["FN"]) == [1, 3]
    assert m["acuracia"] == 6 / 10                                             # rejeitadas também são erros
