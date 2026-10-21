# Métricas por integrante, a partir da matriz de confusão (que vem do scikit-learn).
import numpy as np


def calcular_metricas(matriz):
    # Tira VP, FP, FN e VN da matriz (um valor por integrante) e calcula as métricas.
    # Linhas = integrante real, colunas = integrante previsto.
    VP = np.diag(matriz)                      # diagonal: era X e a rede disse X
    FP = matriz.sum(axis=0) - VP              # resto da coluna: a rede disse X, mas era outra pessoa
    FN = matriz.sum(axis=1) - VP              # resto da linha: era X, mas a rede disse outra pessoa
    VN = matriz.sum() - VP - FP - FN          # todo o resto da matriz

    # np.maximum evita divisão por zero quando alguém não aparece no teste
    precisao       = VP / np.maximum(VP + FP, 1)
    recall         = VP / np.maximum(VP + FN, 1)
    f1             = 2 * precisao * recall / np.maximum(precisao + recall, 1e-9)
    especificidade = VN / np.maximum(VN + FP, 1)
    acuracia       = VP.sum() / max(matriz.sum(), 1)

    return {
        "VP": VP, "FP": FP, "FN": FN, "VN": VN,
        "precisao": precisao, "recall": recall, "f1": f1, "especificidade": especificidade,
        "acuracia": acuracia,
    }
