# Divisão das fotos em treino (~70%), validação (~15%) e teste (~15%).
# Fotos da MESMA captura (mesma luz, roupa, lugar e expressão) são quase idênticas. Se o teste tiver fotos da mesma
# captura que o treino, a rede pode "decorar a captura" e mesmo assim tirar 100%: o overfitting fica escondido.
# Por isso, com 3 capturas ou mais por pessoa, a validação e o teste usam CAPTURAS INTEIRAS que o treino nunca viu.
# Com menos capturas, sorteamos blocos de fotos seguidas (melhor que fotos soltas, mas ainda otimista).
import numpy as np

from src.config import SEED


def dividir_dados(rotulos, classes, sessoes):
    # Devolve os índices das fotos de treino, validação e teste.
    # A divisão é feita por integrante, para que todos apareçam nos três conjuntos.
    gerador = np.random.default_rng(SEED)
    treino, validacao, teste = [], [], []

    for classe in range(len(classes)):
        indices = np.where(rotulos == classe)[0]                          # fotos desta pessoa, em ordem de captura
        assert len(indices) >= 3, f"{classes[classe]} tem só {len(indices)} foto(s): são necessárias pelo menos 3"

        capturas = np.unique(sessoes[indices])
        if len(capturas) >= 3:
            grupos = [indices[sessoes[indices] == s] for s in gerador.permutation(capturas)]
        else:
            print(f"  Aviso: {classes[classe]} tem só {len(capturas)} captura(s); a validação e o teste usam blocos "
                  "de fotos seguidas da mesma captura e ficam otimistas. Façam 3 capturas ou mais, em lugares diferentes.")
            # Blocos de até 10 fotos seguidas. Com poucas fotos, os blocos diminuem para formar pelo menos ~7 blocos
            tamanho_bloco = min(10, max(1, len(indices) // 7))
            grupos = [indices[i:i + tamanho_bloco] for i in range(0, len(indices), tamanho_bloco)]
            grupos = [grupos[i] for i in gerador.permutation(len(grupos))]  # embaralha a ordem dos blocos

        n = max(1, round(0.15 * len(grupos)))                             # ~15% para teste e ~15% para validação
        teste     += grupos[:n]
        validacao += grupos[n:2 * n]
        treino    += grupos[2 * n:]                                       # o restante (~70%) para treino

    return np.concatenate(treino), np.concatenate(validacao), np.concatenate(teste)
