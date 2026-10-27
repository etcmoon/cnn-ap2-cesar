# Busca de hiperparâmetros (rode UMA vez, separada do treino): sorteia combinações, treina cada uma e compara na
# validação. A melhor vai para modelo/hiperparametros.json, que o treino normal (python -m src.treinar) passa a usar.
# Uso (na raiz do repositório):  python -m src.buscar_hiperparametros [quantidade de tentativas, padrão 12]
import os
import sys
import json
import time
import numpy as np

from src.config import MODELO_DIR, CAMINHO_HP, SEED, HP
from src.divisao import dividir_dados
from src.modelo import prever, decidir
from src.treinar import carregar_dados, treinar_modelo

# Valores sorteados em cada tentativa
OPCOES = {
    "taxa":    [0.0005, 0.001, 0.002],
    "dropout": [0.3, 0.5],
    "filtros": [[8, 16, 32], [16, 32, 64]],
    "densa":   [64, 128],
    "s":       [16.0, 32.0],
    "m":       [0.2, 0.35, 0.5],
    "l2":      [0.0, 0.0001],
}


def nota(cossenos, y, limiar):
    # Nota de uma combinação, na validação:
    #   acerto  = foto reconhecida como a pessoa certa E acima do limiar
    #   impostor = foto em que OUTRO integrante também passa do limiar (faria um desconhecido parecido ser aceito)
    # nota = acerto - impostor: premia rostos de cada pessoa juntos e longe dos outros, sem fotos de desconhecidos.
    acerto = np.mean(decidir(cossenos, limiar) == y)
    outros = cossenos.copy()
    outros[np.arange(len(y)), y] = -np.inf
    impostor = np.mean(outros.max(axis=1) >= limiar)
    return float(acerto - impostor), float(acerto), float(impostor)


def main():
    tentativas = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    gerador = np.random.default_rng(SEED)
    classes, _, X, y, sessoes = carregar_dados()                   # pré-processamento feito uma vez só
    idx_treino, idx_val, _ = dividir_dados(y, classes, sessoes)   # o teste fica guardado para o treino final

    combinacoes = [dict(HP)]                                        # a 1ª tentativa é o padrão, para comparar
    while len(combinacoes) < tentativas:
        hp = {chave: valores[gerador.integers(len(valores))] for chave, valores in OPCOES.items()}
        if hp not in combinacoes:
            combinacoes.append(hp)

    resultados = []
    for i, hp in enumerate(combinacoes, 1):
        print(f"\nTentativa {i}/{len(combinacoes)}: {json.dumps(hp)}")
        inicio = time.time()
        params, historico, limiar = treinar_modelo(X[idx_treino], y[idx_treino], X[idx_val], y[idx_val],
                                                   len(classes), hp, gerador)
        total, acerto, impostor = nota(prever(params, X[idx_val]), y[idx_val], limiar)
        resultados.append({"hp": hp, "nota": total, "acerto": acerto, "impostor": impostor, "limiar": limiar,
                           "epocas": len(historico["loss"]), "segundos": round(time.time() - inicio)})
        print(f"  nota {total:.3f} (acerto {acerto:.3f} - impostor {impostor:.3f}), limiar {limiar:.3f}")

    resultados.sort(key=lambda r: -r["nota"])
    print(f"\n{'nota':>6} {'acerto':>7} {'impostor':>9} {'épocas':>7}  hiperparâmetros")
    for r in resultados:
        print(f"{r['nota']:>6.3f} {r['acerto']:>7.3f} {r['impostor']:>9.3f} {r['epocas']:>7}  {json.dumps(r['hp'])}")

    os.makedirs(MODELO_DIR, exist_ok=True)
    with open(os.path.join(MODELO_DIR, "busca_hiperparametros.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    with open(CAMINHO_HP, "w", encoding="utf-8") as f:
        json.dump(resultados[0]["hp"], f, indent=2)
    print(f"\nMelhor combinação gravada em {CAMINHO_HP}: o próximo treino (python -m src.treinar ou o botão do "
          "site) já usa estes valores. Apague o arquivo para voltar ao HP do src/config.py.")


if __name__ == "__main__":
    main()
