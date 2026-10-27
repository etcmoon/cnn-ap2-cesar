# Script principal: executa o pipeline completo.
# Leitura -> pré-processamento -> divisão -> CNN -> treino -> avaliação -> teste do integrante de fora -> gravação.
# Uso (na raiz do repositório):  python -m src.treinar
# Os hiperparâmetros vêm de modelo/hiperparametros.json (gravado por python -m src.buscar_hiperparametros)
# ou, se ele não existir, do HP do src/config.py.
import os
import json
import time
import numpy as np
from sklearn.metrics import confusion_matrix

from src.config import (DATASET_DIR, MODELO_DIR, CAMINHO_MODELO, CAMINHO_RESULTADOS, EPOCAS, PACIENCIA, BATCH, SEED,
                        PERCENTIL, hiperparametros)
from src.dados import ler_imagem, listar_dataset, sessao_da_foto
from src.preprocessamento import preprocessar
from src.divisao import dividir_dados
from src.aumento import aumentar
from src.modelo import (iniciar_pesos, propagar, retropropagar, cosface_entropia, passo_adam, prever, calibrar_limiar,
                        decidir, resumo_camadas, salvar_modelo)
from src.metricas import calcular_metricas


def carregar_dados():
    # Lê e pré-processa todas as fotos (mesma função usada no reconhecimento ao vivo).
    # Fotos em que o detector não acha rosto ficam de fora (de costas, borradas, rosto cortado...).
    print("Etapa 1/8: leitura das fotos")
    classes, caminhos, rotulos = listar_dataset(DATASET_DIR)
    assert len(classes) >= 2, f"São necessários pelo menos 2 integrantes com fotos em {DATASET_DIR}."

    print("Etapa 2/8: pré-processamento (detecção do rosto, alinhamento pelos olhos, CLAHE, suavização)")
    X, usadas = [], []
    for i, caminho in enumerate(caminhos):
        rosto = preprocessar(ler_imagem(caminho))
        if rosto is not None:
            X.append(rosto)
            usadas.append(i)
    caminhos, y, X = [caminhos[i] for i in usadas], rotulos[usadas], np.array(X)
    sessoes = np.array([sessao_da_foto(c) for c in caminhos])
    print("Integrantes:")
    for numero, nome in enumerate(classes):
        sem_rosto = np.sum(rotulos == numero) - np.sum(y == numero)
        print(f"  {numero} = {nome}: {np.sum(y == numero)} fotos em {len(np.unique(sessoes[y == numero]))} captura(s)"
              f" ({sem_rosto} sem rosto detectado, ignoradas)")
    print("X:", X.shape, "-> (quantidade de fotos, altura, largura, canais)")
    return classes, rotulos, X, y, sessoes


def treinar_rede(params, X_treino, y_treino, X_val, y_val, hp, gerador):
    # O laço do treino: a cada época, embaralha as fotos e, lote a lote, aumenta -> propaga -> calcula o erro
    # -> retropropaga -> ajusta os pesos (Adam). No fim da época, mede o erro na validação (fotos reais, sem aumento).
    # Parada antecipada: se o erro na validação não melhorar por PACIENCIA épocas, para e volta aos melhores pesos.
    historico = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    estado_adam, melhor_erro, melhores, sem_melhora = {}, np.inf, None, 0
    for epoca in range(1, EPOCAS + 1):
        inicio = time.time()
        ordem = gerador.permutation(len(X_treino))
        soma_erro, acertos = 0.0, 0
        for i in range(0, len(ordem), BATCH):
            lote = ordem[i:i + BATCH]
            x = np.array([aumentar(img, gerador) for img in X_treino[lote]])     # cópias novas a cada época
            cossenos, cache = propagar(params, x, hp["dropout"], gerador)        # gerador = dropout ligado
            erro, d_cossenos = cosface_entropia(cossenos, y_treino[lote], hp["s"], hp["m"])
            passo_adam(params, retropropagar(params, cache, d_cossenos), estado_adam, hp["taxa"], hp["l2"])
            soma_erro += float(erro) * len(lote)
            acertos += int((cossenos.argmax(axis=1) == y_treino[lote]).sum())

        cos_val = prever(params, X_val)
        historico["loss"].append(soma_erro / len(X_treino))
        historico["accuracy"].append(acertos / len(X_treino))
        historico["val_loss"].append(float(cosface_entropia(cos_val, y_val, hp["s"], hp["m"])[0]))
        historico["val_accuracy"].append(float(np.mean(cos_val.argmax(axis=1) == y_val)))
        print(f"Época {epoca}/{EPOCAS} - {time.time() - inicio:.0f}s - " +
              " - ".join(f"{chave}: {valores[-1]:.4f}" for chave, valores in historico.items()))

        if historico["val_loss"][-1] < melhor_erro:
            melhor_erro, melhores, sem_melhora = historico["val_loss"][-1], {k: v.copy() for k, v in params.items()}, 0
        else:
            sem_melhora += 1
            if sem_melhora >= PACIENCIA:
                print(f"Parada antecipada: {PACIENCIA} épocas sem melhora na validação.")
                break
    params.update(melhores)                       # volta para os pesos da melhor época
    return historico


def treinar_modelo(X_treino, y_treino, X_val, y_val, n_classes, hp, gerador):
    # Cria a rede, treina e calibra o limiar do "desconhecido" na validação
    params = iniciar_pesos(n_classes, gerador, hp)
    historico = treinar_rede(params, X_treino, y_treino, X_val, y_val, hp, gerador)
    limiar = calibrar_limiar(prever(params, X_val), y_val, PERCENTIL)
    return params, historico, limiar


def teste_integrante_de_fora(X, y, idx_treino, idx_val, idx_teste, classes, hp, gerador):
    # Como medir o "desconhecido" sem fotos de desconhecidos? Fingindo que um integrante é desconhecido:
    # treina a rede SEM ele e vê quantas fotos dele saem "desconhecido". Repete para cada integrante.
    if len(classes) < 3:
        print("  Pulado: com só 2 integrantes, tirar um deixa a rede com uma classe só (não há o que separar).")
        return []
    resultados = []
    for k, nome in enumerate(classes):
        print(f"Integrante de fora {k + 1}/{len(classes)}: treinando sem {nome}")
        sem_k = lambda idx: idx[y[idx] != k]
        renumerar = lambda yy: yy - (yy > k)                          # os rótulos depois de k descem uma posição
        t, v, te = sem_k(idx_treino), sem_k(idx_val), sem_k(idx_teste)
        params, _, limiar = treinar_modelo(X[t], renumerar(y[t]), X[v], renumerar(y[v]), len(classes) - 1, hp, gerador)
        fora = decidir(prever(params, X[y == k]), limiar)              # todas as fotos dele: a rede nunca o viu
        dentro = decidir(prever(params, X[te]), limiar)
        resultados.append({"integrante": nome, "fotos": int(np.sum(y == k)), "limiar": limiar,
                           "rejeitadas": float(np.mean(fora == -1)),
                           "conhecidos_corretos": float(np.mean(dentro == renumerar(y[te])))})
        print(f"  {nome}: {resultados[-1]['rejeitadas']:.1%} das fotos viraram 'desconhecido'; "
              f"os outros foram reconhecidos certo em {resultados[-1]['conhecidos_corretos']:.1%} das fotos do teste")
    return resultados


def main():
    gerador = np.random.default_rng(SEED)        # semente fixa: o resultado se repete a cada execução
    hp = hiperparametros()
    print("Hiperparâmetros:", json.dumps(hp))

    # As linhas "Etapa N/8" mostram o andamento no terminal e na barra de progresso do site
    classes, rotulos, X, y, sessoes = carregar_dados()

    print("Etapa 3/8: divisão em treino, validação e teste")
    idx_treino, idx_val, idx_teste = dividir_dados(y, classes, sessoes)
    X_treino, y_treino = X[idx_treino], y[idx_treino]
    X_val,    y_val    = X[idx_val],    y[idx_val]
    X_teste,  y_teste  = X[idx_teste],  y[idx_teste]

    print("Etapa 4/8: montagem da CNN")
    params = iniciar_pesos(len(classes), gerador, hp)
    camadas = resumo_camadas(params, hp)
    for camada in camadas:
        print(f"  {camada['nome']:>9}  {camada['tipo']:<30} saída {camada['saida']}  {camada['parametros']} pesos")
    print(f"Etapa 5/8: treinamento (até {EPOCAS} épocas)")
    params, historico, limiar = treinar_modelo(X_treino, y_treino, X_val, y_val, len(classes), hp, gerador)

    # Avaliação no conjunto de teste, com a decisão aberta: abaixo do limiar, a foto vira "desconhecido"
    print("Etapa 6/8: avaliação no conjunto de teste")
    n = len(classes)
    y_previsto = decidir(prever(params, X_teste), limiar)
    matriz = confusion_matrix(y_teste, np.where(y_previsto < 0, n, y_previsto), labels=range(n + 1))[:n]
    m = calcular_metricas(matriz)
    acuracias = {nome: float(np.mean(decidir(prever(params, Xc), limiar) == yc))
                 for nome, Xc, yc in [("treino", X_treino, y_treino), ("validacao", X_val, y_val),
                                      ("teste", X_teste, y_teste)]}

    print("Etapa 7/8: teste do integrante de fora (desconhecidos simulados)")
    de_fora = teste_integrante_de_fora(X, y, idx_treino, idx_val, idx_teste, classes, hp, gerador)

    # Salva o modelo (pesos + nomes + limiar) e todos os resultados (lidos pelo site)
    print("Etapa 8/8: gravação do modelo")
    os.makedirs(MODELO_DIR, exist_ok=True)
    salvar_modelo(params, classes, limiar, CAMINHO_MODELO)

    contar = lambda yc: {nome: int(np.sum(yc == c)) for c, nome in enumerate(classes)}   # fotos por integrante
    resultados = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "hiperparametros": hp,
        "limiar": limiar,
        "fotos_por_conjunto": {"treino": contar(y_treino), "validacao": contar(y_val), "teste": contar(y_teste)},
        "totais": {"fotos": len(rotulos), "com_rosto": len(y), "treino": len(y_treino),
                   "validacao": len(y_val), "teste": len(y_teste)},
        "historico": historico,
        "epocas_treinadas": len(historico["loss"]),
        "melhor_epoca": int(np.argmin(historico["val_loss"])) + 1,
        "matriz_confusao": matriz.tolist(),
        "metricas": {chave: (valor.tolist() if isinstance(valor, np.ndarray) else float(valor))
                     for chave, valor in m.items()},
        "acuracias": acuracias,
        "integrante_de_fora": de_fora,
        "camadas": camadas,
        "parametros_total": sum(c["parametros"] for c in camadas),
    }
    with open(CAMINHO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # Resumo no terminal
    print(f"\nLimiar do 'desconhecido' (cosseno): {limiar:.3f}")
    print("Matriz de confusão (teste) - linhas = real, colunas = previsto:")
    print(" " * 16 + "".join(f"{nome[:10]:>11}" for nome in classes + ["desconhec."]))
    for i, nome in enumerate(classes):
        print(f"{nome:>15} " + "".join(f"{v:>11}" for v in matriz[i]))
    print(f"\n{'integrante':>15} | precisão | recall |   F1  | especificidade | fotos no teste")
    for i, nome in enumerate(classes):
        print(f"{nome:>15} |   {m['precisao'][i]:.3f}  |  {m['recall'][i]:.3f} | {m['f1'][i]:.3f} |"
              f"     {m['especificidade'][i]:.3f}      | {matriz[i].sum()}")
    print(f"{'média macro':>15} |   {m['precisao'].mean():.3f}  |  {m['recall'].mean():.3f} |"
          f" {m['f1'].mean():.3f} |     {m['especificidade'].mean():.3f}")
    for nome, acc in acuracias.items():
        print(f"Acurácia em {nome}: {acc:.3f}")
    print(f"Diferença treino - validação: {acuracias['treino'] - acuracias['validacao']:+.3f} "
          "(perto de 0 = generaliza; grande = overfitting)")
    if de_fora:
        print(f"Integrante de fora: em média {np.mean([r['rejeitadas'] for r in de_fora]):.1%} "
              "das fotos de quem a rede não conhecia viraram 'desconhecido'")
    print(f"\nModelo salvo em {CAMINHO_MODELO}")
    print(f"Resultados em {CAMINHO_RESULTADOS}")


if __name__ == "__main__":
    main()
