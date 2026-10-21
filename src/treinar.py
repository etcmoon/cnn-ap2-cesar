# Script principal: executa o pipeline completo.
# Leitura -> pré-processamento -> divisão -> aumento -> CNN -> treino -> avaliação -> imagens intermediárias.
# Uso (na raiz do repositório):  python -m src.treinar
import os
import json
import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from src.config import DATASET_DIR, MODELO_DIR, CAMINHO_MODELO, CAMINHO_RESULTADOS, EPOCAS, PACIENCIA, BATCH, SEED
from src.dados import ler_imagem, listar_dataset
from src.preprocessamento import preprocessar
from src.divisao import dividir_dados
from src.aumento import aumentar
from src.modelo import (iniciar_pesos, propagar, retropropagar, softmax_entropia, passo_adam, prever,
                        resumo_camadas, salvar_modelo)
from src.metricas import calcular_metricas
from src import visualizacoes


def treinar_rede(params, X_treino, y_treino, X_val, y_val, gerador):
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
            pontuacoes, cache = propagar(params, x, gerador)                      # gerador = dropout ligado
            erro, d_pontuacoes = softmax_entropia(pontuacoes, y_treino[lote])
            passo_adam(params, retropropagar(params, cache, d_pontuacoes), estado_adam)
            soma_erro += float(erro) * len(lote)
            acertos += int((pontuacoes.argmax(axis=1) == y_treino[lote]).sum())

        probs_val = prever(params, X_val)
        historico["loss"].append(soma_erro / len(X_treino))
        historico["accuracy"].append(acertos / len(X_treino))
        historico["val_loss"].append(float(-np.mean(np.log(probs_val[np.arange(len(y_val)), y_val] + 1e-9))))
        historico["val_accuracy"].append(float(np.mean(probs_val.argmax(axis=1) == y_val)))
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


def main():
    gerador = np.random.default_rng(SEED)        # semente fixa: o resultado se repete a cada execução

    # As linhas "Etapa N/8" mostram o andamento no terminal e na barra de progresso do site
    print("Etapa 1/8: leitura das fotos")
    classes, caminhos, rotulos = listar_dataset(DATASET_DIR)
    assert len(classes) >= 2, f"São necessários pelo menos 2 integrantes com fotos em {DATASET_DIR}."

    # Pré-processamento de todas as fotos (mesma função usada no reconhecimento ao vivo).
    # Fotos em que o detector não acha rosto ficam de fora (de costas, borradas, rosto cortado...).
    print("Etapa 2/8: pré-processamento (detecção do rosto, CLAHE, suavização)")
    X, usadas = [], []
    for i, caminho in enumerate(caminhos):
        rosto = preprocessar(ler_imagem(caminho))
        if rosto is not None:
            X.append(rosto)
            usadas.append(i)
    caminhos, y, X = [caminhos[i] for i in usadas], rotulos[usadas], np.array(X)
    print("Integrantes:")
    for numero, nome in enumerate(classes):
        sem_rosto = np.sum(rotulos == numero) - np.sum(y == numero)
        print(f"  {numero} = {nome}: {np.sum(y == numero)} fotos ({sem_rosto} sem rosto detectado, ignoradas)")
    print("X:", X.shape, "-> (quantidade de fotos, altura, largura, canais)")

    # Divisão em treino, validação e teste por blocos de fotos seguidas
    print("Etapa 3/8: divisão em treino, validação e teste")
    idx_treino, idx_val, idx_teste = dividir_dados(y, classes)
    X_treino, y_treino = X[idx_treino], y[idx_treino]
    X_val,    y_val    = X[idx_val],    y[idx_val]
    X_teste,  y_teste  = X[idx_teste],  y[idx_teste]
    visualizacoes.fotos_originais(classes, caminhos, y)
    visualizacoes.etapas_preprocessamento(classes, caminhos, y)

    # Aumento de dados: é feito durante o treino, em cada lote; aqui só geramos a imagem de exemplo
    print("Etapa 4/8: aumento de dados")
    visualizacoes.exemplos_aumento(classes, X_treino, y_treino)

    print("Etapa 5/8: montagem da CNN")
    params = iniciar_pesos(len(classes), gerador)
    camadas = resumo_camadas(params)
    for camada in camadas:
        print(f"  {camada['nome']:>8}  {camada['tipo']:<22} saída {camada['saida']}  {camada['parametros']} pesos")
    print(f"Etapa 6/8: treinamento (até {EPOCAS} épocas)")
    historico = treinar_rede(params, X_treino, y_treino, X_val, y_val, gerador)

    # Avaliação no conjunto de teste
    print("Etapa 7/8: avaliação no conjunto de teste")
    y_previsto = prever(params, X_teste).argmax(axis=1)
    matriz = confusion_matrix(y_teste, y_previsto, labels=range(len(classes)))
    m = calcular_metricas(matriz)
    acuracias = {nome: float(accuracy_score(yc, prever(params, Xc).argmax(axis=1)))
                 for nome, Xc, yc in [("treino", X_treino, y_treino), ("validacao", X_val, y_val),
                                      ("teste", X_teste, y_teste)]}

    # Imagens intermediárias da avaliação e da rede
    print("Etapa 8/8: imagens intermediárias e gravação do modelo")
    visualizacoes.curvas_treino(historico)
    visualizacoes.desenhar_matriz(matriz, classes)
    visualizacoes.barras_metricas(m, classes)
    visualizacoes.fotos_erradas(X_teste, y_teste, y_previsto, classes)
    visualizacoes.mapas_caracteristicas(params, X_teste[0], classes[y_teste[0]])
    visualizacoes.gradcam_por_integrante(params, X_teste, y_teste, classes)

    # Salva o modelo (pesos + nomes) e todos os resultados (lidos pelo site)
    os.makedirs(MODELO_DIR, exist_ok=True)
    salvar_modelo(params, classes, CAMINHO_MODELO)

    contar = lambda yc: {nome: int(np.sum(yc == c)) for c, nome in enumerate(classes)}   # fotos por integrante
    resultados = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
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
        "camadas": camadas,
        "parametros_total": sum(c["parametros"] for c in camadas),
    }
    with open(CAMINHO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # Resumo no terminal
    print("\nMatriz de confusão (teste) - linhas = real, colunas = previsto:")
    print(" " * 16 + "".join(f"{nome[:10]:>11}" for nome in classes))
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
    print(f"\nModelo salvo em {CAMINHO_MODELO}")
    print(f"Resultados em {CAMINHO_RESULTADOS} e imagens em web/img/resultados/")


if __name__ == "__main__":
    main()
