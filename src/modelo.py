# A CNN, feita à mão: só NumPy, sem TensorFlow, Keras ou PyTorch.
# Cada camada tem uma função "forward" (calcula a saída) e uma "backward" (calcula os gradientes pela regra da cadeia).
# Arquitetura: 3 blocos (convolução 3x3 + ReLU + max pooling 2x2) -> dropout -> densa 64 + ReLU -> densa (1 por classe).
# Os pesos ficam num dicionário {"W1": ..., "b1": ..., ...}, que é salvo direto com np.savez.
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import IMG_SIZE

FILTROS = (8, 16, 32)   # filtros de cada convolução: bordas -> texturas -> partes do rosto
DENSA   = 64            # neurônios da camada densa
DROPOUT = 0.5           # fração de neurônios desligados a cada passo do treino


def iniciar_pesos(n_classes, gerador, tamanho=IMG_SIZE):
    # Pesos aleatórios com a inicialização de He (desvio sqrt(2 / entradas), feita para a ReLU): assim o sinal
    # não some nem explode ao atravessar as camadas. Os vieses começam em zero.
    canais = (1,) + FILTROS
    formas = {f"W{i + 1}": (3, 3, canais[i], canais[i + 1]) for i in range(3)}    # filtros 3x3
    formas["W4"] = ((tamanho // 8) ** 2 * FILTROS[-1], DENSA)                     # 3 poolings: 64 -> 8
    formas["W5"] = (DENSA, n_classes)
    params = {}
    for nome, forma in formas.items():
        entradas = int(np.prod(forma[:-1]))                                       # quantos valores entram em cada neurônio
        params[nome] = gerador.normal(0, np.sqrt(2 / entradas), forma).astype(np.float32)
        params["b" + nome[1:]] = np.zeros(forma[-1], np.float32)
    return params


def janelas_3x3(x):
    # Todas as janelas 3x3 da imagem (com 1 pixel de borda de zeros, para a saída ter o mesmo tamanho: "same").
    # x: (N, altura, largura, canais) -> (N, altura, largura, canais, 3, 3). É o "im2col": a convolução vira
    # uma multiplicação de matrizes, que o NumPy faz muito rápido.
    return sliding_window_view(np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0))), (3, 3), axis=(1, 2))


def conv_forward(x, W, b):
    # Cada pixel de saída = soma de (janela 3x3 de todos os canais) x (pesos do filtro) + viés
    return np.tensordot(janelas_3x3(x), W, axes=([4, 5, 3], [0, 1, 2])) + b


def conv_backward(x, W, d_saida):
    # Gradientes da convolução, a partir do gradiente da saída (d_saida):
    #   dW = cada janela da entrada multiplicada pelo gradiente da saída naquela posição, somado em tudo
    #   dx = "convolução" do gradiente da saída com o filtro girado 180°
    dW = np.tensordot(janelas_3x3(x), d_saida, axes=([0, 1, 2], [0, 1, 2])).transpose(1, 2, 0, 3)
    db = d_saida.sum(axis=(0, 1, 2))
    dx = np.tensordot(janelas_3x3(d_saida), W[::-1, ::-1], axes=([4, 5, 3], [0, 1, 3]))
    return dW, db, dx


def pool_forward(x):
    # Max pooling 2x2: cada bloco 2x2 vira o seu maior valor (a imagem fica com metade da altura e da largura)
    n, altura, largura, canais = x.shape
    return x.reshape(n, altura // 2, 2, largura // 2, 2, canais).max(axis=(2, 4))


def pool_backward(x, saida, d_saida):
    # O gradiente de cada bloco vai só para o pixel que era o máximo (os outros não influenciaram a saída)
    ampliar = lambda a: a.repeat(2, axis=1).repeat(2, axis=2)
    return (x == ampliar(saida)) * ampliar(d_saida)


def softmax(pontuacoes):
    # Pontuações -> probabilidades que somam 1 (subtrair o máximo evita estouro na exponencial)
    e = np.exp(pontuacoes - pontuacoes.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def softmax_entropia(pontuacoes, y):
    # Erro (entropia cruzada média) e o seu gradiente em relação às pontuações: probabilidade - 1 na classe certa
    p = softmax(pontuacoes)
    linhas = np.arange(len(y))
    perda = -np.mean(np.log(p[linhas, y] + 1e-9))
    p[linhas, y] -= 1
    return perda, p / len(y)


def propagar(params, x, gerador=None):
    # Passa um lote x (N, 64, 64, 1) pela rede e devolve as pontuações (N, classes).
    # Com um gerador aleatório, é modo treino (dropout ligado). O cache guarda o que a retropropagação precisa.
    convs = []
    a = x
    for i in (1, 2, 3):
        z = np.maximum(conv_forward(a, params[f"W{i}"], params[f"b{i}"]), 0)    # convolução + ReLU
        s = pool_forward(z)                                                     # pooling
        convs.append((a, z, s))                                                 # entrada, mapas, saída do bloco
        a = s
    plano = a.reshape(len(a), -1)                                               # flatten: 8x8x32 -> 2048
    mascara = 1.0
    if gerador is not None:                                                     # dropout "invertido": desliga
        mascara = ((gerador.random(plano.shape) >= DROPOUT) / (1 - DROPOUT)).astype(plano.dtype)   # neurônios
    h_entrada = plano * mascara                                                 # e compensa a escala
    h = np.maximum(h_entrada @ params["W4"] + params["b4"], 0)                  # densa + ReLU
    pontuacoes = h @ params["W5"] + params["b5"]                                # 1 pontuação por classe
    return pontuacoes, {"convs": convs, "forma": a.shape, "mascara": mascara, "h_entrada": h_entrada, "h": h}


def retropropagar(params, cache, d_pontuacoes, ate=None):
    # Regra da cadeia, da saída para a entrada: devolve os gradientes de todos os pesos.
    # Com ate="conv3" (ou conv1, conv2), para no meio e devolve o gradiente dos mapas dessa camada (Grad-CAM).
    g = {"W5": cache["h"].T @ d_pontuacoes, "b5": d_pontuacoes.sum(axis=0)}
    d_h = (d_pontuacoes @ params["W5"].T) * (cache["h"] > 0)                    # ReLU: só passa onde estava ativo
    g["W4"], g["b4"] = cache["h_entrada"].T @ d_h, d_h.sum(axis=0)
    d = ((d_h @ params["W4"].T) * cache["mascara"]).reshape(cache["forma"])     # desfaz dropout e flatten
    for i in (3, 2, 1):
        a, z, s = cache["convs"][i - 1]
        d_mapas = pool_backward(z, s, d)
        if ate == f"conv{i}":
            return d_mapas
        g[f"W{i}"], g[f"b{i}"], d = conv_backward(a, params[f"W{i}"], d_mapas * (z > 0))
    return g


def passo_adam(params, grads, estado, taxa=0.001, b1=0.9, b2=0.999, eps=1e-8):
    # Otimizador Adam: cada peso anda contra o seu gradiente, com uma média móvel do gradiente (m, "inércia")
    # dividida pela raiz da média móvel do gradiente ao quadrado (v, adapta o passo de cada peso).
    t = estado["t"] = estado.get("t", 0) + 1
    for k in params:
        m = estado[k + "_m"] = b1 * estado.get(k + "_m", 0) + (1 - b1) * grads[k]
        v = estado[k + "_v"] = b2 * estado.get(k + "_v", 0) + (1 - b2) * grads[k] ** 2
        params[k] -= (taxa * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)).astype(params[k].dtype)


def prever(params, X, lote=256):
    # Probabilidades de cada classe para muitas fotos, em lotes (para não usar memória demais)
    return np.concatenate([softmax(propagar(params, X[i:i + lote])[0]) for i in range(0, len(X), lote)])


def resumo_camadas(params):
    # Cada camada com o formato da saída e o número de pesos (mostrado na documentação do site)
    t, canais = IMG_SIZE, 1
    camadas = [{"nome": "entrada", "tipo": "Entrada", "saida": [t, t, 1], "parametros": 0}]
    for i, f in enumerate(FILTROS, 1):
        camadas.append({"nome": f"conv{i}", "tipo": "Convolução 3x3 + ReLU", "saida": [t, t, f],
                        "parametros": 9 * canais * f + f})
        t, canais = t // 2, f
        camadas.append({"nome": f"pool{i}", "tipo": "Max pooling 2x2", "saida": [t, t, f], "parametros": 0})
    n_classes = params["W5"].shape[1]
    camadas += [{"nome": "flatten", "tipo": "Flatten", "saida": [t * t * canais], "parametros": 0},
                {"nome": "dropout", "tipo": f"Dropout {DROPOUT}", "saida": [t * t * canais], "parametros": 0},
                {"nome": "densa", "tipo": "Densa + ReLU", "saida": [DENSA], "parametros": params["W4"].size + DENSA},
                {"nome": "saida", "tipo": "Densa + softmax", "saida": [n_classes],
                 "parametros": params["W5"].size + n_classes}]
    return camadas


def salvar_modelo(params, classes, caminho):
    # Grava os pesos e os nomes dos integrantes num arquivo só. Grava num temporário e troca no fim:
    # o servidor nunca lê um arquivo pela metade.
    temporario = caminho + ".tmp.npz"
    np.savez(temporario, classes=np.array(classes), **params)
    os.replace(temporario, caminho)


def carregar_modelo(caminho):
    # Lê os pesos e os nomes dos integrantes (na ordem dos rótulos)
    with np.load(caminho) as arquivo:
        classes = [str(nome) for nome in arquivo["classes"]]
        params = {k: arquivo[k] for k in arquivo.files if k != "classes"}
    return params, classes
