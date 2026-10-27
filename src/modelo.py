# A CNN feita à mão em NumPy. A saída não é um softmax comum: a rede gera um EMBEDDING (a "assinatura" do rosto,
# de tamanho 1) e compara com o "rosto médio" de cada integrante pelo cosseno. Se nenhum cosseno passa do limiar,
# a pessoa é "desconhecida". No treino, a margem do CosFace obriga as fotos de cada pessoa a ficarem juntas.
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import IMG_SIZE


def iniciar_pesos(n_classes, gerador, hp, tamanho=IMG_SIZE):
    # Pesos aleatórios com a inicialização de He (desvio sqrt(2 / entradas), feita para a ReLU): assim o sinal
    # não some nem explode ao atravessar as camadas. Os vieses começam em zero.
    filtros = hp["filtros"]
    canais = [1] + list(filtros)
    formas = {f"W{i + 1}": (3, 3, canais[i], canais[i + 1]) for i in range(3)}    # filtros 3x3
    formas["W4"] = ((tamanho // 8) ** 2 * filtros[-1], hp["densa"])               # 3 poolings: 64 -> 8
    formas["W5"] = (hp["densa"], n_classes)                                       # "rosto médio" de cada integrante
    params = {}
    for nome, forma in formas.items():
        entradas = int(np.prod(forma[:-1]))                                       # quantos valores entram em cada neurônio
        params[nome] = gerador.normal(0, np.sqrt(2 / entradas), forma).astype(np.float32)
        if nome != "W5":                                                          # o cosseno não tem viés
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


def normalizar(v, eixo):
    # Divide cada vetor pelo seu tamanho: sobra só a direção (é o que o cosseno compara)
    return v / (np.linalg.norm(v, axis=eixo, keepdims=True) + 1e-9)


def softmax(pontuacoes):
    # Pontuações -> probabilidades que somam 1 (subtrair o máximo evita estouro na exponencial)
    e = np.exp(pontuacoes - pontuacoes.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cosface_entropia(cossenos, y, s, m):
    # Erro do CosFace e o seu gradiente em relação aos cossenos.
    # Pontuação = s * cosseno, mas na classe certa tiramos a margem m: para errar pouco, a foto precisa ficar
    # MUITO mais perto do próprio integrante do que dos outros (não basta ganhar por pouco).
    linhas = np.arange(len(y))
    pontuacoes = s * cossenos
    pontuacoes[linhas, y] -= s * m
    p = softmax(pontuacoes)
    # log da probabilidade sem passar por p (com s = 16, p da classe certa chega a 1e-9 no começo do treino)
    maximo = pontuacoes.max(axis=1)
    log_p = pontuacoes[linhas, y] - maximo - np.log(np.exp(pontuacoes - maximo[:, None]).sum(axis=1))
    perda = -np.mean(log_p)
    p[linhas, y] -= 1
    return perda, s * p / len(y)


def propagar(params, x, dropout=0.0, gerador=None):
    # Passa um lote x (N, 64, 64, 1) pela rede e devolve os cossenos (N, classes) entre o embedding de cada foto e o
    # "rosto médio" de cada integrante. Com um gerador aleatório, é modo treino (dropout ligado).
    # O cache guarda o que a retropropagação precisa.
    convs = []
    a = x
    for i in (1, 2, 3):
        z = np.maximum(conv_forward(a, params[f"W{i}"], params[f"b{i}"]), 0)    # convolução + ReLU
        s = pool_forward(z)                                                     # pooling
        convs.append((a, z, s))                                                 # entrada, mapas, saída do bloco
        a = s
    plano = a.reshape(len(a), -1)                                               # flatten: 8x8x32 -> 2048
    mascara = 1.0
    if gerador is not None and dropout > 0:                                     # dropout "invertido": desliga
        mascara = ((gerador.random(plano.shape) >= dropout) / (1 - dropout)).astype(plano.dtype)   # neurônios
    h_entrada = plano * mascara                                                 # e compensa a escala
    h = h_entrada @ params["W4"] + params["b4"]                                 # densa = embedding
    e = normalizar(h, 1)                                                        # embedding de tamanho 1
    w = normalizar(params["W5"], 0)                                             # "rostos médios" de tamanho 1
    cossenos = e @ w                                                            # 1 cosseno por integrante
    return cossenos, {"convs": convs, "forma": a.shape, "mascara": mascara, "h_entrada": h_entrada,
                      "h": h, "e": e, "w": w}


def retropropagar(params, cache, d_cossenos, ate=None):
    # Regra da cadeia, da saída para a entrada: devolve os gradientes de todos os pesos.
    # Com ate="conv3" (ou conv1, conv2), para no meio e devolve o gradiente dos mapas dessa camada (Grad-CAM).
    # Normalizar v -> v/|v| tem gradiente (d - v̂ (d · v̂)) / |v|: só a parte de d que muda a DIREÇÃO de v conta.
    e, w = cache["e"], cache["w"]
    d_w = e.T @ d_cossenos                                                      # cosseno = e · w
    d_e = d_cossenos @ w.T
    W5 = params["W5"]
    g = {"W5": (d_w - w * (d_w * w).sum(axis=0, keepdims=True)) / (np.linalg.norm(W5, axis=0, keepdims=True) + 1e-9)}
    h = cache["h"]
    d_h = (d_e - e * (d_e * e).sum(axis=1, keepdims=True)) / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
    g["W4"], g["b4"] = cache["h_entrada"].T @ d_h, d_h.sum(axis=0)
    d = ((d_h @ params["W4"].T) * cache["mascara"]).reshape(cache["forma"])     # desfaz dropout e flatten
    for i in (3, 2, 1):
        a, z, s = cache["convs"][i - 1]
        d_mapas = pool_backward(z, s, d)
        if ate == f"conv{i}":
            return d_mapas
        g[f"W{i}"], g[f"b{i}"], d = conv_backward(a, params[f"W{i}"], d_mapas * (z > 0))
    return g


def passo_adam(params, grads, estado, taxa=0.001, l2=0.0, b1=0.9, b2=0.999, eps=1e-8):
    # Otimizador Adam: cada peso anda contra o seu gradiente, com uma média móvel do gradiente (m, "inércia")
    # dividida pela raiz da média móvel do gradiente ao quadrado (v, adapta o passo de cada peso).
    # l2 > 0 vira AdamW: a cada passo os pesos (não os vieses) também encolhem um pouco, o que segura a decoreba.
    t = estado["t"] = estado.get("t", 0) + 1
    for k in params:
        m = estado[k + "_m"] = b1 * estado.get(k + "_m", 0) + (1 - b1) * grads[k]
        v = estado[k + "_v"] = b2 * estado.get(k + "_v", 0) + (1 - b2) * grads[k] ** 2
        passo = taxa * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if k.startswith("W"):
            passo = passo + taxa * l2 * params[k]
        params[k] -= passo.astype(params[k].dtype)


def prever(params, X, lote=256):
    # Cossenos de cada integrante para muitas fotos, em lotes (para não usar memória demais)
    return np.concatenate([propagar(params, X[i:i + lote])[0] for i in range(0, len(X), lote)])


def calibrar_limiar(cossenos, y, percentil):
    # Limiar do "desconhecido": o cosseno que 95% (com percentil 5) das fotos VERDADEIRAS da validação alcançam
    # com o próprio integrante. Só usa fotos dos integrantes: não precisamos de fotos de desconhecidos.
    return float(np.percentile(cossenos[np.arange(len(y)), y], percentil))


def decidir(cossenos, limiar):
    # Integrante mais parecido com cada foto, ou -1 ("desconhecido") se nem ele passa do limiar
    return np.where(cossenos.max(axis=1) >= limiar, cossenos.argmax(axis=1), -1)


def resumo_camadas(params, hp):
    # Cada camada com o formato da saída e o número de pesos (impresso no terminal durante o treino)
    t, canais = IMG_SIZE, 1
    camadas = [{"nome": "entrada", "tipo": "Entrada", "saida": [t, t, 1], "parametros": 0}]
    for i in (1, 2, 3):
        f = params[f"W{i}"].shape[-1]
        camadas.append({"nome": f"conv{i}", "tipo": "Convolução 3x3 + ReLU", "saida": [t, t, f],
                        "parametros": 9 * canais * f + f})
        t, canais = t // 2, f
        camadas.append({"nome": f"pool{i}", "tipo": "Max pooling 2x2", "saida": [t, t, f], "parametros": 0})
    densa, n_classes = params["W5"].shape
    camadas += [{"nome": "flatten", "tipo": "Flatten", "saida": [t * t * canais], "parametros": 0},
                {"nome": "dropout", "tipo": f"Dropout {hp['dropout']}", "saida": [t * t * canais], "parametros": 0},
                {"nome": "embedding", "tipo": "Densa + normalização", "saida": [densa],
                 "parametros": params["W4"].size + densa},
                {"nome": "cosseno", "tipo": f"Cosseno (CosFace s={hp['s']}, m={hp['m']})", "saida": [n_classes],
                 "parametros": params["W5"].size}]
    return camadas


def salvar_modelo(params, classes, limiar, caminho):
    # Grava os pesos, os nomes dos integrantes e o limiar num arquivo só. Grava num temporário e troca no fim:
    # o servidor nunca lê um arquivo pela metade.
    temporario = caminho + ".tmp.npz"
    np.savez(temporario, classes=np.array(classes), limiar=np.array(limiar), **params)
    os.replace(temporario, caminho)


def carregar_modelo(caminho):
    # Lê os pesos, os nomes dos integrantes (na ordem dos rótulos) e o limiar do "desconhecido"
    with np.load(caminho) as arquivo:
        classes = [str(nome) for nome in arquivo["classes"]]
        limiar = float(arquivo["limiar"])
        params = {k: arquivo[k] for k in arquivo.files if k not in ("classes", "limiar")}
    return params, classes, limiar
