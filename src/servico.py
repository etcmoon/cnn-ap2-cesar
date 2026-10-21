# Processo Python que cuida SÓ da rede neural. Não é um servidor web: quem cuida do HTTP é o Node (servidor.js).
# O Node abre este processo uma vez; o modelo é carregado uma vez e os pedidos chegam pelo stdin.
# Protocolo: cada linha recebida é um pedido em JSON ({"id", "acao", ...}) e cada linha devolvida é a resposta,
# com o mesmo "id" para o Node saber a qual pedido ela pertence.
# Uso: aberto automaticamente pelo servidor.js (python -m src.servico).
import io
import os
import sys
import json
import base64
import cv2
import numpy as np
from PIL import Image

from src.config import CAMINHO_MODELO
from src.dados import imagem_de_dataurl
from src.modelo import carregar_modelo, prever
from src.preprocessamento import preprocessar
from src.reconhecimento import reconhecer_quadro, grad_cam

MSG_SEM_MODELO = "O modelo ainda não foi treinado: rode `python -m src.treinar` primeiro."

saida = sys.stdout          # canal exclusivo das respostas para o Node
cache = {"data": None}      # modelo carregado na memória


def obter_modelo():
    # Carrega o modelo do disco só quando precisa, e de novo se ele foi retreinado (o arquivo mudou)
    if not os.path.exists(CAMINHO_MODELO):
        return None
    data = os.path.getmtime(CAMINHO_MODELO)
    if cache["data"] != data:
        params, classes = carregar_modelo(CAMINHO_MODELO)
        cache.update(data=data, params=params, classes=classes)
    return cache


def png_base64(matriz):
    # Converte uma matriz de pixels (0-255) em um PNG no formato dataURL, que o navegador exibe direto
    buffer = io.BytesIO()
    Image.fromarray(np.clip(matriz, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def reconhecer(m, pedido):
    # Reconhecimento ao vivo: acha os rostos do quadro e devolve as probabilidades de cada um
    quadro = imagem_de_dataurl(pedido["imagem"])
    return {"caixas": reconhecer_quadro(m["params"], quadro), "classes": m["classes"]}


def etapas(m, pedido):
    # Botão "Ver o que a rede vê": as etapas do pré-processamento, o Grad-CAM e a previsão
    quadro = imagem_de_dataurl(pedido["imagem"])
    entrada, imagens = preprocessar(quadro, mostrar_etapas=True)      # imagens = [cinza, 1, 2, 3, 4, 5]
    if entrada is None:
        return {"erro": "Nenhum rosto encontrado no quadro: olhe de frente para a câmera.", "status": 422}
    probs = prever(m["params"], entrada[np.newaxis])[0]

    # Grad-CAM: mapa de calor colorido (escala "jet") misturado com a imagem de entrada em cinza
    calor, _ = grad_cam(m["params"], entrada)
    colorido = cv2.applyColorMap((calor * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]   # BGR -> RGB
    mistura = 0.55 * entrada * 255 + 0.45 * colorido                  # (64,64,1) + (64,64,3) -> (64,64,3)

    return {
        "etapas": [png_base64(img) for img in imagens[1:5]] + [png_base64(entrada[:, :, 0] * 255)],
        "gradcam": png_base64(mistura),
        "probabilidades": probs.tolist(),
        "classes": m["classes"],
    }


ACOES = {"reconhecer": reconhecer, "etapas": etapas}   # nome da ação pedida pelo Node -> função que atende


def atender(pedido):
    # Executa um pedido; sem modelo treinado, devolve o erro 503 com a mensagem para o site
    m = obter_modelo()
    if m is None:
        return {"erro": MSG_SEM_MODELO, "status": 503}
    return ACOES[pedido["acao"]](m, pedido)


def main():
    sys.stdout = sys.stderr                # qualquer outro print vai para o terminal,
                                           # e o stdout fica só para as respostas ao Node
    obter_modelo()                         # já carrega o modelo, para o primeiro pedido não demorar
    print("Python pronto para receber pedidos.")
    for linha in sys.stdin:                # espera pedidos até o Node fechar o processo
        pedido = json.loads(linha)
        try:
            resposta = atender(pedido)
        except Exception as erro:          # um pedido com problema não derruba o processo
            resposta = {"erro": f"Erro no Python: {erro}", "status": 500}
        saida.write(json.dumps({"id": pedido["id"], **resposta}) + "\n")
        saida.flush()                      # envia a resposta na hora


if __name__ == "__main__":
    main()
