# Configurações gerais do projeto (é só aqui que precisamos mexer).
import os
import json

# Pasta raiz do repositório (a pasta acima de src/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Onde ficam as fotos: uma subpasta por integrante.
# A variável de ambiente DATASET_DIR permite testar com outra pasta sem mexer no código.
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(RAIZ, "dataset"))

MODELO_DIR     = os.path.join(RAIZ, "modelo")                        # modelo treinado e resultados
CAMINHO_MODELO = os.path.join(MODELO_DIR, "modelo_cnn.npz")          # pesos da CNN + nomes + limiar
CAMINHO_RESULTADOS = os.path.join(MODELO_DIR, "resultados.json")
CAMINHO_HP     = os.path.join(MODELO_DIR, "hiperparametros.json")    # gravado pela busca de hiperparâmetros
CAMINHO_YUNET  = os.path.join(MODELO_DIR, "face_detection_yunet_2023mar.onnx")   # detector de rostos (OpenCV Zoo)

IMG_SIZE     = 64     # todo rosto vira 64 x 64 pixels
EPOCAS       = 150    # número máximo de passadas pelos dados de treino (a parada antecipada costuma parar antes)
PACIENCIA    = 10     # o treino para se o erro na validação não melhorar por 10 épocas seguidas
BATCH        = 32     # quantas fotos a rede vê antes de cada ajuste dos pesos
SEED         = 42     # semente aleatória: garante o mesmo resultado a cada execução
PERCENTIL    = 5      # limiar do "desconhecido": 95% das fotos verdadeiras da validação passam dele

# Hiperparâmetros da CNN. A busca (python -m src.buscar_hiperparametros) grava os melhores em
# modelo/hiperparametros.json, que substituem estes no treino normal. Apague o JSON para voltar a estes.
HP = {
    "taxa":    0.001,          # taxa de aprendizado do Adam
    "dropout": 0.5,            # fração de neurônios desligados a cada passo do treino
    "filtros": [8, 16, 32],    # filtros de cada convolução: bordas -> texturas -> partes do rosto
    "densa":   128,            # tamanho do embedding (a "assinatura" do rosto)
    "s":       16.0,           # escala do cosseno (CosFace)
    "m":       0.35,           # margem do cosseno: obriga as fotos de cada pessoa a ficarem juntas
    "l2":      0.0001,         # decaimento dos pesos (AdamW): pesos menores = menos decoreba
}


def hiperparametros():
    # HP padrão, trocado pelos valores achados pela busca, se ela já rodou
    if not os.path.exists(CAMINHO_HP):
        return dict(HP)
    with open(CAMINHO_HP, encoding="utf-8") as f:
        return {**HP, **json.load(f)}
