# Configurações gerais do projeto (é só aqui que precisamos mexer).
import os

# Pasta raiz do repositório (a pasta acima de src/)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Onde ficam as fotos: uma subpasta por integrante.
# A variável de ambiente DATASET_DIR permite testar com outra pasta sem mexer no código.
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(RAIZ, "dataset"))

MODELO_DIR     = os.path.join(RAIZ, "modelo")                        # modelo treinado e resultados
CAMINHO_MODELO = os.path.join(MODELO_DIR, "modelo_cnn.npz")          # pesos da CNN + nomes dos integrantes
CAMINHO_RESULTADOS = os.path.join(MODELO_DIR, "resultados.json")
IMAGENS_DIR    = os.path.join(RAIZ, "web", "img", "resultados")      # PNGs exibidos na documentação

IMG_SIZE     = 64     # todo rosto vira 64 x 64 pixels
MARGEM_ROSTO = 0.1    # aumenta 10% de cada lado da caixa do detector (pega a testa e o queixo inteiros)
EPOCAS       = 150    # número máximo de passadas pelos dados de treino (a parada antecipada costuma parar antes)
PACIENCIA    = 10     # o treino para se o erro na validação não melhorar por 10 épocas seguidas
BATCH        = 32     # quantas fotos a rede vê antes de cada ajuste dos pesos
SEED         = 42     # semente aleatória: garante o mesmo resultado a cada execução
LIMIAR       = 0.6    # abaixo de 60% de confiança, a pessoa é "desconhecida"
