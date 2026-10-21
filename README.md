# Reconhecimento facial com CNN

Uma Rede Neural Convolucional (CNN) **feita à mão, só com NumPy** (sem TensorFlow, Keras ou PyTorch), que reconhece
os integrantes do grupo pelo rosto. O repositório tem o pipeline completo em Python (detecção e tratamento do rosto,
divisão, aumento de dados, CNN, treino, métricas e imagens intermediárias) e um **site local**, servido por Node.js, com:

- **no topo:** o reconhecimento **ao vivo pela webcam**, usando o modelo treinado;
- **logo abaixo:** a **documentação** passo a passo, com o código real de cada etapa, as imagens
  intermediárias e as métricas do último treino.

O notebook original do Google Colab está em [`notebooks/reconhecimento_facial_cnn.ipynb`](notebooks/reconhecimento_facial_cnn.ipynb)
e roda o mesmo pipeline importando as funções de `src/`, etapa por etapa.

## Como as partes se conversam

```
navegador (webcam, desenho)  --HTTP-->  servidor.js (Node + Express)  --stdin/stdout, JSON por linha-->  src/servico.py (Python + CNN)
```

- O **Node** cuida de todo o HTTP: entrega o site, salva as fotos da coleta, lê os resultados e o código da documentação.
- O **Python** cuida só da rede neural. Ele não é um servidor web: o Node abre **um** processo `python -m src.servico`,
  que carrega o modelo uma vez e responde a cada pedido com uma linha de JSON. Assim o reconhecimento ao vivo usa
  exatamente o mesmo `preprocessar` do treino.

## Estrutura

```
├── servidor.js             servidor Node (Express): site, rotas da API e ponte com o Python
├── package.json            dependência do Node (express)
├── dataset/<nome>/         fotos de cada integrante (o nome da pasta é o rótulo)
├── modelo/                 gerado pelo treino: modelo_cnn.npz (pesos + nomes), resultados.json
├── src/
│   ├── config.py           todos os parâmetros (IMG_SIZE, EPOCAS, PACIENCIA, BATCH, SEED, LIMIAR...)
│   ├── dados.py            ler_imagem (com correção EXIF), listar_dataset, imagem_de_dataurl
│   ├── preprocessamento.py detectar_rostos (Haar), tratar_rosto (64x64, CLAHE, suavização, máscara oval), preprocessar
│   ├── divisao.py          treino/validação/teste por blocos de fotos seguidas
│   ├── aumento.py          aumentar (rotação, zoom, deslocamento, espelho, brilho, contraste, ruído), a cada lote
│   ├── modelo.py           a CNN feita à mão: convolução, pooling, densa, softmax, retropropagação, Adam
│   ├── metricas.py         VP/FP/FN/VN, precisão, recall, F1, especificidade e acurácia a partir da matriz
│   ├── visualizacoes.py    imagens intermediárias em PNG
│   ├── reconhecimento.py   reconhecer_quadro (todos os rostos do quadro), grad_cam
│   ├── servico.py          processo Python aberto pelo servidor.js: recebe pedidos e roda a CNN
│   └── treinar.py          laço de treino (treinar_rede) e o pipeline completo (main)
├── web/                    site: index.html, css/, js/, img/resultados/ (PNGs gerados pelo treino)
├── tests/                  testes com pytest
└── notebooks/              notebook original
```

## Como usar

### 1. Pré-requisitos e ambiente virtual

- Python **3.10 ou mais novo** (testado com Python 3.11);
- Node.js **18 ou mais novo** (testado com Node 22).

Na pasta do repositório, crie o ambiente virtual do Python:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt   # Python: NumPy, OpenCV, scikit-learn, Pillow, matplotlib, pytest
npm install                       # Node: Express
```

O servidor Node usa sozinho o Python do `.venv`. Se o ambiente estiver em outro lugar,
indique o caminho do Python na variável de ambiente `PYTHON`.

### 3. Coletar as fotos

Cada integrante precisa de uma pasta `dataset/<nome>/` com as fotos dele. Há duas formas:

- **Pelo site:** rode `npm start`, abra `http://localhost:5000` e use a área **“Preparar o modelo”**, passo 1
  (ela abre sozinha enquanto não houver modelo treinado). Digite o nome (ou clique em um nome já cadastrado),
  clique em “Iniciar captura”: depois de uma contagem de 3 s, a página tira N fotos (padrão 80, uma a cada 250 ms)
  e salva em `dataset/<nome>/`, continuando a numeração.
- **Copiando arquivos:** coloque fotos `.jpg`, `.jpeg`, `.png` ou `.heic` (iPhone) em `dataset/<nome>/`. Os arquivos são lidos em
  ordem alfabética, que deve ser a ordem em que foram tirados (ex.: `ana_0000.jpg`, `ana_0001.jpg`...).

Dicas para a rede aprender o rosto, e não o fundo:

- façam **3 ou mais sessões** por pessoa, em **lugares e iluminações diferentes**;
- fotografem **todos nos mesmos lugares**: se cada pessoa tiver um cenário só dela, o cenário vira uma pista mais fácil que o rosto;
- olhem de frente para a câmera e mexam levemente a cabeça durante a captura (fotos em que o detector não acha
  o rosto, como de perfil ou com a mão no rosto, são descartadas no treino);
- tirem fotos **perto e longe** da câmera: se uma classe só aparece de perto, a distância vira pista;
- **confiram as pastas antes de treinar**: cada pasta deve ter só fotos daquela classe (a rede aprende os erros de rótulo também).
  Fotos que não servem podem ir para `dataset/_descartadas/`, que o treino ignora;
- se existir uma pasta `dataset/ninguem/` de uma versão antiga, movam-na para `dataset/_descartadas/`: agora
  “ninguém” é quando o detector não acha rosto, e não uma classe treinada;
- juntem **cerca de 150 a 300 fotos por pessoa**, em **quantidades parecidas** entre as pessoas;
- nomes de pasta só com letras minúsculas sem acento, números, `_` ou `-` (pastas que começam com `.` ou `_` são ignoradas).

### 4. Treinar

**Pelo site (mais fácil):** na área “Preparar o modelo”, passo 2, a página mostra quantas fotos cada integrante tem
e avisa o que falta (pelo menos 2 integrantes e 3 fotos por pessoa). Clique em **Treinar modelo** e acompanhe a
barra de progresso (etapa, época e acurácia na validação) e o log. Marcando **“Treinar automaticamente depois da
coleta”** no passo 1, o treino começa sozinho quando as fotos são salvas. Sem GPU, o treino leva alguns minutos
(mais fotos, mais tempo); dá para cancelar a qualquer momento sem perder o modelo anterior.

**Pelo terminal:** é o mesmo comando que o site usa por trás.

```bash
python -m src.treinar
```

O script faz, na ordem: leitura → pré-processamento (detecção do rosto, 64×64, CLAHE, suavização e máscara oval) →
divisão em treino/validação/teste (por blocos de fotos seguidas) → montagem da CNN → treino (até 150 épocas, com
aumento de dados a cada lote e parada antecipada) → avaliação no teste → imagens intermediárias. No fim, imprime as
métricas no terminal e salva:

| Arquivo | Conteúdo |
|---|---|
| `modelo/modelo_cnn.npz` | os pesos da rede e os nomes dos integrantes, na ordem dos rótulos |
| `modelo/resultados.json` | métricas, matriz de confusão, histórico do treino, divisão dos dados e camadas |
| `web/img/resultados/*.png` | fotos originais, etapas do pré-processamento, aumento, curvas, matriz, métricas, erros, mapas de características e Grad-CAM |

Sempre que coletarem fotos novas, rodem o treino de novo. O servidor percebe o modelo novo sozinho (não precisa reiniciar).

### 5. Abrir o site

```bash
npm start
```

O terminal mostra `Site em http://localhost:5000` e, alguns segundos depois, `Python pronto para receber pedidos.`
(é o tempo de carregar o modelo). Abra **http://localhost:5000**. Clique em “Ligar webcam” e permita o acesso à câmera.
O navegador só libera a webcam em `localhost` ou em HTTPS, por isso o servidor escuta apenas em `127.0.0.1`.

- **Rostos:** o detector acha os rostos em qualquer lugar do quadro, e cada um ganha um oval com o nome.
- **Ninguém:** sem rosto no quadro, aparece “ninguém na imagem”.
- **Limiar:** abaixo dele (padrão 60%), o oval fica vermelho com “desconhecido”.
- **Ver o que a rede vê:** congela o quadro e mostra as etapas do pré-processamento, o Grad-CAM e as probabilidades.

Se o modelo ainda não foi treinado, o site avisa para rodar `python -m src.treinar` primeiro.

### 6. Testes

```bash
pytest
```

Testa a CNN (cada gradiente da retropropagação contra a derivada numérica, a rede decorando um lote pequeno e
salvar/carregar), o pré-processamento (formato `(64, 64, 1)` em `float32` entre 0 e 1, foto sem rosto, máscara oval,
aumento), as métricas (contra valores calculados à mão) e a documentação do site (cada nota aponta para uma linha que
existe no código).

## Regras do trabalho

**A CNN é feita à mão:** nada de TensorFlow, Keras, PyTorch, autograd ou modelo pré-treinado. A ida, a volta
(retropropagação de cada camada), o Adam, o dropout e a parada antecipada são código nosso com NumPy. O resto do
pipeline pode usar bibliotecas.

| Etapa | Ferramenta |
|---|---|
| Abrir e salvar o arquivo .jpg/.png/.heic | `PIL` + `pillow-heif` para .heic |
| Achar o rosto na foto | detector Haar do `OpenCV` (só diz *onde* há um rosto, não *quem* é) |
| Tons de cinza, redimensionamento, CLAHE, suavização, aumento de dados | `OpenCV` |
| **A CNN:** convolução, pooling, densa, softmax, retropropagação, Adam, dropout, Grad-CAM | **código próprio com NumPy** (`src/modelo.py`) |
| Máscara oval, divisão dos dados em blocos | código próprio com NumPy |
| Matriz de confusão e acurácia | `scikit-learn` (precisão, recall, F1 e especificidade calculados por nós a partir da matriz) |
| Gráficos e imagens intermediárias | `matplotlib` |
| Servidor local | Node.js com `Express` (a rede roda num processo Python aberto por ele) |
| Site e captura da webcam | HTML, CSS e JavaScript puros (`navigator.mediaDevices.getUserMedia`) |

O `opencv-python-headless` fica em `<5`: o OpenCV 5 removeu o detector Haar (`CascadeClassifier`).

## Privacidade

Este repositório guarda **fotos de rostos** dos integrantes. Recomendamos deixá-lo **privado** ou ter o
consentimento de todos antes de publicá-lo. Se a pasta `dataset/` passar de ~100 MB, usem
[Git LFS](https://git-lfs.com/) para as imagens (`git lfs track "dataset/**/*.jpg"`).
