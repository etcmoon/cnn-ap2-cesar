# Reconhecimento facial com CNN

Uma Rede Neural Convolucional (CNN) **feita à mão, só com NumPy** (sem TensorFlow, Keras ou PyTorch) que reconhece
os integrantes do grupo pelo rosto, diz **“desconhecido”** para quem ela não conhece e mostra a **emoção** de cada
rosto (alegre, triste, bravo, com medo, surpreso, nojo, neutro). Um **site local**, servido por Node.js, faz o
reconhecimento ao vivo pela webcam, a coleta de fotos e o treino, e mostra a matriz de confusão e as métricas.

## Como funciona o “desconhecido” (sem fotos de desconhecidos)

Um softmax comum sempre distribui 100% **entre os integrantes**, então um rosto novo sempre vira alguém, quase
sempre com confiança alta. Por isso a rede não termina em um softmax:

1. A CNN transforma o rosto num **embedding**: um vetor de tamanho 1, a “assinatura” do rosto.
2. Cada integrante tem um vetor aprendido, o seu **“rosto médio”**. A rede compara o embedding com cada um pelo
   **cosseno** (1 = idêntico).
3. No treino, a perda **CosFace** exige que o cosseno com o integrante certo ganhe por uma **margem `m`**. Isso
   obriga as fotos de cada pessoa a ficarem **juntas**, e não só “do lado certo da linha”.
4. O **limiar** é calibrado na validação: o cosseno que 95% das fotos verdadeiras alcançam. Se nenhum integrante
   passa dele, a pessoa é **desconhecida**. O limiar fica salvo no modelo, e o site deixa ajustá-lo.

Para **medir** isso sem um dataset de desconhecidos, o treino faz o **teste do integrante de fora**: treina a rede
sem cada integrante e conta quantas fotos dele saem “desconhecido”. Esse teste precisa de 3 integrantes ou mais.

## Como as partes se conversam

```
navegador (webcam, desenho)  --HTTP-->  servidor.js (Node + Express)  --stdin/stdout, JSON por linha-->  src/servico.py (Python)
```

- O **Node** cuida de todo o HTTP: entrega o site, salva as fotos da coleta, roda o treino e lê os resultados.
- O **Python** cuida das redes. O Node abre **um** processo `python -m src.servico`, que carrega o modelo e o
  DeepFace uma vez e responde a cada pedido com uma linha de JSON. Assim o ao vivo usa exatamente o mesmo
  pré-processamento do treino.

## Estrutura

```
├── servidor.js               servidor Node (Express): site, rotas da API e ponte com o Python
├── dataset/<nome>/           fotos de cada integrante (o nome da pasta é o rótulo)
├── modelo/                   detector YuNet (.onnx) e, gerados: modelo_cnn.npz, resultados.json, hiperparametros.json
├── src/
│   ├── config.py             parâmetros (IMG_SIZE, EPOCAS, PACIENCIA, PERCENTIL...) e os hiperparâmetros HP
│   ├── dados.py              ler_imagem (com correção EXIF), listar_dataset, sessao_da_foto, imagem_de_dataurl
│   ├── preprocessamento.py   detectar_rostos (YuNet), alinhar (pelos olhos), tratar_rosto (CLAHE, máscara oval)
│   ├── divisao.py            treino/validação/teste por captura inteira (ou por blocos, com poucas capturas)
│   ├── aumento.py            aumento de dados a cada lote (geometria, luz de um lado, desfoque, pedaço tampado...)
│   ├── modelo.py             a CNN feita à mão: convolução, pooling, embedding, cosseno, CosFace, Adam/AdamW
│   ├── metricas.py           VP/FP/FN/VN, precisão, recall, F1, especificidade e acurácia (com a coluna desconhecido)
│   ├── treinar.py            o pipeline completo e o teste do integrante de fora
│   ├── buscar_hiperparametros.py  busca avulsa dos melhores hiperparâmetros
│   ├── emocao.py             emoções com o DeepFace
│   ├── reconhecimento.py     reconhecer_quadro (todos os rostos do quadro), grad_cam
│   └── servico.py            processo Python aberto pelo servidor.js
├── web/                      site: index.html, css/, js/
└── tests/                    testes com pytest
```

## Como usar

### 1. Pré-requisitos e instalação

- Python **3.10 a 3.12** (testado com 3.11; o TensorFlow do DeepFace ainda não cobre todas as versões novas);
- Node.js **18 ou mais novo**.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell);  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt     # NumPy, OpenCV, scikit-learn, Pillow, DeepFace (+ TensorFlow, ~1 GB), pytest
npm install                         # Express
```

O servidor usa sozinho o Python do `.venv` (ou o caminho da variável `PYTHON`). Na primeira vez, o DeepFace baixa
os pesos do modelo de emoções (~6 MB) para `~/.deepface`.

### 2. Coletar as fotos

Rode `npm start`, abra `http://localhost:5000` e use **“Preparar o modelo” → passo 1**. Cada clique em “Iniciar
captura” é **uma captura**: a página pede poses diferentes (de frente, virar, olhar para cima e para baixo, sorrir,
sério, boca aberta) e salva as fotos como `dataset/<nome>/<nome>_sNN_0000.jpg`, onde `sNN` é o número da captura.

**O mais importante para a rede não “decorar”:** façam **3 capturas ou mais por pessoa**, cada uma num **lugar, luz e
roupa diferentes**. Com 3 ou mais capturas, a validação e o teste usam **capturas inteiras que a rede nunca viu**.
Assim o limiar e as métricas medem a generalização de verdade. Com menos capturas, o treino avisa que o resultado fica
otimista.

Também dá para copiar fotos para `dataset/<nome>/`. Fotos sem `_sNN` no nome contam como uma única captura. Pastas que
começam com `.` ou `_` (como `dataset/_descartadas/`) são ignoradas.

### 3. Treinar

Pelo site (passo 2, **Treinar modelo**) ou pelo terminal, que é o mesmo comando:

```bash
python -m src.treinar
```

As etapas são: leitura → pré-processamento → divisão → montagem da CNN → treino → avaliação no teste → teste do
integrante de fora (um treino a mais por integrante) → gravação. O pré-processamento inclui detecção com YuNet,
alinhamento pelos olhos, 64×64, CLAHE, suavização e máscara oval. O treino vai até 150 épocas, com aumento de dados e
parada antecipada.

O terminal mostra a matriz de confusão (com a coluna **desconhecido**), as métricas, a acurácia em treino, validação e
teste, a **diferença treino − validação** (o termômetro do overfitting) e o resultado do integrante de fora. No site,
tudo isso fica no botão **“Ver resultados do treino”**.

### 4. Hiperparâmetros: buscar uma vez, depois treinar “leve”

```bash
python -m src.buscar_hiperparametros 12    # 12 combinações sorteadas; leva de ~30 a 60 min sem GPU
```

A busca sorteia taxa de aprendizado, dropout, filtros, tamanho do embedding, escala `s` e margem `m` do CosFace e o
decaimento L2. Ela treina cada combinação e dá uma nota na validação: **acerto − impostor**. O impostor é uma foto em
que **outro** integrante também passa do limiar. Isso premia rostos de cada pessoa juntos e longe dos outros, sem
fotos de desconhecidos.

A tabela é impressa e a melhor combinação vai para `modelo/hiperparametros.json`. A partir daí, **todo treino normal
usa esses valores**, sem repetir a busca. Para voltar ao `HP` do `src/config.py`, apague o JSON.

### 5. Reconhecimento ao vivo

`npm start` → **http://localhost:5000** → “Ligar webcam”.

- Cada rosto ganha um oval com o **nome e o cosseno** (verde) ou **“desconhecido”** (vermelho), e a **emoção** embaixo.
  Os dois são a média dos últimos 5 quadros.
- **Limiar de semelhança:** começa no valor calibrado pelo treino. Subir o limiar deixa a rede mais exigente (mais
  “desconhecido”); descer deixa mais permissiva.
- **Ver o que a rede vê:** congela o quadro e mostra as etapas do pré-processamento, o Grad-CAM e o cosseno com cada
  integrante.

### 6. Testes

```bash
pytest
```

Os testes cobrem:

- os gradientes da CNN, incluindo a cabeça de cosseno, contra a derivada numérica;
- a rede decorando um lote pequeno;
- o limiar e a decisão do desconhecido;
- o alinhamento pelos olhos e o formato da entrada;
- o aumento de dados;
- as métricas, com a coluna desconhecido;
- a sintaxe dos JavaScript.

## Regras do trabalho

**A CNN que reconhece os integrantes é feita à mão:** a ida, a volta (retropropagação de cada camada, incluindo a
normalização do embedding e o CosFace), o Adam/AdamW, o dropout e a parada antecipada são código nosso com NumPy.
A emoção pode usar um modelo pronto, e usa o DeepFace.

| Etapa | Ferramenta |
|---|---|
| Abrir as fotos .jpg/.png/.heic | `PIL` + `pillow-heif` |
| Achar o rosto e os olhos | detector **YuNet** do `OpenCV` (só diz *onde* há um rosto, não *quem* é) |
| Alinhamento, cinza, CLAHE, suavização, aumento de dados | `OpenCV` + NumPy |
| **A CNN:** convolução, pooling, embedding, cosseno, CosFace, retropropagação, Adam, Grad-CAM | **código próprio com NumPy** (`src/modelo.py`) |
| Emoções | **DeepFace** (modelo pronto), aplicado no rosto já recortado e alinhado |
| Matriz de confusão | `scikit-learn` (as métricas são calculadas por nós a partir da matriz) |
| Servidor e site | Node.js com `Express`; HTML, CSS e JavaScript puros |

## Privacidade

As fotos e o modelo treinado têm os **rostos** dos integrantes e ficam fora do git (`dataset/` e `modelo/` estão no
`.gitignore`). Só o detector YuNet, que é público, fica no repositório.
