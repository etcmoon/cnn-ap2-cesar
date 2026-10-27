// Servidor local em Node.js (Express): entrega o site da pasta web/ e as rotas da API.
// A rede neural continua em Python: ao iniciar, o servidor abre UM processo Python (src/servico.py), que carrega
// o modelo uma vez e responde pelo stdin/stdout, um JSON por linha. O Python não precisa de servidor web.
// Uso (na raiz do repositório):  npm start   e abra http://localhost:5000

const express = require("express");
const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline");
const { spawn } = require("node:child_process");

const RAIZ = __dirname;
const PASTA_WEB = path.join(RAIZ, "web");
const DATASET_DIR = process.env.DATASET_DIR || path.join(RAIZ, "dataset");   // mesma variável do src/config.py
const CAMINHO_RESULTADOS = path.join(RAIZ, "modelo", "resultados.json");
const CAMINHO_MODELO = path.join(RAIZ, "modelo", "modelo_cnn.npz");
const MINIMO_FOTOS = 3;          // a divisão em treino/validação/teste precisa de pelo menos 3 fotos por pessoa
const MSG_SEM_MODELO = "O modelo ainda não foi treinado: rode `python -m src.treinar` primeiro.";
const PORTA = 5000;

// ======================= Ponte com o Python =======================

let python = null;              // processo Python em execução
let proximoId = 0;              // cada pedido ganha um número, que volta na resposta
const pendentes = new Map();    // id do pedido -> funções que resolvem a Promise dele

function caminhoDoPython() {
  // Usa o Python do ambiente virtual .venv (onde as bibliotecas estão instaladas), se existir
  const candidatos = [path.join(RAIZ, ".venv", "Scripts", "python.exe"), path.join(RAIZ, ".venv", "bin", "python")];
  return process.env.PYTHON || candidatos.find((c) => fs.existsSync(c)) || "python";
}

function encerrarPython(motivo) {
  // Recusa os pedidos que estavam esperando; o Python será aberto de novo no próximo pedido
  console.error(motivo);
  for (const pedido of pendentes.values()) pedido.rejeitar(new Error(motivo));
  pendentes.clear();
  python = null;
}

function iniciarPython() {
  // Abre "python -m src.servico"; os avisos do Python (stderr) aparecem direto neste terminal
  python = spawn(caminhoDoPython(), ["-m", "src.servico"], {
    cwd: RAIZ,
    stdio: ["pipe", "pipe", "inherit"],
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },   // o DeepFace escreve emojis no log
  });
  readline.createInterface({ input: python.stdout }).on("line", (linha) => {
    const resposta = JSON.parse(linha);             // cada linha é a resposta de um pedido
    const pedido = pendentes.get(resposta.id);
    pendentes.delete(resposta.id);
    if (pedido) pedido.resolver(resposta);
  });
  python.on("error", (erro) => encerrarPython("Não foi possível abrir o Python: " + erro.message));
  python.on("exit", (codigo) => encerrarPython(`O processo Python terminou (código ${codigo}).`));
}

function pedirAoPython(pedido) {
  // Envia um pedido ao Python (uma linha JSON) e devolve uma Promise que se resolve quando a resposta chegar
  if (!python) iniciarPython();
  const id = ++proximoId;
  return new Promise((resolver, rejeitar) => {
    pendentes.set(id, { resolver, rejeitar });
    python.stdin.write(JSON.stringify({ id, ...pedido }) + "\n");
  });
}

async function responderComPython(res, pedido) {
  // Repassa um pedido ao Python e devolve a resposta dele ao navegador (com o código de erro, se houver)
  try {
    const { id, erro, status, ...dados } = await pedirAoPython(pedido);
    if (erro) return res.status(status || 500).json({ erro });
    res.json(dados);
  } catch (erro) {
    res.status(500).json({ erro: erro.message });
  }
}

// ======================= Funções da API =======================

function fotosDaPasta(pasta) {
  // Fotos aceitas pelo Python: .jpg, .jpeg, .png, .heic e .heif
  return fs.readdirSync(pasta).filter((arquivo) => /\.(jpe?g|png|hei[cf])$/i.test(arquivo));
}

function sessaoDaFoto(arquivo) {
  // Mesma regra do src/dados.py: "ana_s03_0042.jpg" -> 3; fotos antigas, sem "_sNN", são a captura 0
  const achou = arquivo.match(/_s(\d+)_/);
  return achou ? Number(achou[1]) : 0;
}

function listarDataset() {
  // Lista os integrantes (subpastas de dataset/), quantas fotos e quantas capturas cada um tem.
  // Ignora pastas que começam com "." ou "_", como o Python.
  if (!fs.existsSync(DATASET_DIR)) return [];
  return fs.readdirSync(DATASET_DIR, { withFileTypes: true })
    .filter((item) => item.isDirectory() && !/^[._]/.test(item.name))
    .map((item) => item.name).sort()
    .map((nome) => {
      const fotos = fotosDaPasta(path.join(DATASET_DIR, nome));
      return { nome, fotos: fotos.length, sessoes: new Set(fotos.map(sessaoDaFoto)).size };
    });
}

function salvarFotos(nome, imagens) {
  // Salva as fotos (JPEG em base64) em dataset/<nome>/ como uma captura nova: "ana_s03_0042.jpg"
  // (s03 = 3ª captura). A numeração das fotos continua a já existente.
  const pasta = path.join(DATASET_DIR, nome);
  fs.mkdirSync(pasta, { recursive: true });
  const jaExistem = fotosDaPasta(pasta);
  const sessao = String(Math.max(0, ...jaExistem.map(sessaoDaFoto)) + 1).padStart(2, "0");
  imagens.forEach((texto, i) => {
    const bytes = Buffer.from(texto.split(",")[1], "base64");      // tira o cabeçalho "data:image/jpeg;base64,"
    const numero = String(jaExistem.length + i).padStart(4, "0");  // 7 -> "0007"
    fs.writeFileSync(path.join(pasta, `${nome}_s${sessao}_${numero}.jpg`), bytes);
  });
  return pasta;
}

// ======================= Rotas =======================

const app = express();
app.use(express.json({ limit: "100mb" }));   // as fotos da coleta chegam todas juntas, em base64
app.use(express.static(PASTA_WEB));          // GET / entrega web/index.html; também css/, js/ e img/

// Reconhecimento ao vivo: o Python acha os rostos do quadro e devolve as probabilidades de cada um.
// O servidor não guarda estado: a média dos últimos quadros é feita no JavaScript do navegador.
app.post("/api/reconhecer", (req, res) => {
  responderComPython(res, { acao: "reconhecer", imagem: req.body.imagem });
});

// Botão "Ver o que a rede vê": as 5 etapas do pré-processamento, o Grad-CAM e a previsão (calculados no Python)
app.post("/api/etapas", (req, res) => {
  responderComPython(res, { acao: "etapas", imagem: req.body.imagem });
});

// Coleta: recebe as fotos tiradas no navegador e salva em dataset/<nome>/
app.post("/api/fotos", (req, res) => {
  const nome = String(req.body.nome || "").trim().toLowerCase();
  const imagens = req.body.imagens || [];
  // Só letras minúsculas sem acento, números, _ e - (evita criar pastas fora do dataset)
  if (!/^[a-z0-9][a-z0-9_-]*$/.test(nome)) {
    return res.status(400).json({ erro: "Nome inválido: use só letras minúsculas sem acento, números, _ ou -." });
  }
  if (!imagens.length || !imagens.every((t) => typeof t === "string" && t.startsWith("data:image/jpeg;base64,"))) {
    return res.status(400).json({ erro: "Nenhuma foto JPEG recebida." });
  }
  const pasta = salvarFotos(nome, imagens);
  res.json({ salvas: imagens.length, pasta: path.relative(RAIZ, pasta).split(path.sep).join("/") });
});

// Métricas, matriz de confusão e histórico do treino (gerados por src/treinar.py)
app.get("/api/resultados", (req, res) => {
  if (!fs.existsSync(CAMINHO_RESULTADOS)) return res.status(503).json({ erro: MSG_SEM_MODELO });
  res.type("json").send(fs.readFileSync(CAMINHO_RESULTADOS, "utf-8"));
});

// Situação do dataset: fotos e capturas por integrante e se já existe um modelo treinado
app.get("/api/dataset", (req, res) => {
  res.json({ integrantes: listarDataset(), minimo_fotos: MINIMO_FOTOS, modelo_treinado: fs.existsSync(CAMINHO_MODELO) });
});

// ======================= Treino pelo site =======================

const treino = { rodando: false, linhas: [], inicio: null, fim: null, codigo: null };   // situação do último treino
let processoTreino = null;

function iniciarTreino() {
  // Roda "python -m src.treinar" (o mesmo comando do terminal) e guarda cada linha que ele imprime.
  // O site consulta essas linhas a cada segundo para mostrar a etapa, a época e o log.
  Object.assign(treino, { rodando: true, linhas: [], inicio: Date.now(), fim: null, codigo: null });
  processoTreino = spawn(caminhoDoPython(), ["-u", "-m", "src.treinar"], {   // -u: cada linha sai na hora
    cwd: RAIZ,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },   // acentos certos no log
  });
  const guardarLinha = (linha) => { if (linha.trim()) treino.linhas.push(linha); };
  readline.createInterface({ input: processoTreino.stdout }).on("line", guardarLinha);
  readline.createInterface({ input: processoTreino.stderr }).on("line", guardarLinha);   // avisos e erros também
  const terminar = (codigo) => {
    Object.assign(treino, { rodando: false, fim: Date.now(), codigo });
    processoTreino = null;
  };
  processoTreino.on("error", (erro) => { guardarLinha("Não foi possível abrir o Python: " + erro.message); terminar(-1); });
  processoTreino.on("close", terminar);    // código 0 = deu certo
}

// Começa um treino (só um por vez), depois de conferir se o dataset tem o mínimo necessário
app.post("/api/treino", (req, res) => {
  if (treino.rodando) return res.status(409).json({ erro: "Já existe um treino em andamento." });
  const integrantes = listarDataset();
  if (integrantes.length < 2) return res.status(400).json({ erro: "São necessários pelo menos 2 integrantes com fotos." });
  const comPoucas = integrantes.find((i) => i.fotos < MINIMO_FOTOS);
  if (comPoucas) return res.status(400).json({ erro: `${comPoucas.nome} tem só ${comPoucas.fotos} foto(s): o mínimo é ${MINIMO_FOTOS}.` });
  iniciarTreino();
  res.json(treino);
});

// Situação do treino: se está rodando e as linhas impressas até agora
app.get("/api/treino", (req, res) => res.json(treino));

// Cancela o treino em andamento (o modelo anterior continua valendo: os arquivos só são gravados no fim)
app.post("/api/treino/cancelar", (req, res) => {
  if (processoTreino) {
    // No Windows, o taskkill /T encerra também o processo Python aberto pelo .venv
    if (process.platform === "win32") spawn("taskkill", ["/pid", String(processoTreino.pid), "/T", "/F"]);
    else processoTreino.kill();
  }
  res.json({ ok: true });
});

const servidor = app.listen(PORTA, "127.0.0.1", () => {
  // Só nesta máquina: o navegador só libera a webcam em localhost ou HTTPS
  console.log(`Site em http://localhost:${PORTA}`);
  iniciarPython();   // já abre o Python, para o modelo carregar enquanto a página abre
});
servidor.on("error", (erro) => {
  // O erro mais comum: outro servidor (ou uma janela antiga deste) ainda está usando a porta
  if (erro.code === "EADDRINUSE") console.error(`A porta ${PORTA} já está em uso: feche o outro servidor e tente de novo.`);
  else console.error(erro.message);
  process.exit(1);
});
