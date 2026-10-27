// Coleta de fotos para o dataset pelo navegador.
// "Ver câmera" liga a webcam antes da captura, para a pessoa se posicionar com calma.
// "Iniciar captura" faz uma contagem regressiva de 3 s, tira N fotos (uma a cada 400 ms) pedindo poses diferentes e
// envia para /api/fotos, que salva em dataset/<nome>/ como uma captura nova (_sNN no nome do arquivo).
// Depois atualiza a tabela do passo 2 e, se a opção estiver marcada, já começa o treino.
// Usa esperar, enviarJSON, capturarQuadro e mostrarStatus de ao_vivo.js, e atualizarDataset/iniciarTreino de treino.js.

// Não há guia na tela: no treino, o detector de rostos acha o rosto em qualquer lugar da foto.

// Poses pedidas durante a captura, cada uma por 1/8 das fotos: fotos variadas ensinam a IDENTIDADE,
// e não "a pessoa séria olhando de frente"
const POSES = ["olhe de frente", "vire um pouco para a esquerda", "vire um pouco para a direita", "olhe um pouco para cima",
  "olhe um pouco para baixo", "sorria", "faça cara séria", "abra a boca, como se falasse"];

const videoColeta = document.getElementById("coleta-video");
const botaoColeta = document.getElementById("coleta-iniciar");
const botaoPrevia = document.getElementById("coleta-previa");
const statusColeta = document.getElementById("coleta-status");
const contagem = document.getElementById("coleta-contagem");

let streamColeta = null;    // câmera da coleta (ligada pela prévia ou pela captura)

function normalizarNome(texto) {
  // "Ana Clara" -> "ana_clara": minúsculas, sem acento e sem espaço (o nome vira o nome da pasta)
  return texto.trim().toLowerCase()
    .normalize("NFD").replace(/[̀-ͯ]/g, "")    // remove os acentos
    .replace(/\s+/g, "_");
}

async function ligarCameraColeta() {
  // Liga a webcam da coleta (se ainda não estiver ligada)
  if (streamColeta) return true;
  try {
    streamColeta = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (erro) {
    mostrarStatus(statusColeta, "Não foi possível abrir a webcam: " + erro.message, true);
    return false;
  }
  videoColeta.srcObject = streamColeta;
  await videoColeta.play();
  botaoPrevia.textContent = "Desligar câmera";
  return true;
}

function desligarCameraColeta() {
  // Desliga a webcam da coleta
  if (streamColeta) streamColeta.getTracks().forEach((faixa) => faixa.stop());
  streamColeta = null;
  videoColeta.srcObject = null;
  botaoPrevia.textContent = "Ver câmera";
}

async function alternarPrevia() {
  // Botão "Ver câmera": liga a câmera para a pessoa se posicionar antes de capturar (ou desliga)
  if (streamColeta) {
    desligarCameraColeta();
    mostrarStatus(statusColeta, "Câmera desligada.");
  } else if (await ligarCameraColeta()) {
    mostrarStatus(statusColeta, "Olhe de frente para a câmera, com o rosto inteiro visível, e clique em “Iniciar captura”.");
  }
}

async function coletarFotos() {
  const nome = normalizarNome(document.getElementById("coleta-nome").value);
  const n = Number(document.getElementById("coleta-n").value);
  if (!/^[a-z0-9][a-z0-9_-]*$/.test(nome)) {
    mostrarStatus(statusColeta, "Digite um nome usando só letras, números, _ ou -.", true);
    return;
  }
  if (!(n >= 1 && n <= 400)) {
    mostrarStatus(statusColeta, "A quantidade de fotos deve ficar entre 1 e 400.", true);
    return;
  }

  // 1. Liga a webcam (se a prévia já estava ligada, usa a mesma câmera e a deixa ligada no fim)
  botaoColeta.disabled = true;
  botaoPrevia.disabled = true;
  const previaLigada = streamColeta !== null;
  if (!(await ligarCameraColeta())) {
    botaoColeta.disabled = false;
    botaoPrevia.disabled = false;
    return;
  }

  // 2. Contagem regressiva de 3 segundos para a pessoa se posicionar
  mostrarStatus(statusColeta, `${nome}: olhe de frente para a câmera.`);
  for (let s = 3; s > 0; s--) {
    contagem.textContent = s;
    await esperar(1000);
  }
  contagem.textContent = "";

  // 3. Tira n fotos, uma a cada 400 ms (guardadas como JPEG em texto), trocando de pose a cada 1/8 da captura
  const fotos = [];
  for (let i = 0; i < n; i++) {
    const pose = POSES[Math.floor(i * POSES.length / n)];
    mostrarStatus(statusColeta, `${nome}: ${pose.toUpperCase()} · foto ${i + 1} de ${n}`);
    fotos.push(capturarQuadro(videoColeta, 0.9));
    await esperar(400);
  }

  // 4. Desliga a câmera (se foi ligada só para a captura) e envia as fotos ao servidor
  if (!previaLigada) desligarCameraColeta();
  botaoPrevia.disabled = false;
  mostrarStatus(statusColeta, "Enviando as fotos para o servidor...");
  try {
    const r = await enviarJSON("/api/fotos", { nome, imagens: fotos });
    const treinarAgora = document.getElementById("coleta-treinar").checked;
    mostrarStatus(statusColeta, `${r.salvas} fotos salvas em ${r.pasta}/. ` + (treinarAgora
      ? "Começando o treino (passo 2)..."
      : "Quando terminar a coleta de todos, clique em “Treinar modelo” (passo 2)."));
    await atualizarDataset();
    if (treinarAgora) iniciarTreino();
  } catch (erro) {
    mostrarStatus(statusColeta, erro.message, true);
  }
  botaoColeta.disabled = false;
}

botaoPrevia.addEventListener("click", alternarPrevia);
botaoColeta.addEventListener("click", coletarFotos);
