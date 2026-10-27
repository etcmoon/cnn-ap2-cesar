// Reconhecimento ao vivo pela webcam.
// Ciclo: captura um quadro -> envia ao servidor (/api/reconhecer) -> recebe, para cada rosto, o cosseno com cada
// integrante e as emoções -> faz a média dos últimos 5 quadros -> desenha o nome (ou "desconhecido") e a emoção
// sobre o vídeo -> repete (~4 vezes por segundo).

const INTERVALO_MS = 250;      // tempo mínimo entre dois quadros enviados
const QUADROS_NA_MEDIA = 5;    // média dos últimos 5 quadros para o nome e a emoção não ficarem "piscando"

const video = document.getElementById("video");
const tela = document.getElementById("sobreposicao");      // canvas transparente por cima do vídeo
const ctx = tela.getContext("2d");
const botaoLigar = document.getElementById("ligar");
const botaoParar = document.getElementById("parar");
const botaoVerRede = document.getElementById("ver-rede");
const limiar = document.getElementById("limiar");
const espelhar = document.getElementById("espelhar");
const statusAoVivo = document.getElementById("status");

let stream = null;        // conexão com a câmera
let ligado = false;       // o loop de reconhecimento está rodando?
let congelado = false;    // true enquanto o painel "Ver o que a rede vê" está aberto
let historico = [];       // historico[i] = cossenos dos últimos quadros da caixa i
let historicoEmocoes = []; // historicoEmocoes[i] = emoções dos últimos quadros da caixa i
let limiarDoModelo = null; // limiar calibrado no treino: o slider volta para ele quando o modelo muda

// ---------- Funções auxiliares (também usadas por coleta.js) ----------

function esperar(ms) {
  // Pausa o código por alguns milissegundos (sem travar a página)
  return new Promise((resolver) => setTimeout(resolver, ms));
}

async function enviarJSON(url, corpo) {
  // Envia um objeto como JSON ao servidor e devolve a resposta; se o servidor mandar um erro, lança a mensagem dele
  const resposta = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(corpo),
  });
  const dados = await resposta.json();
  if (!resposta.ok) throw new Error(dados.erro || "Erro no servidor");
  return dados;
}

function capturarQuadro(origem, qualidade) {
  // Copia o quadro atual do vídeo para um canvas e devolve como JPEG em texto (dataURL).
  // O quadro é capturado SEM espelhamento: igual às fotos usadas no treino.
  const canvas = document.createElement("canvas");
  canvas.width = origem.videoWidth;
  canvas.height = origem.videoHeight;
  canvas.getContext("2d").drawImage(origem, 0, 0);
  return canvas.toDataURL("image/jpeg", qualidade);
}

function mostrarStatus(elemento, texto, erro = false) {
  // Escreve uma mensagem na caixa de status (em vermelho se for erro)
  elemento.textContent = texto;
  elemento.classList.toggle("erro", erro);
}

function decimal(v) {
  // 0.823 -> "0,82"
  return v.toFixed(2).replace(".", ",");
}

function desenharBarras(elemento, classes, similaridades) {
  // Monta uma barra horizontal para cada integrante, com o tamanho proporcional ao cosseno (semelhança)
  const melhor = similaridades.indexOf(Math.max(...similaridades));
  elemento.innerHTML = "";
  classes.forEach((nome, i) => {
    const linha = document.createElement("div");
    linha.className = "barra" + (i === melhor && similaridades[i] >= Number(limiar.value) ? " vencedora" : "");
    linha.innerHTML = `<span></span><div class="barra-fundo"><div class="barra-cheia"></div></div><span></span>`;
    linha.children[0].textContent = nome;
    linha.children[1].firstChild.style.width = (Math.max(similaridades[i], 0) * 100).toFixed(1) + "%";
    linha.children[2].textContent = decimal(similaridades[i]);
    elemento.appendChild(linha);
  });
}

function usarLimiarDoModelo(valor) {
  // Quando chega um modelo novo (limiar diferente), o slider volta ao limiar calibrado no treino
  if (valor === limiarDoModelo) return;
  limiarDoModelo = valor;
  limiar.value = valor.toFixed(2);
  document.getElementById("limiar-valor").textContent = decimal(Number(limiar.value));
}

// ---------- Webcam ----------

async function ligarWebcam() {
  // Pede permissão e liga a câmera; o canvas fica com a mesma resolução do vídeo
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (erro) {
    mostrarStatus(statusAoVivo, "Não foi possível abrir a webcam: " + erro.message, true);
    return;
  }
  video.srcObject = stream;
  await video.play();
  tela.width = video.videoWidth;
  tela.height = video.videoHeight;
  document.getElementById("video-vazio").hidden = true;
  botaoLigar.disabled = true;
  botaoParar.disabled = false;
  botaoVerRede.disabled = false;
  ligado = true;
  historico = [];
  historicoEmocoes = [];
  mostrarStatus(statusAoVivo, "Reconhecendo... olhe de frente para a câmera.");
  loopReconhecimento();
}

function pararWebcam() {
  // Desliga a câmera, para o loop e limpa as caixas
  ligado = false;
  congelado = false;
  if (stream) stream.getTracks().forEach((faixa) => faixa.stop());
  video.srcObject = null;
  ctx.clearRect(0, 0, tela.width, tela.height);
  document.getElementById("video-vazio").hidden = false;
  document.getElementById("leitura-ao-vivo").innerHTML = "";
  botaoLigar.disabled = false;
  botaoParar.disabled = true;
  botaoVerRede.disabled = true;
  mostrarStatus(statusAoVivo, "Webcam desligada.");
}

// ---------- Loop de reconhecimento ----------

async function loopReconhecimento() {
  // Envia um quadro, espera a resposta e só então envia o próximo (as requisições nunca se acumulam)
  while (ligado) {
    const inicio = performance.now();
    if (!congelado) {
      try {
        const resposta = await enviarJSON("/api/reconhecer", { imagem: capturarQuadro(video, 0.8) });
        if (ligado && !congelado) desenharCaixas(resposta);
      } catch (erro) {
        mostrarStatus(statusAoVivo, erro.message, true);
      }
    }
    await esperar(Math.max(0, INTERVALO_MS - (performance.now() - inicio)));   // ~250 ms entre quadros
  }
}

function mediaDosUltimos(lista, i, valores) {
  // Guarda os valores da caixa i em lista[i] e devolve a média dos últimos QUADROS_NA_MEDIA quadros
  lista[i] = (lista[i] || []).concat([valores]).slice(-QUADROS_NA_MEDIA);
  return valores.map((_, k) => lista[i].reduce((soma, quadro) => soma + quadro[k], 0) / lista[i].length);
}

function desenharCaixas(resposta) {
  // Desenha um oval em cada rosto encontrado, com o nome e o cosseno: verde se reconhecido e vermelho
  // ("desconhecido") se abaixo do limiar; embaixo, a emoção. Os rostos vêm do maior para o menor, então a
  // caixa i costuma ser a mesma pessoa de um quadro para o outro.
  usarLimiarDoModelo(resposta.limiar);
  if (historico.length !== resposta.caixas.length) {                 // entrou ou saiu alguém: recomeça a média
    historico = [];
    historicoEmocoes = [];
  }
  ctx.clearRect(0, 0, tela.width, tela.height);
  const tamanhoFonte = Math.round(tela.width / 28);
  ctx.font = `bold ${tamanhoFonte}px system-ui, sans-serif`;
  ctx.lineWidth = Math.max(2, tela.width / 200);
  const leitura = [];

  resposta.caixas.forEach((caixa, i) => {
    const media = mediaDosUltimos(historico, i, caixa.similaridades);
    const melhor = media.indexOf(Math.max(...media));
    const reconhecido = media[melhor] >= Number(limiar.value);
    const texto = reconhecido ? `${resposta.classes[melhor]} (${decimal(media[melhor])})` : "desconhecido";
    const cor = reconhecido ? "#2ca02c" : "#d62728";
    const emocoes = mediaDosUltimos(historicoEmocoes, i, caixa.emocoes);
    const e = emocoes.indexOf(Math.max(...emocoes));
    const emocao = `${resposta.emocoes[e]} ${(emocoes[e] * 100).toFixed(0)}%`;

    // Se o vídeo está espelhado, espelhamos só a POSIÇÃO x da caixa; o texto continua desenhado normalmente
    const x = espelhar.checked ? tela.width - caixa.x - caixa.largura : caixa.x;
    const yNome = Math.max(caixa.y, tamanhoFonte * 1.5);                       // o nome não sai pelo topo do vídeo
    const yBaixo = caixa.y + caixa.altura;

    ctx.strokeStyle = cor;
    ctx.beginPath();
    ctx.ellipse(x + caixa.largura / 2, caixa.y + caixa.altura / 2, caixa.largura / 2, caixa.altura / 2, 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = cor;
    ctx.fillRect(x, yNome - tamanhoFonte * 1.5, ctx.measureText(texto).width + 16, tamanhoFonte * 1.5);   // fundo do nome
    ctx.fillStyle = "#1f2328";
    ctx.fillRect(x, yBaixo, ctx.measureText(emocao).width + 16, tamanhoFonte * 1.5);                       // fundo da emoção
    ctx.fillStyle = "#ffffff";
    ctx.fillText(texto, x + 8, yNome - tamanhoFonte * 0.35);                                                 // nome + cosseno
    ctx.fillText(emocao, x + 8, yBaixo + tamanhoFonte * 1.15);                                              // emoção
    leitura.push(`Rosto ${i + 1}: ${texto} · ${emocao}`);
  });
  document.getElementById("leitura-ao-vivo").textContent = leitura.join(" · ") || "ninguém na imagem";
  mostrarStatus(statusAoVivo, "Reconhecendo... olhe de frente para a câmera.");
}

// ---------- Botão "Ver o que a rede vê" ----------

async function verOQueARedeVe() {
  // Congela o quadro atual e mostra as etapas do pré-processamento, o Grad-CAM e o cosseno com cada integrante
  congelado = true;
  video.pause();
  const painel = document.getElementById("visao-rede");
  const etapas = document.getElementById("etapas");
  etapas.innerHTML = "<p>Processando o quadro...</p>";
  painel.hidden = false;
  try {
    const r = await enviarJSON("/api/etapas", { imagem: capturarQuadro(video, 0.9) });
    const titulos = ["1. rosto alinhado pelos olhos", "2. 64×64", "3. CLAHE", "4. suavizada", "5. máscara oval (entrada da CNN)"];
    etapas.innerHTML = "";
    r.etapas.concat([r.gradcam]).forEach((src, i) => {
      const figura = document.createElement("figure");
      figura.innerHTML = `<img alt=""><figcaption></figcaption>`;
      figura.querySelector("img").src = src;
      figura.querySelector("figcaption").textContent = titulos[i] || "Grad-CAM: onde a rede olhou";
      etapas.appendChild(figura);
    });
    usarLimiarDoModelo(r.limiar);
    desenharBarras(document.getElementById("barras"), r.classes, r.similaridades);
  } catch (erro) {
    etapas.innerHTML = "";
    const p = document.createElement("p");
    p.className = "status erro";
    p.textContent = erro.message;
    etapas.appendChild(p);
  }
  painel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function voltarAoVivo() {
  // Fecha o painel e volta a reconhecer continuamente
  document.getElementById("visao-rede").hidden = true;
  congelado = false;
  if (ligado) video.play();
}

// ---------- Eventos dos controles ----------

botaoLigar.addEventListener("click", ligarWebcam);
botaoParar.addEventListener("click", pararWebcam);
botaoVerRede.addEventListener("click", verOQueARedeVe);
document.getElementById("voltar-ao-vivo").addEventListener("click", voltarAoVivo);
limiar.addEventListener("input", () => {
  document.getElementById("limiar-valor").textContent = decimal(Number(limiar.value));
});
espelhar.addEventListener("change", () => video.classList.toggle("espelhado", espelhar.checked));
