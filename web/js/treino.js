// Treino pelo site, sem precisar do terminal.
// Mostra quantas fotos cada integrante tem, inicia o treino (o servidor roda "python -m src.treinar")
// e acompanha o andamento consultando /api/treino a cada segundo: etapa, época, barra de progresso e log.
// Usa esperar, enviarJSON e mostrarStatus de ao_vivo.js; no fim, chama carregarResultados de resultados.js.

const RECOMENDADO = 150;      // fotos por pessoa recomendadas (abaixo disso, só um aviso)
const CAPTURAS = 3;           // capturas por pessoa para o teste usar capturas que a rede nunca viu
const botaoTreinar = document.getElementById("treinar");
const botaoCancelar = document.getElementById("cancelar-treino");
const statusTreino = document.getElementById("treino-status");

let bloqueios = [];           // motivos que impedem o treino (ex.: menos de 2 integrantes)
let treinoRodando = false;
let acompanhando = false;     // evita dois "loops" de consulta ao mesmo tempo
let cancelado = false;

// ---------- Situação do dataset ----------

async function atualizarDataset() {
  // Busca as fotos por integrante e monta a tabela, os avisos e os botões com os nomes já cadastrados
  const d = await (await fetch("/api/dataset")).json();
  const integrantes = d.integrantes;
  const maior = Math.max(1, ...integrantes.map((i) => i.fotos));

  // Tabela: nome, barra proporcional e quantidade de fotos
  const tabela = document.getElementById("dataset-tabela");
  tabela.innerHTML = integrantes.length ? "" : '<p class="sem-imagem">Nenhuma foto ainda: comece pelo passo 1.</p>';
  integrantes.forEach((i) => {
    const linha = document.createElement("div");
    linha.className = "barra";
    linha.innerHTML = `<span></span><div class="barra-fundo"><div class="barra-cheia"></div></div><span></span>`;
    linha.children[0].textContent = i.nome;
    linha.children[1].firstChild.style.width = (i.fotos / maior * 100).toFixed(1) + "%";
    linha.children[2].textContent = `${i.fotos} (${i.sessoes} cap.)`;
    tabela.appendChild(linha);
  });

  // Bloqueios (impedem o treino) e dicas (só avisam).
  bloqueios = [];
  const dicas = [];
  if (integrantes.length < 2) bloqueios.push("Colete fotos de pelo menos 2 integrantes.");
  integrantes.forEach((i) => {
    if (i.fotos < d.minimo_fotos) bloqueios.push(`${i.nome}: só ${i.fotos} foto(s); o mínimo é ${d.minimo_fotos}.`);
    else if (i.fotos < RECOMENDADO) dicas.push(`${i.nome}: ${i.fotos} fotos. O recomendado é ${RECOMENDADO} ou mais, em capturas diferentes.`);
    if (i.fotos && i.sessoes < CAPTURAS) dicas.push(`${i.nome}: ${i.sessoes} captura(s). Façam ${CAPTURAS} ou mais, em lugares e luzes diferentes: com menos, o teste fica otimista e o overfitting passa despercebido.`);
  });
  const menor = Math.min(...integrantes.map((i) => i.fotos));
  const maiorPessoa = Math.max(1, ...integrantes.map((i) => i.fotos));
  if (integrantes.length >= 2 && maiorPessoa > 2 * menor) dicas.push("As quantidades estão bem diferentes entre as pessoas: tentem deixá-las parecidas.");
  const avisos = document.getElementById("dataset-avisos");
  avisos.innerHTML = "";
  bloqueios.map((t) => [t, "bloqueio"]).concat(dicas.map((t) => [t, "dica"])).forEach(([texto, tipo]) => {
    const item = document.createElement("li");
    item.className = tipo;
    item.textContent = texto;
    avisos.appendChild(item);
  });

  // Botões com os nomes já cadastrados: um clique preenche o campo da coleta (evita erro de digitação)
  const fichas = document.getElementById("coleta-nomes");
  fichas.innerHTML = "";
  integrantes.forEach((i) => {
    const ficha = document.createElement("button");
    ficha.type = "button";
    ficha.className = "ficha";
    ficha.textContent = `${i.nome} (${i.fotos})`;
    ficha.addEventListener("click", () => { document.getElementById("coleta-nome").value = i.nome; });
    fichas.appendChild(ficha);
  });

  if (!d.modelo_treinado) document.getElementById("preparo").open = true;   // sem modelo: mostra o passo a passo
  botaoTreinar.disabled = treinoRodando || bloqueios.length > 0;
}

// ---------- Treino ----------

async function iniciarTreino() {
  // Pede ao servidor para começar o treino e passa a acompanhar o andamento
  cancelado = false;
  try {
    await enviarJSON("/api/treino", {});
  } catch (erro) {
    statusTreino.hidden = false;
    mostrarStatus(statusTreino, erro.message, true);
    return;
  }
  acompanharTreino();
}

async function cancelarTreino() {
  cancelado = true;
  await enviarJSON("/api/treino/cancelar", {});
}

async function acompanharTreino() {
  // Consulta /api/treino a cada segundo até o treino terminar
  if (acompanhando) return;
  acompanhando = true;
  let estado;
  do {
    estado = await (await fetch("/api/treino")).json();
    desenharTreino(estado);
    if (estado.rodando) await esperar(1000);
  } while (estado.rodando);
  acompanhando = false;
  await terminarTreino(estado);
}

function ultima(linhas, regra) {
  // Procura, de trás para frente, a última linha do log que combina com a expressão regular
  for (let i = linhas.length - 1; i >= 0; i--) {
    const achou = linhas[i].match(regra);
    if (achou) return achou;
  }
  return null;
}

function formatarTempo(ms) {
  // 83000 -> "1 min 23 s"
  const s = Math.round(ms / 1000);
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}

function desenharTreino(estado) {
  // Mostra a etapa atual, a época, a barra de progresso e o log, a partir das linhas impressas pelo treino
  treinoRodando = estado.rodando;
  botaoTreinar.disabled = estado.rodando || bloqueios.length > 0;
  botaoTreinar.textContent = estado.rodando ? "Treinando..." : "Treinar modelo";
  botaoCancelar.hidden = !estado.rodando;
  document.getElementById("treino-progresso").hidden = !estado.rodando;

  const log = document.getElementById("treino-log");
  log.textContent = estado.linhas.join("\n");
  log.scrollTop = log.scrollHeight;                         // acompanha a última linha
  if (!estado.rodando) return;

  statusTreino.hidden = true;
  const etapa = ultima(estado.linhas, /^Etapa (\d)\/8: (.*)/);            // "Etapa 5/8: treinamento (até 150 épocas)"
  const epoca = ultima(estado.linhas, /^Época (\d+)\/(\d+)/);             // "Época 12/150" (impresso por treinar_rede)
  const validacao = ultima(estado.linhas, /val_accuracy: ([\d.]+)/);      // acurácia na validação da última época
  const fora = ultima(estado.linhas, /^Integrante de fora (\d+)\/(\d+): (.*)/);   // "Integrante de fora 2/5: treinando sem ana"
  const n = etapa ? Number(etapa[1]) : 0;
  const ep = epoca ? Number(epoca[1]) : 0;
  const total = epoca ? Number(epoca[2]) : 150;

  // As etapas demoradas são a 5 (treinamento: de 10% a 50% da barra, conforme as épocas) e a 7 (um treino a mais
  // por integrante, sem ele: de 52% a 98%, conforme o integrante e as épocas)
  const k = fora ? Number(fora[1]) - 1 : 0;
  const pct = n <= 4 ? n * 2 : n === 5 ? 10 + 40 * (ep / total) : n === 6 ? 51
    : n === 7 ? 52 + 46 * (fora ? (k + ep / total) / Number(fora[2]) : 0) : 99;
  document.getElementById("treino-barra").style.width = pct.toFixed(1) + "%";
  document.getElementById("treino-etapa").textContent = etapa ? `Etapa ${n} de 8: ${etapa[2]}` : "Iniciando o Python...";

  let detalhe = `Tempo: ${formatarTempo(Date.now() - estado.inicio)}`;
  if (n === 7 && fora) detalhe = `Integrante de fora ${fora[1]} de ${fora[2]}: ${fora[3]} · ${detalhe}`;
  if ((n === 5 || (n === 7 && fora)) && ep) {
    detalhe = `Época ${ep} de ${total} (a parada antecipada pode parar antes)` +
      (validacao ? ` · acurácia na validação: ${(Number(validacao[1]) * 100).toFixed(0)}%` : "") + ` · ${detalhe}`;
  }
  document.getElementById("treino-detalhe").textContent = detalhe;
}

async function terminarTreino(estado) {
  // Mostra o resultado; se deu certo, recarrega as métricas da documentação e recomeça a média do ao vivo
  statusTreino.hidden = false;
  const duracao = formatarTempo(estado.fim - estado.inicio);
  if (estado.codigo === 0) {
    const r = await (await fetch("/api/resultados")).json();
    mostrarStatus(statusTreino, `Treino concluído em ${duracao}. Acurácia no teste: ` +
      `${(r.acuracias.teste * 100).toFixed(1)}% (${r.totais.teste} fotos). Veja tudo em “Ver resultados do treino”.`);
    historico = [];              // ao_vivo.js: os integrantes podem ter mudado
    historicoEmocoes = [];
    carregarResultados();        // resultados.js: matriz e métricas do treino novo
  } else if (cancelado) {
    mostrarStatus(statusTreino, "Treino cancelado. O modelo anterior continua valendo.", true);
  } else {
    const ultimaLinha = estado.linhas[estado.linhas.length - 1] || "";
    mostrarStatus(statusTreino, `O treino falhou (veja o log abaixo). Última mensagem: ${ultimaLinha}`, true);
  }
  atualizarDataset();
}

// ---------- Início ----------

botaoTreinar.addEventListener("click", iniciarTreino);
botaoCancelar.addEventListener("click", cancelarTreino);

atualizarDataset();
// Se a página foi recarregada no meio de um treino, volta a acompanhar; senão, mostra o log do último treino
fetch("/api/treino").then((r) => r.json()).then((estado) => {
  if (estado.rodando) acompanharTreino();
  else desenharTreino(estado);
});
