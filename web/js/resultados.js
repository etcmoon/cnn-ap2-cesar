// Janela "Ver resultados do treino": matriz de confusão, métricas e teste do integrante de fora,
// montados a partir de /api/resultados (resultados.json, gravado por src/treinar.py).

const porcento = (v) => (v * 100).toFixed(1).replace(".", ",") + "%";   // 0.953 -> "95,3%"

function escaparHTML(texto) {
  // Evita que um nome de pasta com < ou & quebre o HTML
  return String(texto).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function tabela(cabecalho, linhas, classeUltimaLinha) {
  // Monta o HTML de uma tabela simples a partir de listas de textos
  const th = cabecalho.map((c) => `<th>${c}</th>`).join("");
  const trs = linhas.map((linha, i) => {
    const classe = classeUltimaLinha && i === linhas.length - 1 ? ` class="${classeUltimaLinha}"` : "";
    return `<tr${classe}>` + linha.map((c) => `<td>${c}</td>`).join("") + "</tr>";
  }).join("");
  return `<div class="tabela-rolagem"><table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table></div>`;
}

function preencherResultados(r) {
  const nomes = r.classes.map(escaparHTML);
  const m = r.metricas;

  // Matriz de confusão: a cor de cada célula é a porcentagem da linha; a última coluna é "desconhecido"
  let html = '<div class="tabela-rolagem"><table class="matriz"><thead><tr><th class="canto">real ↓ / previsto →</th>' +
    nomes.concat(["desconhecido"]).map((n) => `<th>${n}</th>`).join("") + "</tr></thead><tbody>";
  r.matriz_confusao.forEach((linha, i) => {
    const totalLinha = Math.max(linha.reduce((a, b) => a + b, 0), 1);
    html += `<tr><th>${nomes[i]}</th>`;
    linha.forEach((valor, j) => {
      const p = valor / totalLinha;
      const classe = (i === j ? "diagonal" : "") + (p > 0.5 ? " forte" : "");
      html += `<td class="${classe}" style="--p: ${(p * 100).toFixed(0)}%">${valor}<small>${(p * 100).toFixed(0)}%</small></td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table></div>";

  // Métricas por integrante + média macro
  const media = (lista) => lista.reduce((a, b) => a + b, 0) / lista.length;
  const linhas = r.classes.map((_, i) => [nomes[i], m.VP[i], m.FP[i], m.FN[i], m.VN[i],
    porcento(m.precisao[i]), porcento(m.recall[i]), porcento(m.f1[i]), porcento(m.especificidade[i])]);
  linhas.push(["<strong>média macro</strong>", "", "", "", "", porcento(media(m.precisao)),
    porcento(media(m.recall)), porcento(media(m.f1)), porcento(media(m.especificidade))]);
  html += tabela(["Integrante", "VP", "FP", "FN", "VN", "Precisão", "Recall", "F1", "Especificidade"], linhas, "total");

  const a = r.acuracias;
  html += `<p class="nota">Acurácia no teste: <strong>${porcento(a.teste)}</strong> (${r.totais.teste} fotos) · ` +
    `treino ${porcento(a.treino)} · validação ${porcento(a.validacao)}. Uma diferença grande entre treino e validação ` +
    `indica overfitting. Limiar do “desconhecido”: cosseno ${r.limiar.toFixed(2).replace(".", ",")}. ` +
    `Treino de ${escaparHTML(r.gerado_em)}.</p>`;

  // Teste do integrante de fora: a rede treinada sem a pessoa deveria chamá-la de "desconhecido"
  if (r.integrante_de_fora.length) {
    html += "<h3>Desconhecidos simulados (integrante de fora)</h3>" +
      tabela(["Treinada sem", "Fotos dele(a)", "Viraram “desconhecido”", "Outros reconhecidos certo"],
        r.integrante_de_fora.map((f) => [escaparHTML(f.integrante), f.fotos, porcento(f.rejeitadas), porcento(f.conhecidos_corretos)]));
  } else {
    html += '<p class="nota">O teste do integrante de fora precisa de 3 integrantes ou mais.</p>';
  }
  document.getElementById("janela-matriz-conteudo").innerHTML = html;
}

async function carregarResultados() {
  // Busca o resultados.json; se o modelo ainda não foi treinado, a janela mostra o aviso
  const conteudo = document.getElementById("janela-matriz-conteudo");
  try {
    const resposta = await fetch("/api/resultados");
    const r = await resposta.json();
    if (!resposta.ok) throw new Error(r.erro);
    preencherResultados(r);
  } catch (erro) {
    conteudo.innerHTML = "";
    const p = document.createElement("p");
    p.className = "sem-imagem";
    p.textContent = erro.message;
    conteudo.appendChild(p);
  }
}

// Botão "Ver resultados do treino": abre a janela; clicar fora dela também fecha
const janelaMatriz = document.getElementById("janela-matriz");
document.getElementById("ver-matriz").addEventListener("click", () => janelaMatriz.showModal());
janelaMatriz.addEventListener("click", (evento) => { if (evento.target === janelaMatriz) janelaMatriz.close(); });
carregarResultados();
