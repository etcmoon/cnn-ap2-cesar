# Confere a sintaxe dos arquivos JavaScript com o Node: um erro de sintaxe impede o arquivo inteiro de carregar
# no navegador (e as funções dele somem para os outros arquivos, como o atualizarDataset do treino.js).
# Rodar na raiz do repositório com:  pytest
import os
import glob
import shutil
import subprocess

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVOS = sorted(glob.glob(os.path.join(RAIZ, "web", "js", "*.js"))) + [os.path.join(RAIZ, "servidor.js")]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js não instalado")
@pytest.mark.parametrize("arquivo", ARQUIVOS, ids=os.path.basename)
def test_javascript_sem_erro_de_sintaxe(arquivo):
    resultado = subprocess.run(["node", "--check", arquivo], capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stderr
