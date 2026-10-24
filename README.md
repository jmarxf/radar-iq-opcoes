# Radar IQ — Opções B3 | versão online

App em Streamlit para leitura de posições em aberto e cálculo do:

**IQ = Titulares / Lançadores**

## Arquivos necessários para publicar

- `streamlit_app.py`
- `requirements.txt`

Os demais arquivos são apenas documentação.

## Fonte de dados

O app lê diretamente o dataset público configurado no código:

`b3_opcoes_posicoes_aberto.csv.gz`

Não é necessário subir `exemplo_COGN.csv`.

## Publicação no Streamlit Community Cloud

1. Crie um repositório no GitHub.
2. Envie os arquivos desta pasta para a raiz do repositório.
3. Entre em `https://share.streamlit.io`.
4. Conecte sua conta GitHub.
5. Clique em **Create app**.
6. Escolha o repositório e a branch.
7. Em **Main file path**, use:

   `streamlit_app.py`

8. Clique em **Deploy**.

## Importante sobre histórico

O arquivo `data/historico_iq.csv` funciona normalmente enquanto a instância
estiver ativa, mas o armazenamento local do Streamlit Community Cloud não deve
ser tratado como banco de dados permanente.

Para histórico confiável entre dias, uma próxima versão pode usar Google
Sheets, Supabase ou outro armazenamento persistente.

## Arquivos que NÃO são necessários online

- `exemplo_COGN.csv`
- `TESTAR_CALCULO.py`
- `INICIAR_APP.bat`

Eles serviam somente para teste/local.
