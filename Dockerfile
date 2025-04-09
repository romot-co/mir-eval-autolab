# ベースイメージ（適切なPythonバージョンを選択 - 例: 3.9）
FROM python:3.9-slim

# 環境変数（非対話モードなど）
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 作業ディレクトリ設定
WORKDIR /app

# システム依存関係インストール (オプション - 必要に応じて追加)
# 例: git やビルドツールが必要な場合
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Python依存関係インストール
# まず requirements.txt をコピーしてインストール（キャッシュを活用）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトコード全体をコピー
COPY . .

# 環境変数設定（ビルド時引数から設定、または実行時に -e で渡す）
# .env ファイルを直接コンテナに含めるのは非推奨
ARG MCP_SERVER_URL=http://localhost:5002
ARG ANTHROPIC_API_KEY=""
ARG OPENAI_API_KEY=""
ARG MIREX_WORKSPACE=/app/mcp_workspace
# ... 他に必要な .env 変数 ...

ENV MCP_SERVER_URL=${MCP_SERVER_URL} \
    ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
    OPENAI_API_KEY=${OPENAI_API_KEY} \
    MIREX_WORKSPACE=${MIREX_WORKSPACE}
# ... 他の ENV ...

# ワークスペースディレクトリを作成 (ボリュームマウントしない場合)
RUN mkdir -p ${MIREX_WORKSPACE}

# ポート公開
EXPOSE 5002

# デフォルトの起動コマンド (mcp_server.py を実行)
# エントリーポイントスクリプトは必要に応じて変更
CMD ["python", "mcp_server.py", "--port", "5002"]