# ベースイメージ (pyproject.toml の requires-python に合わせる)
FROM python:3.10-slim

# uv をインストール
RUN pip install --no-cache-dir uv

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

# --- 依存関係インストール (uv とロックファイルを使用) ---
# 1. pyproject.toml をコピー (これだけでも uv は動作するが、ロックファイル推奨)
COPY pyproject.toml ./

# 2. (推奨) ロックファイルをコピー (事前に生成しておくこと)
#    例: requirements-lock.txt を使う場合
COPY requirements-lock.txt ./
#    例: requirements.txt と requirements-dev.txt を使う場合
# COPY requirements.txt requirements-dev.txt ./

# 3. uv で依存関係を同期 (ロックファイルから)
#    例: requirements-lock.txt を使う場合 (--system でシステムPythonにインストール)
RUN uv pip sync --system --no-cache requirements-lock.txt
#    例: requirements.txt と requirements-dev.txt を使う場合
# RUN uv pip sync --system --no-cache requirements.txt requirements-dev.txt
#    ロックファイルを使わない場合 (ビルドごとに解決が発生)
# RUN uv pip install --system --no-cache .[all] # all は適宜調整 (dev, numba, crepe など)

# --- プロジェクトコードコピー ---
COPY . .
# setup.py が不要な場合は COPY setup.py . は不要

# --- 環境変数設定 --- 
# .env ファイルを直接含めず、実行時に渡すか、他の方法 (secrets等) を検討
ARG MCP_SERVER_URL=http://localhost:5002
# ARG ANTHROPIC_API_KEY="" # ビルド時引数は機密情報には非推奨
# ARG OPENAI_API_KEY=""
ARG MIREX_WORKSPACE=/app/mcp_workspace
# ... 他の必要なビルド時引数 ...

ENV MCP_SERVER_URL=${MCP_SERVER_URL} \
    MIREX_WORKSPACE=${MIREX_WORKSPACE} \
    # PYTHONPATH: /app と /app/src を Python がモジュール検索できるように設定
    PYTHONPATH="/app:/app/src"

# ANTHROPIC_API_KEY や OPENAI_API_KEY は実行時に渡す
# 例: docker run -e ANTHROPIC_API_KEY="your_key" ...
# ... 他の ENV (実行時に設定するものが主) ...

# ワークスペースディレクトリを作成 (ボリュームマウントしない場合)
RUN mkdir -p ${MIREX_WORKSPACE}

# ポート公開
EXPOSE 5002

# デフォルトの起動コマンド (mcp_server.py を実行)
CMD ["python", "mcp_server.py", "--port", "5002"]