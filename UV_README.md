# uvの操作メモ

## 基本操作

### 仮想環境の作成と有効化
```bash
# 仮想環境の作成
uv venv

# 仮想環境の有効化
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
```

### パッケージのインストール
```bash
# 通常のインストール
uv pip install パッケージ名

# 複数パッケージのインストール
uv pip install パッケージ1 パッケージ2

# requirements.txtからのインストール
uv pip install -r requirements.txt

# 開発モードでのインストール
uv pip install -e .
uv pip install -e ".[dev]"  # オプション依存関係も含む
```

### パッケージのアンインストール
```bash
uv pip uninstall パッケージ名
```

### パッケージの更新
```bash
uv pip install --upgrade パッケージ名
```

### パッケージ一覧の表示
```bash
uv pip list
```

### パッケージ情報の表示
```bash
uv pip show パッケージ名
```

### キャッシュの管理
```bash
# キャッシュディレクトリの表示
uv cache dir

# キャッシュの削除
uv cache clean
```

## ロックファイルの管理

### ロックファイルの生成
```bash
uv pip compile pyproject.toml -o requirements.lock
```

### ロックファイルからのインストール
```bash
uv pip install -r requirements.lock
```

## その他の便利な機能

### 高速なインストール
```bash
# 並列インストール（デフォルトで有効）
uv pip install numpy pandas scipy matplotlib
```

### ローカルディレクトリのインストール
```bash
uv pip install /path/to/local/package
```

### シンボリックリンクでのインストール
```bash
uv pip install -e /path/to/local/package
```

### システムパッケージのアップグレード
```bash
uv pip install --upgrade pip setuptools wheel
```
