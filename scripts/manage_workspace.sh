#!/bin/bash
# MIREXワークスペース管理スクリプト
# .mcp_server_dataディレクトリの作成と消去を行います

set -e

# ワークスペースのベースディレクトリを設定
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)/.mcp_server_data"

# サブディレクトリの構造定義
SUBDIRS=(
  "code/src/detectors"
  "code/src/templates"
  "code/improved_versions"
  "datasets/synthesized_v1/audio"
  "datasets/synthesized_v1/labels"
  "datasets/synthetic_v1/audio"
  "datasets/synthetic_v1/labels"
  "datasets/synthetic_poly_v1/audio"
  "datasets/synthetic_poly_v1/labels"
  "output/evaluation_results"
  "output/grid_search_results"
  "output/visualizations"
  "output/scientific_output"
  "db"
)

# 色の定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# ヘルプメッセージを表示
show_help() {
  echo "使用方法: $(basename "$0") [オプション]"
  echo
  echo "オプション:"
  echo "  -c, --create     ワークスペースディレクトリを作成"
  echo "  -d, --delete     ワークスペースディレクトリを削除"
  echo "  -r, --reset      ワークスペースディレクトリをリセット（削除して再作成）"
  echo "  -s, --show       現在のワークスペース構造を表示"
  echo "  --force          確認なしで実行する"
  echo "  -h, --help       このヘルプメッセージを表示"
  echo
  echo "例:"
  echo "  $(basename "$0") --create     # ワークスペースを作成"
  echo "  $(basename "$0") --delete     # ワークスペースを削除（確認あり）"
  echo "  $(basename "$0") --reset      # ワークスペースをリセット（確認あり）"
  echo "  $(basename "$0") --delete --force  # 確認なしでワークスペースを削除"
}

# ワークスペース作成
create_workspace() {
  if [ -d "$WORKSPACE_DIR" ]; then
    echo -e "${YELLOW}ワークスペースはすでに存在します: $WORKSPACE_DIR${NC}"
    return 0
  fi
  
  echo -e "${GREEN}ワークスペースを作成しています: $WORKSPACE_DIR${NC}"
  mkdir -p "$WORKSPACE_DIR"
  
  for subdir in "${SUBDIRS[@]}"; do
    mkdir -p "$WORKSPACE_DIR/$subdir"
    echo -e "${GREEN}ディレクトリを作成しました: $WORKSPACE_DIR/$subdir${NC}"
  done
  
  echo -e "${GREEN}ワークスペースの作成が完了しました。${NC}"
}

# ワークスペース削除
delete_workspace() {
  if [ ! -d "$WORKSPACE_DIR" ]; then
    echo -e "${YELLOW}ワークスペースが存在しません: $WORKSPACE_DIR${NC}"
    return 0
  fi
  
  if [ "$FORCE" != "true" ]; then
    echo -e "${RED}警告: 次のディレクトリを削除します: $WORKSPACE_DIR${NC}"
    echo -n "続行しますか？ [y/N] "
    read -r response
    if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
      echo "中止しました。"
      return 1
    fi
  fi
  
  echo -e "${RED}ワークスペースを削除しています: $WORKSPACE_DIR${NC}"
  rm -rf "$WORKSPACE_DIR"
  echo -e "${GREEN}ワークスペースの削除が完了しました。${NC}"
}

# ワークスペース構造の表示
show_workspace() {
  if [ ! -d "$WORKSPACE_DIR" ]; then
    echo -e "${YELLOW}ワークスペースが存在しません: $WORKSPACE_DIR${NC}"
    return 0
  fi
  
  echo -e "${GREEN}ワークスペース構造:${NC}"
  find "$WORKSPACE_DIR" -type d | sort | sed "s|$WORKSPACE_DIR|.mcp_server_data|"
}

# デフォルト値
ACTION=""
FORCE="false"

# コマンドライン引数の処理
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--create)
      ACTION="create"
      shift
      ;;
    -d|--delete)
      ACTION="delete"
      shift
      ;;
    -r|--reset)
      ACTION="reset"
      shift
      ;;
    -s|--show)
      ACTION="show"
      shift
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "不明なオプション: $1"
      show_help
      exit 1
      ;;
  esac
done

# アクション実行
case "$ACTION" in
  "create")
    create_workspace
    ;;
  "delete")
    delete_workspace
    ;;
  "reset")
    delete_workspace && create_workspace
    ;;
  "show")
    show_workspace
    ;;
  "")
    echo "アクションが指定されていません。"
    show_help
    exit 1
    ;;
esac

exit 0 