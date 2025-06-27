# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Lython は、TypeScript レベルの厳密な静的型チェックを提供し、C/C++レベルのパフォーマンスを実現する AOT コンパイルを備えた Python コンパイラツールチェーンです。LLVM/MLIR 上に構築され、Python からネイティブコードへの多段階コンパイルパイプラインを実装しています。

## 一般的な開発コマンド

```bash
# 環境セットアップ
uv sync                    # 依存関係のインストール

# コード品質
black src/                 # Pythonコードのフォーマット
isort src/                 # インポートのソート
mypy src/lython           # 型チェック
flake8 src/               # リンティング

# 将来のビルドコマンド (実装時)
cmake --build build/       # MLIR/LLVMコンポーネントのビルド
lythonc sample.py         # Pythonをネイティブコードにコンパイル
```

## アーキテクチャ概要

### コンパイルパイプライン

プロジェクトは多層アプローチを採用しています:

1. **フロントエンド**: Python AST → 厳密な型推論を伴う TypedAST
2. **Python 方言**: 型認識 Python 操作のためのカスタム MLIR 方言
3. **高レベル IR**: 構造化操作のための SCF、tensor、linalg 方言
4. **低レベル IR**: バッファ操作のための memref、arith 方言
5. **バックエンド**: LLVM 方言 → LLVM IR → ネイティブコード

### 主なコンポーネント

- **`src/lython/`**: コア Python パッケージ（現在は最小限）
- **`docs/`**: 包括的なドキュメント、以下を含む：
  - `STRUCT.md`: トップレベルのアーキテクチャ概要
  - `GUIDE.md`: 詳細な実装ガイド
  - `LOADMAP.md`: 2 年間の開発ロードマップ
- **`lowering_sample/`**: コンパイルパイプラインを示す MLIR のローワリング例
- **`build/`**: LLVM 統合を含む CMake ビルドディレクトリ

### 技術スタック

- **Python 3.12+**: フロントエンドとツール
- **MLIR**: 多段階 IR コンパイル
- **LLVM 20+**: コード生成と最適化
- **UV**: 依存関係管理
- **CMake**: ビルドシステム統合

## 開発状況

現在フェーズ 0-1 (初期開発段階):

- ✅ リポジトリ構造とドキュメント
- ✅ 基本的な Python パッケージセットアップ
- ✅ MLIR ローワリング例
- 🔄 TypedAST コア実装
- ⏳ MLIR Python 方言の開発

## 主要なアーキテクチャ原則

### デュアル型システム

Lython は、`@native` デコレータシステムを通じて PyObject 互換性とネイティブ型の両方を維持します:

```python
from native import native, i32

@native
def add(a: i32, b: i32) -> i32:
    return a + b
```

### 型安全性

Hindley-Milner スタイルの型推論を実装し、Python 構文互換性を維持しながら TypeScript レベルの静的型チェックを提供します。

### パフォーマンス目標

以下を通じて CPython に対して 50 倍のパフォーマンス向上を目指します:

- ネイティブコードへの AOT コンパイル
- シャドウスタック GC 統合
- MLIR ベースの最適化パス

## 重要なファイル

- **`pyproject.toml`**: Python パッケージの設定と依存関係
- **`uv.lock`**: 依存関係ロックファイル
- **`docs/STRUCT.md`**: 全体的なアーキテクチャを理解するための必読資料
- **`docs/GUIDE.md`**: 詳細な実装ガイド
- **`lowering_sample/`**: MLIR コンパイルパターンのリファレンス実装

## テストアプローチ

計画されている多層テスト:

- **ユニットテスト**: フロントエンド/型推論用の pytest
- **MLIR テスト**: 方言/パスのテスト用 lit (LLVM 統合テスター)
- **統合テスト**: コンパイルパイプラインのエンドツーエンドテスト
- **ベンチマーク**: CPython に対するパフォーマンス検証

## 開発ガイド

- **新機能の追加**: 大規模なものは `docs/rfcs/` に RFC を作成
- **バグ修正**: `tests/` に対応するユニットテストを追加
- **コードスタイル**: `isort ... && black ...` の実行
- **ドキュメント**: 変更に応じて更新
- **言語の使い分け**: コメントやドキュメント類、ユーザーへの応答は日本語で記述、コンソール出力やコード例は英語で統一
