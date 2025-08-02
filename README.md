---
title: AI Image Generator
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.4.0
app_file: app.py
pinned: false
license: mit
---

# AI画像生成アプリ

OpenAIのGPT Image 1を活用した高品質画像生成アプリケーション。

## 主な機能

- **3つの生成モード**: プロンプト直接入力、画像参照生成、AIチャット
- **高度なプロンプト生成**: YAML形式の詳細プロンプトを自動生成
- **複数画像生成**: 一度に最大4枚の画像を同時生成
- **フォーマット対応**: PNG、JPEG、WebP形式をサポート
- **透明背景対応**: PNG/WebP形式で透明背景の生成が可能

## 使い方

1. **APIキー設定**: OpenAI APIキーを入力
2. **モード選択**: 3つの生成モードから選択
3. **設定調整**: サイズ、品質、形式などを設定
4. **画像生成**: プロンプトを入力して生成ボタンをクリック

## 技術スタック

- **フロントエンド**: Gradio 4.44.1
- **画像生成**: OpenAI GPT Image 1 API
- **画像処理**: Pillow (PIL)
- **設定管理**: YAML

## ライセンス

MIT License