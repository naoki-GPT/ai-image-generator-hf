"""設定管理モジュール（必要最小限）"""

# 画像生成に関する定数定義
IMAGE_SIZES = {
    "1024x1024": {"width": 1024, "height": 1024, "description": "正方形（標準）"},
    "1536x1024": {"width": 1536, "height": 1024, "description": "横長（ワイド）"},
    "1024x1536": {"width": 1024, "height": 1536, "description": "縦長（ポートレート）"}
}

# 品質設定
QUALITY_SETTINGS = {
    "auto": "自動（推奨）",
    "low": "低品質（高速）",
    "medium": "中品質",
    "high": "高品質",
    "very-high": "最高品質"
}

# フォーマット設定
FORMAT_SETTINGS = {
    "png": "PNG（透過対応）",
    "jpeg": "JPEG（高圧縮）",
    "webp": "WebP（最新形式）"
}