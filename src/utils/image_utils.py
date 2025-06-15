"""画像処理ユーティリティ（必要最小限）"""

import base64
from PIL import Image
from io import BytesIO

def encode_image_to_base64(image_path):
    """画像ファイルをBase64エンコード"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def decode_base64_to_image(base64_string):
    """Base64文字列を画像オブジェクトに変換"""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def get_image_info(image_data):
    """画像の情報を取得"""
    if isinstance(image_data, str):
        image = decode_base64_to_image(image_data)
    else:
        image = Image.open(BytesIO(image_data))
    
    return {
        'size': image.size,
        'mode': image.mode,
        'format': image.format
    }