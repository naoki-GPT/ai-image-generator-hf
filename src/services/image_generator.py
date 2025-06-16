from openai import OpenAI
import base64
from typing import Dict, Optional, List
import time

class ImageGenerator:
    """GPT Image 1を使用した画像生成サービス"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def generate_image(self,
                      prompt: str,
                      size: str = "1024x1024",
                      quality: str = "auto",
                      format: str = "png",
                      transparent_bg: bool = False,
                      output_compression: Optional[int] = None,
                      moderation: str = "auto",
                      n: int = 1,
                      reference_images: List = None) -> Dict:
        """画像を生成"""
        
        try:
            # 実際のコスト計算をアプリ側で行うため、プレースホルダーのみ
            cost_estimate = {"cost_usd": "計算中", "cost_jpy": "計算中", "tokens": "N/A"}
            
            # 生成パラメータの準備
            generation_params = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "n": max(1, min(n, 10))  # 1-10の範囲で制限
            }
            
            # オプションパラメータの追加（SDK互換性考慮）
            if format != "png":
                generation_params["format"] = format
                
            if transparent_bg and format in ["png", "webp"]:
                generation_params["background"] = "transparent"
            
            # 圧縮設定（JPEG/WebPのみ有効）
            if output_compression is not None and format in ["jpeg", "webp"]:
                if 0 <= output_compression <= 100:
                    generation_params["output_compression"] = output_compression
                else:
                    raise ValueError("output_compressionは0-100の範囲で指定してください")
            
            # モデレーション設定
            if moderation in ["auto", "low"]:
                generation_params["moderation"] = moderation
            elif moderation != "auto":
                raise ValueError("moderationは'auto'または'low'で指定してください")
            
            # 参考画像がある場合もプロンプトに含めて通常生成
            return self._generate_simple(generation_params, cost_estimate, format, output_compression)
                
        except Exception as e:
            raise Exception(f"画像生成エラー: {str(e)}")
    
    def _generate_simple(self, params: Dict, cost_estimate: Dict, target_format: str = "png", compression: int = None) -> Dict:
        """シンプルな画像生成（format互換性対応）"""
        start_time = time.time()
        
        try:
            # 新しいSDKでformat引数を試行
            response = self.client.images.generate(**params)
        except TypeError as e:
            if 'format' in str(e) or 'output_compression' in str(e):
                # 古いSDKの場合：format関連パラメータを除去してPNGで生成
                fallback_params = {k: v for k, v in params.items() 
                                 if k not in ['format', 'output_compression', 'background']}
                response = self.client.images.generate(**fallback_params)
                # 後でPILで形式変換する（実装は後述）
            else:
                raise
        
        generation_time = time.time() - start_time
        
        # 画像データの取得（複数対応）
        images = []
        for data in response.data:
            image_base64 = data.b64_json
            image_data = base64.b64decode(image_base64)
            
            # 形式変換が必要な場合（古いSDK対応）
            if target_format != "png" and target_format in ["jpeg", "webp"]:
                image_data = self._convert_image_format(image_data, target_format, compression)
            
            images.append(image_data)
        
        # 単一画像の場合は従来の形式を保持（互換性）
        if len(images) == 1:
            return {
                "image_data": images[0],
                "generation_time": round(generation_time, 2),
                "estimated_cost": f"${cost_estimate['cost_usd']} (¥{cost_estimate['cost_jpy']})",
                "tokens_used": cost_estimate['tokens'],
                "revised_prompt": getattr(response.data[0], 'revised_prompt', None),
                "prompt": params.get('prompt', '')
            }
        else:
            # 複数画像の場合
            return {
                "images": images,
                "image_count": len(images),
                "generation_time": round(generation_time, 2),
                "estimated_cost": f"${cost_estimate['cost_usd']} (¥{cost_estimate['cost_jpy']})",
                "tokens_used": cost_estimate['tokens'],
                "revised_prompt": getattr(response.data[0], 'revised_prompt', None),
                "prompt": params.get('prompt', '')
            }
    
    def generate_with_reference_image(self,
                                    prompt: str,
                                    reference_image_data: bytes,
                                    size: str = "1024x1024",
                                    quality: str = "auto",
                                    format: str = "png",
                                    transparent_bg: bool = False,
                                    output_compression: Optional[int] = None,
                                    moderation: str = "auto") -> Dict:
        """参照画像を使用した画像生成（Image Edit API）"""
        
        try:
            start_time = time.time()
            
            # 画像データをファイルライクオブジェクトに変換
            from io import BytesIO
            image_file = BytesIO(reference_image_data)
            image_file.name = "reference.png"
            
            # 生成パラメータの準備（Image Edit APIは基本パラメータのみ）
            edit_params = {
                "model": "gpt-image-1",
                "image": image_file,
                "prompt": prompt,
                "size": size,
                "quality": quality
            }
            
            # Image Edit APIはresponse_formatとoutput_compressionをサポートしていない
            # PNG以外が必要な場合は後でconvertするか、images.generateにフォールバック
            # 現在はPNGのみ対応
            
            # モデレーション設定（Image Edit APIではサポートされていない可能性があるため注意）
            # if moderation in ["auto", "low"]:
            #     edit_params["moderation"] = moderation
            
            response = self.client.images.edit(**edit_params)
            
            generation_time = time.time() - start_time
            
            # 画像データの取得
            image_base64 = response.data[0].b64_json
            image_data = base64.b64decode(image_base64)
            
            return {
                "image_data": image_data,
                "generation_time": round(generation_time, 2),
                "estimated_cost": "参照画像生成のため計算複雑",
                "tokens_used": "N/A",
                "revised_prompt": getattr(response.data[0], 'revised_prompt', None),
                "prompt": prompt,
                "generation_type": "reference_image"
            }
            
        except Exception as e:
            raise Exception(f"参照画像生成エラー: {str(e)}")
    
    
    def generate_variations(self, original_prompt: str, num_variations: int = 3, **kwargs) -> List[Dict]:
        """バリエーション生成"""
        variations = []
        
        variation_prompts = [
            f"{original_prompt}, variation 1 with different composition",
            f"{original_prompt}, variation 2 with different color scheme", 
            f"{original_prompt}, variation 3 with different style approach"
        ]
        
        for i, prompt in enumerate(variation_prompts[:num_variations]):
            try:
                result = self.generate_image(prompt=prompt, **kwargs)
                result['variation_number'] = i + 1
                variations.append(result)
            except Exception as e:
                print(f"バリエーション{i+1}の生成に失敗: {e}")
                
        return variations
    
    
    def validate_prompt(self, prompt: str) -> Dict:
        """プロンプトの妥当性チェック"""
        issues = []
        suggestions = []
        
        # 長さチェック
        if len(prompt) < 10:
            issues.append("プロンプトが短すぎます")
            suggestions.append("より詳細な説明を追加してください")
        
        if len(prompt) > 4000:
            issues.append("プロンプトが長すぎます")
            suggestions.append("簡潔にまとめてください")
        
        # 不適切なコンテンツチェック（基本的なもの）
        prohibited_words = ["violence", "gore", "explicit", "暴力", "露骨"]
        for word in prohibited_words:
            if word in prompt.lower():
                issues.append(f"不適切な内容が含まれている可能性があります: {word}")
        
        # 改善提案
        if "color" not in prompt.lower() and "色" not in prompt:
            suggestions.append("色に関する指示を追加することを検討してください")
        
        if "style" not in prompt.lower() and "スタイル" not in prompt:
            suggestions.append("スタイルに関する指示を追加することを検討してください")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "score": max(0, 100 - len(issues) * 20 - len(suggestions) * 5)
        }
    
    def optimize_prompt(self, original_prompt: str) -> str:
        """プロンプトの最適化"""
        optimized = original_prompt
        
        # 品質向上キーワードの追加
        quality_keywords = [
            "high quality", "professional", "detailed", "crisp", "clear"
        ]
        
        # 既に含まれていないキーワードを追加
        for keyword in quality_keywords:
            if keyword not in optimized.lower():
                optimized += f", {keyword}"
        
        return optimized
    
    def edit_image(self, image_data: bytes, prompt: str, size: str = "1024x1024", quality: str = "auto") -> Dict:
        """画像の編集"""
        try:
            start_time = time.time()
            
            # 画像データをファイルライクオブジェクトに変換
            from io import BytesIO
            image_file = BytesIO(image_data)
            image_file.name = "image.png"
            
            response = self.client.images.edit(
                model="gpt-image-1",
                image=image_file,
                prompt=prompt,
                size=size,
                quality=quality
            )
            
            generation_time = time.time() - start_time
            
            # 画像データの取得（base64形式）
            image_base64 = response.data[0].b64_json
            image_data = base64.b64decode(image_base64)
            
            return {
                "image_data": image_data,
                "generation_time": round(generation_time, 2),
                "estimated_cost": "編集のため計算複雑",
                "tokens_used": "N/A",
                "revised_prompt": getattr(response.data[0], 'revised_prompt', None),
                "prompt": prompt
            }
            
        except Exception as e:
            raise Exception(f"画像編集エラー: {str(e)}")
    
    def create_variation(self, image_data: bytes, size: str = "1024x1024", quality: str = "hd") -> Dict:
        """画像のバリエーション生成"""
        try:
            start_time = time.time()
            
            # バリエーション生成は1024x1024のみサポート
            variation_size = "1024x1024"
            
            # 画像データをファイルライクオブジェクトに変換
            from io import BytesIO
            image_file = BytesIO(image_data)
            image_file.name = "image.png"
            
            # バリエーション生成APIはqualityパラメータをサポートしていない
            response = self.client.images.create_variation(
                image=image_file,
                n=1,
                size=variation_size
            )
            
            generation_time = time.time() - start_time
            
            # 画像データの取得（base64形式またはURL）
            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                image_base64 = response.data[0].b64_json
                image_data = base64.b64decode(image_base64)
            else:
                # URLから画像をダウンロード
                import requests
                image_url = response.data[0].url
                image_response = requests.get(image_url)
                image_data = image_response.content
            
            return {
                "image_data": image_data,
                "generation_time": round(generation_time, 2),
                "estimated_cost": "バリエーションのため計算複雜",
                "tokens_used": "N/A"
            }
            
        except Exception as e:
            raise Exception(f"バリエーション生成エラー: {str(e)}")
    
    def _convert_image_format(self, image_data: bytes, target_format: str, compression: int = None) -> bytes:
        """PIL使用して画像形式を変換（古いSDK対応）"""
        try:
            from PIL import Image
            from io import BytesIO
            
            # PNG画像データをPILで読み込み
            image = Image.open(BytesIO(image_data))
            
            # RGBA to RGB変換（JPEG用）
            if target_format == "jpeg" and image.mode in ("RGBA", "LA"):
                # 白背景で合成
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "RGBA":
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image)
                image = background
            
            # バイトストリームに保存
            output = BytesIO()
            save_kwargs = {}
            
            if target_format == "jpeg":
                save_format = "JPEG"
                if compression is not None:
                    save_kwargs["quality"] = compression
                else:
                    save_kwargs["quality"] = 90  # デフォルト品質
            elif target_format == "webp":
                save_format = "WebP" 
                if compression is not None:
                    save_kwargs["quality"] = compression
                else:
                    save_kwargs["quality"] = 90  # デフォルト品質
            else:
                save_format = "PNG"
            
            image.save(output, format=save_format, **save_kwargs)
            return output.getvalue()
            
        except Exception as e:
            # 変換に失敗した場合は元の画像データを返す
            print(f"画像形式変換エラー: {e}")
            return image_data