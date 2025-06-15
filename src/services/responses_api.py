from openai import OpenAI
import base64
from typing import Dict, Optional, List, AsyncIterator
import time

class ResponsesAPI:
    """OpenAI Responses APIを使用した画像生成サービス"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def generate_with_responses(self, 
                               prompt: str,
                               model: str = "gpt-4o-mini",
                               size: str = "1024x1024",
                               quality: str = "auto",
                               format: str = "png",
                               background: str = "auto",
                               output_compression: Optional[int] = None,
                               moderation: str = "auto",
                               partial_images: Optional[int] = None,
                               stream: bool = False) -> Dict:
        """Responses APIを使用して画像を生成"""
        
        try:
            start_time = time.time()
            
            # ツールパラメータの構築
            tool_params = {
                "type": "image_generation"
            }
            
            # Responses API対応パラメータのみ追加（2025年6月時点）
            if size != "1024x1024":
                tool_params["size"] = size
            if quality != "auto":
                tool_params["quality"] = quality
            if moderation != "auto":
                tool_params["moderation"] = moderation
            if partial_images is not None and stream:
                tool_params["partial_images"] = partial_images
            
            # 以下は未サポートのため削除：response_format, background, output_compression
            
            # APIリクエスト
            if stream:
                return self._generate_stream(prompt, model, tool_params, start_time)
            else:
                response = self.client.responses.create(
                    model=model,
                    input=prompt,
                    tools=[tool_params]
                )
                
                generation_time = time.time() - start_time
                
                # 画像データの抽出
                image_data = []
                revised_prompts = []
                
                for output in response.output:
                    if output.type == "image_generation_call":
                        if hasattr(output, 'result') and output.result:
                            image_data.append(base64.b64decode(output.result))
                        if hasattr(output, 'revised_prompt'):
                            revised_prompts.append(output.revised_prompt)
                
                if not image_data:
                    raise Exception("画像生成に失敗しました")
                
                # 単一画像の場合は従来の形式を保持
                if len(image_data) == 1:
                    return {
                        "image_data": image_data[0],
                        "generation_time": round(generation_time, 2),
                        "revised_prompt": revised_prompts[0] if revised_prompts else None,
                        "prompt": prompt,
                        "response_id": response.id
                    }
                else:
                    return {
                        "images": image_data,
                        "image_count": len(image_data),
                        "generation_time": round(generation_time, 2),
                        "revised_prompts": revised_prompts,
                        "prompt": prompt,
                        "response_id": response.id
                    }
                    
        except Exception as e:
            raise Exception(f"Responses API エラー: {str(e)}")
    
    def _generate_stream(self, prompt: str, model: str, tool_params: Dict, start_time: float):
        """ストリーミング生成（Generator返却）"""
        try:
            stream = self.client.responses.create(
                model=model,
                input=prompt,
                tools=[tool_params],
                stream=True
            )
            
            partial_images = []
            final_image = None
            revised_prompt = None
            
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == "response.image_generation_call.partial_image":
                        # 部分画像の処理
                        partial_data = {
                            "type": "partial",
                            "index": getattr(event, 'partial_image_index', 0),
                            "image_data": base64.b64decode(event.partial_image_b64)
                        }
                        partial_images.append(partial_data)
                        yield partial_data
                        
                    elif event.type == "response.done":
                        # 最終画像の処理
                        generation_time = time.time() - start_time
                        
                        for output in event.output:
                            if output.type == "image_generation_call" and hasattr(output, 'result'):
                                final_image = base64.b64decode(output.result)
                                if hasattr(output, 'revised_prompt'):
                                    revised_prompt = output.revised_prompt
                        
                        yield {
                            "type": "final",
                            "image_data": final_image,
                            "generation_time": round(generation_time, 2),
                            "revised_prompt": revised_prompt,
                            "prompt": prompt,
                            "partial_count": len(partial_images),
                            "response_id": getattr(event, 'id', None)
                        }
                        
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def continue_generation(self,
                          previous_response_id: str,
                          prompt: str,
                          model: str = "gpt-4o-mini",
                          **kwargs) -> Dict:
        """前回の生成を継続（マルチターン）"""
        
        try:
            start_time = time.time()
            
            # ツールパラメータ（Responses API対応パラメータのみ）
            tool_params = {"type": "image_generation"}
            supported_params = {"size", "quality", "moderation", "partial_images"}
            
            for key, value in kwargs.items():
                if value is None:
                    continue
                if key in supported_params:
                    tool_params[key] = value
                # format系、background、output_compressionは未サポートのため無視
            
            response = self.client.responses.create(
                model=model,
                previous_response_id=previous_response_id,
                input=prompt,
                tools=[tool_params]
            )
            
            generation_time = time.time() - start_time
            
            # 画像データの抽出
            for output in response.output:
                if output.type == "image_generation_call" and hasattr(output, 'result'):
                    return {
                        "image_data": base64.b64decode(output.result),
                        "generation_time": round(generation_time, 2),
                        "revised_prompt": getattr(output, 'revised_prompt', None),
                        "prompt": prompt,
                        "response_id": response.id,
                        "previous_response_id": previous_response_id
                    }
            
            raise Exception("継続生成に失敗しました")
            
        except Exception as e:
            raise Exception(f"継続生成エラー: {str(e)}")
    
    def generate_with_context(self,
                            prompt: str,
                            context_images: List[Dict],
                            model: str = "gpt-4o-mini",
                            **kwargs) -> Dict:
        """コンテキスト画像を含めた生成"""
        
        try:
            start_time = time.time()
            
            # 入力コンテンツの構築
            content = [{"type": "input_text", "text": prompt}]
            
            # コンテキスト画像を追加
            for img in context_images:
                if "file_id" in img:
                    content.append({
                        "type": "input_image",
                        "file_id": img["file_id"]
                    })
                elif "base64" in img:
                    content.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img['base64']}"
                    })
                elif "generation_id" in img:
                    content.append({
                        "type": "image_generation_call",
                        "id": img["generation_id"]
                    })
            
            # ツールパラメータ（Responses API対応パラメータのみ）
            tool_params = {"type": "image_generation"}
            supported_params = {"size", "quality", "moderation", "partial_images"}
            
            for key, value in kwargs.items():
                if value is None:
                    continue
                if key in supported_params:
                    tool_params[key] = value
                # format系、background、output_compressionは未サポートのため無視
            
            response = self.client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": content
                }],
                tools=[tool_params]
            )
            
            generation_time = time.time() - start_time
            
            # 画像データの抽出
            for output in response.output:
                if output.type == "image_generation_call" and hasattr(output, 'result'):
                    return {
                        "image_data": base64.b64decode(output.result),
                        "generation_time": round(generation_time, 2),
                        "revised_prompt": getattr(output, 'revised_prompt', None),
                        "prompt": prompt,
                        "response_id": response.id,
                        "context_images_count": len(context_images)
                    }
            
            raise Exception("コンテキスト付き生成に失敗しました")
            
        except Exception as e:
            raise Exception(f"コンテキスト生成エラー: {str(e)}")