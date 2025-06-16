import gradio as gr
import os
import sys
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime
import tempfile
import json
import time
from pathlib import Path

# Gradioバージョン確認
try:
    print(f"Gradio version: {gr.__version__}")
    import gradio_client
    print(f"Gradio Client version: {gradio_client.__version__}")
except Exception as e:
    print(f"バージョン確認エラー: {e}")

# Gradioのバグ修正用猿パッチ（ChatGPT推奨）
try:
    import gradio_client.utils as _gcu
    _orig = _gcu._json_schema_to_python_type  # keep ref

    def _patched(schema, defs=None):
        if isinstance(schema, bool):          # ← 追加
            schema = {}                       # bool → 空dict に変換
        return _orig(schema, defs or {})

    _gcu._json_schema_to_python_type = _patched
    print("Gradio猿パッチを適用しました")
except Exception as e:
    print(f"Gradio猿パッチの適用に失敗: {e}")

# OpenAI proxies引数互換パッチ（古いSDK用）
try:
    import inspect
    import functools
    from openai import OpenAI
    
    if 'proxies' in inspect.signature(OpenAI.__init__).parameters:
        OpenAI.__init__ = functools.partialmethod(OpenAI.__init__, proxies=None)
        print("OpenAI proxies猿パッチを適用しました")
    else:
        print("OpenAI proxies引数は存在しません（新しいSDK）")
except Exception as e:
    print(f"OpenAI猿パッチの適用に失敗: {e}")

# HuggingFace Spaces用パス設定
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# HuggingFace Spaces用プロキシ設定クリーンアップ
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(key, None)

try:
    from src.services.image_generator import ImageGenerator
    from src.services.responses_api import ResponsesAPI
except ImportError as e:
    print(f"モジュールのインポートエラー: {e}")

# 設定定数
APP_CONFIG = {
    'title': 'AI画像生成',
    'default_compression': 80
}

# GPT Image 1価格設定（2025年6月最新価格）
GPT_IMAGE_PRICING = {
    'low': {
        '1024x1024': 0.011,
        '1024x1536': 0.016,
        '1536x1024': 0.016
    },
    'medium': {
        '1024x1024': 0.042,
        '1024x1536': 0.063,
        '1536x1024': 0.063
    },
    'hd': {  # high
        '1024x1024': 0.167,
        '1024x1536': 0.25,
        '1536x1024': 0.25
    }
}

def calculate_image_cost(size, quality, image_count=1):
    """サイズと品質に基づいて正確なコストを計算"""
    try:
        # 品質マッピング
        quality_map = {
            'auto': 'medium',  # デフォルトはmedium
            'standard': 'medium',
            'hd': 'hd'
        }
        
        mapped_quality = quality_map.get(quality, 'medium')
        cost_per_image = GPT_IMAGE_PRICING[mapped_quality][size]
        total_cost_usd = cost_per_image * image_count
        total_cost_jpy = total_cost_usd * 150  # 1USD = 150円で概算
        
        return {
            'cost_usd': f"{total_cost_usd:.3f}",
            'cost_jpy': f"{total_cost_jpy:.1f}",
            'per_image_usd': f"{cost_per_image:.3f}"
        }
    except Exception as e:
        # フォールバック価格
        fallback_cost = 0.042 * image_count
        return {
            'cost_usd': f"{fallback_cost:.3f}",
            'cost_jpy': f"{fallback_cost * 150:.1f}",
            'per_image_usd': "0.042"
        }

SIZE_MAP = {
    "1024x1024 (正方形)": "1024x1024",
    "1024x1536 (縦長)": "1024x1536", 
    "1536x1024 (横長)": "1536x1024"
}

def get_app_css():
    """アプリケーション用CSS"""
    return """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    .gr-button-primary {
        background: #0066cc !important;
        color: white !important;
        border: none !important;
    }
    
    .gr-button-secondary {
        background: #4a4a4a !important;
        color: white !important;
        border: none !important;
    }
    
    .gr-textbox, .gr-dropdown, .gr-slider {
        background: #2d2d2d !important;
        color: white !important;
        border: 1px solid #4a4a4a !important;
    }
    
    .gr-panel {
        background: #1e1e1e !important;
        border: 1px solid #333 !important;
    }
    
    .gr-box {
        border-radius: 8px !important;
    }
    
    .gr-form {
        background: #1e1e1e !important;
    }
    
    .gr-input, .gr-dropdown {
        background: #2d2d2d !important;
        color: white !important;
    }
    
    .gr-checkbox {
        accent-color: #0066cc !important;
    }
    
    .gradio-container h1, .gradio-container h2, .gradio-container h3 {
        color: #ffffff !important;
    }
    
    .gr-markdown {
        background: transparent !important;
    }
    
    .gr-markdown p, .gr-markdown li {
        color: #e0e0e0 !important;
    }
    
    .gr-image {
        border-radius: 8px !important;
    }
    
    .gr-gallery {
        background: #1e1e1e !important;
    }
    
    /* スクロールバーのスタイル */
    ::-webkit-scrollbar {
        width: 8px;
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a4a4a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #666;
    }
    """


def validate_api_key(api_key):
    """APIキー検証（高速）"""
    return bool(api_key and api_key.startswith('sk-')), "APIキーを入力してください" if not api_key else ""

def build_simple_prompt(purpose, message, style, colors, elements, additional):
    """シンプルで高速なプロンプト生成"""
    parts = []
    
    # 基本構造
    if message:
        parts.append(f"Create: {message}")
    if purpose:
        parts.append(f"for {purpose}")
    if style:
        parts.append(f"in {style} style")
    if colors:
        parts.append(f"using colors: {colors}")
    if elements:
        parts.append(f"including: {elements}")
    if additional:
        parts.append(additional)
    
    # 品質向上の基本指示
    parts.append("high quality, professional, detailed")
    
    return ", ".join(parts)

def create_optimized_app():
    # アプリケーション状態の初期化（完全版）
    app_state = {
        'generation_history': [],
        'current_image': None,
        'current_prompt': "",
        'original_prompt': "",  # 元のプロンプトを保存
        'api_key': None,
        'last_response_id': None,  # Responses API用
        'generation_context': [],  # マルチターン用のコンテキスト
        'api_mode': 'image',  # 'image' or 'responses'
        'current_size': '1024x1024 (正方形)'
    }
    
    def generate_image_fast(api_key, purpose, message, style, colors, elements, additional, 
                           size, quality, format_opt, transparent, compression, moderation, image_count, use_ai_mode, enable_responses_api):
        """高速画像生成（AI緻密設計モード無効化で高速化）"""
        try:
            valid, error_msg = validate_api_key(api_key)
            if not valid:
                return None, f"❌ {error_msg}", "", "", None
            
            os.environ["OPENAI_API_KEY"] = api_key
            app_state['api_key'] = api_key
            app_state['current_size'] = size  # サイズ情報を保存
            
            # シンプルなプロンプト生成（高速）
            if use_ai_mode:
                # AI緻密設計モードでもシンプル化
                prompt = build_simple_prompt(purpose, message, style, colors, elements, additional)
            else:
                parts = [purpose, message, style, colors, elements, additional]
                prompt = ", ".join([p for p in parts if p.strip()])
            
            if not prompt.strip():
                return None, "❌ エラー: 内容を入力してください", "", "", None
            
            # 画像生成（API選択）
            size_key = SIZE_MAP.get(size, "1024x1024")
            
            if enable_responses_api:
                # Responses API使用（対話型編集可能）
                responses_api = ResponsesAPI(api_key)
                result = responses_api.generate_with_responses(
                    prompt=prompt,
                    size=size_key,
                    quality=quality,
                    format=format_opt,
                    background="transparent" if transparent else "auto",
                    output_compression=compression if format_opt in ["jpeg", "webp"] else None,
                    moderation=moderation
                )
            else:
                # Image API使用（従来通り）
                generator = ImageGenerator(api_key)
                result = generator.generate_image(
                    prompt=prompt,
                    size=size_key,
                    quality=quality,
                    format=format_opt,
                    transparent_bg=transparent,
                    output_compression=compression if format_opt in ["jpeg", "webp"] else None,
                    moderation=moderation,
                    n=int(image_count)
                )
            
            # 複数画像対応
            if 'images' in result:
                # 複数画像の場合：最初の画像を表示用に使用
                image = Image.open(BytesIO(result['images'][0]))
                
                # 全ての画像を履歴に保存
                for i, img_data in enumerate(result['images']):
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_opt}")
                    img = Image.open(BytesIO(img_data))
                    img.save(temp_file.name, format=format_opt.upper() if format_opt != 'jpeg' else 'JPEG')
                    
                    history_item = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'image_data': img_data,
                        'prompt': f"{prompt} (画像{i+1}/{len(result['images'])})",
                        'purpose': purpose or "画像生成",
                        'style': style or "標準",
                        'temp_file': temp_file.name
                    }
                    app_state['generation_history'].append(history_item)
                
                app_state['current_image'] = {'image_data': result['images'][0], 'prompt': prompt}
            else:
                # 単一画像の場合（従来通り）
                image = Image.open(BytesIO(result['image_data']))
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_opt}")
                image.save(temp_file.name, format=format_opt.upper() if format_opt != 'jpeg' else 'JPEG')
                
                history_item = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'image_data': result['image_data'],
                    'prompt': prompt,
                    'purpose': purpose or "画像生成",
                    'style': style or "標準",
                    'temp_file': temp_file.name
                }
                app_state['generation_history'].append(history_item)
                app_state['current_image'] = {'image_data': result['image_data'], 'prompt': prompt}
            
            app_state['current_prompt'] = prompt
            app_state['original_prompt'] = prompt  # 元のプロンプトを保存
            
            # Responses API用の状態更新（常に更新）
            if enable_responses_api and 'response_id' in result:
                app_state['last_response_id'] = result['response_id']
                print(f"[DEBUG] 詳細設定: response_id更新 - {result['response_id'][:8]}...")
            elif not enable_responses_api:
                # 従来API使用時はresponse_idをクリア
                app_state['last_response_id'] = None
                print(f"[DEBUG] 詳細設定: 従来API使用、response_idクリア")
            
            # コスト情報（複数画像対応）
            image_count_info = f"x{int(image_count)}" if int(image_count) > 1 else ""
            cost_data = calculate_image_cost(size_value, quality, int(image_count))
            
            cost_info = f"""**生成完了** ⚡
**時間**: {result.get('generation_time', 'N/A')}秒
**画像数**: {result.get('image_count', 1)}枚
**コスト**: 約${cost_data['cost_usd']} (¥{cost_data['cost_jpy']})
**モード**: {'AI最適化' if use_ai_mode else '直接'}
**詳細**: {size_value}, {quality}品質"""
            
            return image, "✅ 画像生成完了！", cost_info, prompt
            
        except Exception as e:
            return None, f"❌ 生成エラー: {str(e)}", "", ""
    
    def generate_from_prompt_fast(api_key, prompt, size, quality, format_opt, transparent, compression, moderation, image_count, enable_responses_api):
        """プロンプト直接生成（最高速）"""
        try:
            valid, error_msg = validate_api_key(api_key)
            if not valid:
                return None, f"❌ {error_msg}", "", "", None
            
            if not prompt.strip():
                return None, "❌ プロンプトを入力してください", "", "", None
            
            os.environ["OPENAI_API_KEY"] = api_key
            app_state['api_key'] = api_key
            
            # サイズ情報をapp_stateに保存
            app_state['current_size'] = size
            
            # プロンプトがYAML形式かチェック
            if prompt.strip().startswith('style:') or 'main_texts:' in prompt:
                final_prompt = prompt
            else:
                final_prompt = convert_to_yaml_prompt(prompt, api_key, size)
            
            size_key = SIZE_MAP.get(size, "1024x1024")
            
            if enable_responses_api:
                # Responses API使用（対話型編集可能）
                responses_api = ResponsesAPI(api_key)
                result = responses_api.generate_with_responses(
                    prompt=final_prompt,
                    size=size_key,
                    quality=quality,
                    format=format_opt,
                    background="transparent" if transparent else "auto",
                    output_compression=compression if format_opt in ["jpeg", "webp"] else None,
                    moderation=moderation
                )
            else:
                # Image API使用（従来通り）
                generator = ImageGenerator(api_key)
                result = generator.generate_image(
                    prompt=final_prompt,
                    size=size_key,
                    quality=quality,
                    format=format_opt,
                    transparent_bg=transparent,
                    output_compression=compression if format_opt in ["jpeg", "webp"] else None,
                    moderation=moderation,
                    n=int(image_count)
                )
            
            # 複数画像対応
            if 'images' in result:
                # 複数画像の場合
                image = Image.open(BytesIO(result['images'][0]))
                
                for i, img_data in enumerate(result['images']):
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_opt}")
                    img = Image.open(BytesIO(img_data))
                    img.save(temp_file.name, format=format_opt.upper() if format_opt != 'jpeg' else 'JPEG')
                    
                    history_item = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'image_data': img_data,
                        'prompt': f"{final_prompt} (画像{i+1}/{len(result['images'])})",
                        'purpose': "直接プロンプト",
                        'style': "カスタム",
                        'temp_file': temp_file.name
                    }
                    app_state['generation_history'].append(history_item)
                app_state['current_image'] = {'image_data': result['images'][0], 'prompt': final_prompt}
            else:
                # 単一画像の場合
                image = Image.open(BytesIO(result['image_data']))
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_opt}")
                image.save(temp_file.name, format=format_opt.upper() if format_opt != 'jpeg' else 'JPEG')
                
                history_item = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'image_data': result['image_data'],
                    'prompt': final_prompt,
                    'purpose': "直接プロンプト",
                    'style': "カスタム",
                    'temp_file': temp_file.name
                }
                app_state['generation_history'].append(history_item)
                app_state['current_image'] = {'image_data': result['image_data'], 'prompt': final_prompt}
            
            app_state['current_prompt'] = final_prompt
            app_state['original_prompt'] = final_prompt  # 元のプロンプトを保存
            
            # Responses API用の状態更新（プロンプト直接）
            if enable_responses_api and 'response_id' in result:
                app_state['last_response_id'] = result['response_id']
                print(f"[DEBUG] プロンプト直接: response_id更新 - {result['response_id'][:8]}...")
            elif not enable_responses_api:
                # 従来API使用時はresponse_idをクリア
                app_state['last_response_id'] = None
                print(f"[DEBUG] プロンプト直接: 従来API使用、response_idクリア")
            
            # コスト情報（複数画像対応）
            cost_data = calculate_image_cost(size_value, quality, int(image_count))
            cost_info = f"""**生成完了** ⚡
**時間**: {result.get('generation_time', 'N/A')}秒
**画像数**: {result.get('image_count', 1)}枚
**コスト**: 約${cost_data['cost_usd']} (¥{cost_data['cost_jpy']})
**モード**: 直接プロンプト
**YAML変換**: {'適用済み' if final_prompt != prompt else 'なし'}
**詳細**: {size_value}, {quality}品質"""
            
            return image, "✅ 画像生成完了！", cost_info, final_prompt
            
        except Exception as e:
            return None, f"❌ 生成エラー: {str(e)}", "", ""
    
    def ai_chat_response(api_key, message, chat_history):
        """GPTsライクなAIチャット機能（STEP0-6フロー）"""
        try:
            if not message.strip():
                return chat_history, ""
            
            # APIキー検証
            valid, error_msg = validate_api_key(api_key)
            if not valid:
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": f"❌ {error_msg}"})
                return chat_history, ""
            
            # ユーザーメッセージを追加
            chat_history.append({"role": "user", "content": message})
            
            # チャットの状態を判定
            current_step = get_chat_step(chat_history)
            
            # OpenAI APIでGPTsライクな応答を生成
            system_prompt = get_system_prompt_for_step(current_step)
            
            # APIコール
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # 最近の会話履歴（最大10回分）
            recent_history = chat_history[-20:] if len(chat_history) > 20 else chat_history
            
            messages = [{"role": "system", "content": system_prompt}]
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = client.chat.completions.create(
                model="gpt-4o",  # より高性能なモデルに変更
                messages=messages,
                temperature=0.7,
                max_tokens=1200  # YAML生成用に増加
            )
            
            ai_response = response.choices[0].message.content
            
            # レスポンスにYAML生成指示が含まれる場合の処理
            if "YAML_GENERATE:" in ai_response:
                # YAML生成部分を抽出
                yaml_part = ai_response.split("YAML_GENERATE:")[1].strip()
                
                # YAML形式でない場合は高精度変換
                if not (yaml_part.startswith("#") or "style:" in yaml_part):
                    current_size = app_state.get('current_size', '1024x1024 (正方形)')
                    yaml_part = convert_to_yaml_prompt(yaml_part, api_key, current_size)
                
                # AIレスポンスにYAML完成メッセージを追加（自動反映は削除）
                ai_response_clean = ai_response.split("YAML_GENERATE:")[0].strip()
                ai_response = f"""{ai_response_clean}

✅ **高精度YAML形式プロンプト生成完了！**

```yaml
{yaml_part}
```

📋 **使用方法:**
1. 上記のYAMLコードを**全選択してコピー**してください
2. 「📋 プロンプト編集」エリアに**ペースト**してください  
3. 「✏️ プロンプト直接」タブで「🚀 画像生成」ボタンをクリック

💡 **ヒント:** YAMLの内容を修正したい場合は、上記から続けて修正内容をお伝えください！"""
            
            chat_history.append({"role": "assistant", "content": ai_response})
            
            return chat_history, ""
            
        except Exception as e:
            chat_history.append({"role": "assistant", "content": f"❌ エラーが発生しました: {str(e)}"})
            return chat_history, ""
    
    def get_chat_step(chat_history):
        """現在のチャットステップを判定"""
        if not chat_history:
            return 0
        
        # 簡易的なステップ判定（後で拡張可能）
        user_messages = [msg for msg in chat_history if msg["role"] == "user"]
        return min(len(user_messages), 6)
    
    def generate_with_reference_image_fast(api_key, reference_image, prompt, size, quality, format_opt, transparent, compression, moderation):
        """参照画像を使用した高速画像生成"""
        try:
            valid, error_msg = validate_api_key(api_key)
            if not valid:
                return None, f"❌ {error_msg}", "", ""
            
            if reference_image is None:
                return None, "❌ 参照画像をアップロードしてください", "", ""
            
            if not prompt.strip():
                return None, "❌ 生成したい内容を入力してください", "", ""
            
            os.environ["OPENAI_API_KEY"] = api_key
            app_state['api_key'] = api_key
            
            # PIL画像をbytesに変換
            from io import BytesIO
            image_buffer = BytesIO()
            reference_image.save(image_buffer, format='PNG')
            reference_image_data = image_buffer.getvalue()
            
            # 画像生成
            generator = ImageGenerator(api_key)
            size_key = SIZE_MAP.get(size, "1024x1024")
            
            result = generator.generate_with_reference_image(
                prompt=prompt,
                reference_image_data=reference_image_data,
                size=size_key,
                quality=quality,
                format=format_opt,
                transparent_bg=transparent,
                output_compression=compression if format_opt in ["jpeg", "webp"] else None,
                moderation=moderation
            )
            
            # PIL画像に変換
            image = Image.open(BytesIO(result['image_data']))
            
            # 一時ファイルとして保存
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_opt}")
            image.save(temp_file.name, format=format_opt.upper() if format_opt != 'jpeg' else 'JPEG')
            
            # 履歴に保存
            history_item = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'image_data': result['image_data'],
                'prompt': f"参照画像: {prompt}",
                'purpose': "画像参照生成",
                'style': "参照ベース",
                'temp_file': temp_file.name
            }
            app_state['generation_history'].append(history_item)
            app_state['current_image'] = {'image_data': result['image_data'], 'prompt': prompt}
            app_state['current_prompt'] = prompt
            app_state['original_prompt'] = prompt
            
            # 最新3件に制限
            if len(app_state['generation_history']) > 3:
                app_state['generation_history'] = app_state['generation_history'][-3:]
            
            cost_info = f"""**参照画像生成完了** 🖼️
**時間**: {result.get('generation_time', 'N/A')}秒
**API**: Image Edit API
**モード**: 参照画像ベース生成"""
            
            return image, "✅ 参照画像生成完了！", cost_info, prompt
            
        except Exception as e:
            return None, f"❌ 参照画像生成エラー: {str(e)}", "", ""
    
    def convert_to_yaml_prompt(text_prompt, api_key, current_size="1024x1024 (正方形)"):
        """通常のプロンプトをYAML形式に変換（完全な構造保持）"""
        try:
            # 基本設定のサイズに基づいてベースYAMLを選択
            if "1536x1024" in current_size or "横長" in current_size:
                base_yaml_path = BASE_DIR / "prompts" / "base_landscape.yaml"
            elif "1024x1536" in current_size or "縦長" in current_size:
                base_yaml_path = BASE_DIR / "prompts" / "base_portrait.yaml"
            else:
                base_yaml_path = BASE_DIR / "prompts" / "base_square.yaml"
            
            # ベースYAMLを読み込み
            try:
                with open(base_yaml_path, 'r', encoding='utf-8') as f:
                    base_yaml = f.read()
            except FileNotFoundError:
                print(f"ベースYAMLファイルが見つかりません: {base_yaml_path}")
                # フォールバック：正方形のベースYAMLを使用
                with open(BASE_DIR / "prompts" / "base_square.yaml", 'r', encoding='utf-8') as f:
                    base_yaml = f.read()
            
            # ベースYAMLの行数を取得
            base_yaml_lines = base_yaml.split('\n')
            total_lines = len(base_yaml_lines)
            
            # OpenAI APIでYAMLに変換
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # 超厳密なシステムプロンプト（自動推論型メタプロンプト対応）
            system_prompt = f"""あなたは画像生成プロンプト専門のYAML変換エージェントです。以下のベースYAMLテンプレートを使用して、ユーザーのリクエストを完全にYAML化してください。

## ベースYAMLテンプレート（{total_lines}行）:
```yaml
{base_yaml}
```

## 絶対遵守のルール（違反は絶対禁止）:

### 1. 完全構造保持（必須）
- 上記ベースYAMLの**全{total_lines}行**を完全に出力
- **全てのコメント行（#で始まる行）**を一字一句そのまま保持
- **全てのセクション名**を削除・省略せずに保持
- 全ての技術的な設定値・座標・サイズは元のまま保持

### 2. AUTO_プレースホルダーの自動推論置換（最重要）
ベースYAMLには`{{AUTO_*}}`形式のプレースホルダーが含まれています。以下のルールで自動的に最適な値に置換してください：

#### 用途別自動設定
- YouTube関連 → サイズ:横長、色:鮮やかで高コントラスト、文字:極太で視認性高、背景:目立つグラデーション
- Instagram関連 → サイズ:正方形、色:おしゃれでトレンド感、文字:シンプルで洗練、背景:統一感
- ビジネス関連 → 色:信頼感（青・緑系）、文字:読みやすく品格、背景:プロフェッショナル
- イベント・募集 → 色:明るく親しみやすい、文字:キャッチー、背景:賑やか

#### 自動推論ルール
- `{{AUTO_STYLE}}` → ユーザーの用途から最適なスタイルを推論
- `{{AUTO_COLORS}}` → 用途・ムードから最適な配色を自動選択（具体的な色名やHEXコード）
- `{{AUTO_MOOD}}` → コンテキストから適切な雰囲気を推論
- `{{AUTO_MAIN_TEXT}}` → ユーザー入力を元に魅力的なキャッチコピーを生成
- `{{AUTO_SUB_TEXT}}` → メインを補完する効果的なサブテキストを自動生成
- `{{AUTO_BG_TYPE}}` → solid/gradient/pattern等から最適なものを選択
- `{{AUTO_FONT_*}}` → 用途に応じた最適なフォント設定
- その他の`{{AUTO_*}}` → コンテキストから論理的に推論

#### 未指定項目の補完
- ユーザーが明示的に指定していない項目も、文脈から推論して適切に埋める
- 空欄や曖昧な値は絶対に残さない
- 全てのプレースホルダーを具体的な値に置換

### 3. 条件付きセクションの処理
- 用途に応じて必要なセクションのみ有効化
- 不要なセクションは適切に省略または最小化
- YouTube → character, icons重視
- ビジネス → badge, cta_banner重視
- SNS → visual_identity, social_elements重視

### 4. 品質保証（必須）
- 全ての`{{AUTO_*}}`を具体的で適切な値に置換
- プロ品質の画像生成に必要な全詳細を自動生成
- 色は具体的な色名またはHEXコードで指定
- フォントサイズは具体的な値（px）または相対値（large等）で指定

### 5. 出力形式（必須）
- **ベースYAMLと同じ{total_lines}行**で出力（構造は保持）
- インデント、配列構造、オブジェクト構造を完全保持
- コードブロック不要、YAMLのみを出力

## 変換例:
入力: "YouTubeサムネイル、料理チャンネル"
- style: "YouTubeサムネイル（料理系チャンネル）" 
- theme_color: "#FF6B6B, #4ECDC4, #FFFFFF"（食欲をそそる配色）
- main_texts[0].content: "超簡単！10分で作れる絶品パスタ"
- background.type: "gradient"
- character.expression: "笑顔で料理を楽しんでいる"

**注意: 全ての{{AUTO_*}}プレースホルダーを適切な具体値に置換し、プロ品質の完全なYAMLを出力してください。**"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"以下のリクエストを完全なYAML形式に変換してください（{total_lines}行で出力）: {text_prompt}"}
                ],
                temperature=0.1,  # 一貫性重視で大幅に下げる
                max_tokens=4000   # トークン数を大幅に増加
            )
            
            yaml_result = response.choices[0].message.content
            
            # コードブロックから抽出
            if "```yaml" in yaml_result:
                yaml_result = yaml_result.split("```yaml")[1].split("```")[0]
            elif "```" in yaml_result:
                yaml_result = yaml_result.split("```")[1].split("```")[0]
            
            yaml_result = yaml_result.strip()
            
            # 品質検証
            result_lines = yaml_result.split('\n')
            if len(result_lines) < total_lines * 0.8:  # 80%未満の場合は警告
                print(f"警告: YAML出力が短すぎます（{len(result_lines)}行/{total_lines}行）")
                print("フォールバック処理を実行...")
                
                # フォールバック：より強制的なプロンプト
                fallback_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"**絶対に{total_lines}行で出力してください**\n\n{system_prompt}"},
                        {"role": "user", "content": f"**必ず{total_lines}行で出力**: {text_prompt}"},
                        {"role": "assistant", "content": yaml_result},
                        {"role": "user", "content": f"行数が不足しています。ベースYAMLの**全{total_lines}行**を完全に出力してください。"}
                    ],
                    temperature=0.05,  # 更に一貫性重視
                    max_tokens=4500
                )
                yaml_result = fallback_response.choices[0].message.content
                
                # 再度抽出
                if "```yaml" in yaml_result:
                    yaml_result = yaml_result.split("```yaml")[1].split("```")[0]
                elif "```" in yaml_result:
                    yaml_result = yaml_result.split("```")[1].split("```")[0]
                yaml_result = yaml_result.strip()
            
            # {{AUTO_BADGE}}プレースホルダーの処理
            if "{{AUTO_BADGE}}" in yaml_result:
                # バッジが必要なキーワードをチェック
                badge_keywords = ["新商品", "新発売", "リリース", "キャンペーン", "限定", "NEW", "期間限定", "特価", "セール", "人気", "おすすめ", "注目"]
                needs_badge = any(keyword in text_prompt for keyword in badge_keywords)
                
                if needs_badge:
                    # バッジセクションを生成
                    badge_section = """badge:
  content: "NEW"
  font_style: "bold"
  font_color: "#FFFFFF"
  background_shape: "circle"
  background_color: "#FF4444"
  font_size: "small"
  position: "top-right"
  padding: "10px" """
                else:
                    # バッジなし（空白）
                    badge_section = ""
                
                yaml_result = yaml_result.replace("{{AUTO_BADGE}}", badge_section)
            
            return yaml_result
            
        except Exception as e:
            print(f"YAML変換エラー: {e}")
            # フォールバック: 元のプロンプトを返す
            return text_prompt
    
    def get_system_prompt_for_step(step):
        """ステップごとのシステムプロンプトを取得"""
        base_prompt = """あなたは日本語で対話する画像生成アシスタントです。
ユーザーと段階的に対話し、マーケティング画像のプロンプトを作成します。
常に日本語で返答し、ユーザー操作は必ず絵文字のない番号選択形式に統一してください。

## 対話フロー【必ず必ず必ず選択肢の先頭に番号を表示】
STEP0: 挨拶し、テーマ入力 or デフォルトサンプルかを 1-click で促す  
STEP1: テーマを受けたら → 目的に適したペルソナを 5 つ日本語で生成  
STEP2: 選択番号を受けたら → 画像内コピー（テキスト）案を5 個提示  
STEP3: 選択を受けたら → これまでの決定を基に「YAML_GENERATE:」に続けて詳細要求テキストを出力（後でconvert_to_yaml_prompt関数が自動変換）
STEP4: 「1: このYAMLをコピーして画像生成タブで使用」「2: YAMLの内容を修正する」の二択を出す  
STEP5: 1 が選ばれたら → コピー方法と使用手順を詳しく案内、2 が選ばれたら → ユーザーの修正要求を受けて再ループ

## YAML_GENERATE出力ルール（STEP3専用）
STEP3では「YAML_GENERATE:」の後に以下の形式で詳細なテキスト要求を出力してください：

**形式例**:
YAML_GENERATE: YouTubeサムネイル、エンタメ系チャンネル、「爆笑必至！今日の挑戦はこれだ！」というメインテキスト、ポップでカラフルなデザイン、明るい黄色とオレンジの配色、賑やかで楽しげな背景、元気いっぱいのYouTuberが楽しんでいる様子のキャラクター

## 重要な注意事項
- YAML_GENERATE:の後は**YAML形式ではなく自然な日本語のテキスト**で出力
- 詳細要求テキストには以下を含める：
  * 画像の目的・用途
  * スタイル・雰囲気
  * メインテキスト内容
  * 配色・テーマカラー
  * 背景の雰囲気
  * キャラクター・人物の描写
  * その他の重要な要素
- **絶対にYAML形式（key: value）では出力しない**
- convert_to_yaml_prompt関数が自動的に89-110行の完全なYAMLに変換する

## ふるまい指針
- 質問があいまいなら 1 度だけ聞き返す
- 依頼が英文でも出力は日本語
- 重要テキストが切れないよう配慮

## 絶対に守るルール
1. **選択肢は必ず「1: 選択肢A」「2: 選択肢B」の形式で番号を先頭に付ける**
2. **ユーザーが番号で選択するまで次のステップに進まない**
3. **STEP3でYAML_GENERATE:の後は詳細な日本語テキストのみ（YAML形式禁止）**
4. **曖昧な回答（「お任せ」など）には具体的な選択肢を再提示**
5. **YAML_GENERATE:の後のテキストは非常に詳細で具体的に記述する**

現在のステップ: STEP{step}"""

        return base_prompt
    
    def get_history_images():
        """履歴画像取得（軽量化）"""
        if not app_state['generation_history']:
            return []
        
        recent = app_state['generation_history'][-3:]  # 最新3件のみ（軽量化）
        images = []
        
        for item in reversed(recent):
            try:
                image = Image.open(BytesIO(item['image_data']))
                images.append(image)
            except Exception as e:
                print(f"履歴画像読み込みエラー: {e}")
                continue
        
        return images
    
    # アプリレイアウト構築
    with gr.Blocks(title=APP_CONFIG['title'], css=get_app_css(), theme=gr.themes.Base()) as app:
        
        # ヘッダー
        gr.HTML(f"""
            <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #333; margin-bottom: 1rem;">
                <h1 style="color: #ffffff; font-size: 1.8rem; margin: 0;">{APP_CONFIG['title']}</h1>
            </div>
        """)
        
        # メインレイアウト
        with gr.Row():
            # 左側：設定（35%）
            with gr.Column(scale=35):
                # APIキー設定
                with gr.Accordion("🔑 API設定", open=True):
                    default_api_key = os.getenv("OPENAI_API_KEY", "")
                    api_key = gr.Textbox(
                        label="OpenAI APIキー",
                        placeholder="sk-...",
                        type="password",
                        value=default_api_key,
                        info="環境変数OPENAI_API_KEYが設定されている場合は自動入力されます"
                    )
                
                # 基本設定
                with gr.Accordion("⚙️ 基本設定", open=True):
                    with gr.Row():
                        size = gr.Dropdown(
                            label="サイズ",
                            choices=["1024x1024 (正方形)", "1024x1536 (縦長)", "1536x1024 (横長)"],
                            value="1024x1024 (正方形)"
                        )
                        quality = gr.Dropdown(
                            label="品質",
                            choices=["auto", "low", "medium", "high"],
                            value="auto"
                        )
                    
                    with gr.Row():
                        format_option = gr.Dropdown(
                            label="形式",
                            choices=["png", "jpeg", "webp"],
                            value="png",
                            info="⚠️ 対話型編集はPNG形式でのみ利用可能"
                        )
                        transparent_bg = gr.Checkbox(
                            label="🎭 透明背景",
                            value=False
                        )
                    
                    with gr.Row():
                        compression_slider = gr.Slider(
                            label="圧縮率 (JPEG/WebP)",
                            minimum=0,
                            maximum=100,
                            value=APP_CONFIG['default_compression'],
                            step=5,
                            visible=False
                        )
                    
                    with gr.Row():
                        moderation_dropdown = gr.Dropdown(
                            label="コンテンツフィルター",
                            choices=["auto", "low"],
                            value="auto",
                            info="auto: 標準（推奨）, low: 制限緩和（アート・医療・教育用）"
                        )
                    
                    with gr.Row():
                        image_count_slider = gr.Slider(
                            label="画像数",
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                            info="同時に生成する画像の枚数（多いほど時間・コストがかかります）"
                        )
                    
                    use_ai_mode = gr.Checkbox(
                        label="⚡ AI最適化モード（推奨）",
                        value=True
                    )
                    
                    enable_responses_api = gr.Checkbox(
                        label="💬 対話型有効（Responses API使用）",
                        value=False,
                        info="⚠️ 制限: PNG形式のみ対応 | 参照画像生成では利用不可"
                    )
                    
                
                # タブ
                with gr.Tabs():
                    with gr.Tab("✏️ プロンプト直接") as tab_direct:
                        prompt = gr.Textbox(
                            label="プロンプト",
                            placeholder="美しい山の夕日、フォトリアリスティック、高品質",
                            lines=6
                        )
                        
                        direct_btn = gr.Button("🚀 画像生成", variant="primary", size="lg")
                    
                    with gr.Tab("🖼️ 画像参照生成") as tab_ref:
                        gr.Markdown("""
                        ### 画像参照生成
                        参照画像をアップロードして、その画像のスタイル・構図・雰囲気を参考に新しい画像を生成します。
                        
                        **例**: 犬の写真 + 「猫が帽子をかぶっている」 → 犬の写真のスタイルで猫が帽子をかぶった画像
                        """)
                        
                        
                        reference_image = gr.Image(
                            label="参照画像",
                            type="pil",
                            height=200,
                            image_mode="RGB",
                            sources=["upload", "clipboard"]
                        )
                        
                        reference_prompt = gr.Textbox(
                            label="生成したい内容",
                            placeholder="例: 猫が帽子をかぶっている、美しい風景、モダンなロゴ",
                            lines=3
                        )
                        
                        reference_generate_btn = gr.Button("🖼️ 参照画像で生成", variant="primary", size="lg")
                    
                    with gr.Tab("🤖 AIチャット") as tab_ai:
                        gr.Markdown("""
                        ### 🤖 AI画像生成アシスタント
                        対話形式でプロンプトを生成します。段階的にテーマ→ペルソナ→コピーを決めていきます。
                        """)
                        
                        # チャットエリア
                        ai_chatbot = gr.Chatbot(
                            label="AI対話",
                            height=400,
                            type="messages", 
                            value=[]
                        )
                        
                        with gr.Row():
                            ai_message_input = gr.Textbox(
                                label="メッセージを入力",
                                placeholder="何を作りたいですか？（例：YouTubeサムネイル、Instagram投稿画像）",
                                scale=4,
                                lines=2,
                                info="Shift+Enterで送信"
                            )
                            ai_send_btn = gr.Button("送信\n(Shift+Enter)", variant="primary", scale=1)
                        
                        # 使用方法の案内
                        gr.Markdown("""
                        ### 📋 生成されたYAMLプロンプトの使用方法
                        1. 対話完了後、YAMLコードをコピー
                        2. 「📋 プロンプト編集」エリアにペースト  
                        3. 「✏️ プロンプト直接」タブで画像生成
                        """)
                        
                        with gr.Row():
                            ai_clear_btn = gr.Button("🗑️ チャットクリア", variant="secondary")
                            ai_restart_btn = gr.Button("🔄 最初から", variant="secondary")
            
            # 中央：画像表示（40%）
            with gr.Column(scale=40):
                output_image = gr.Image(label="生成画像", height=400)
                status_display = gr.Markdown("📸 画像生成の準備完了")
                cost_info = gr.Markdown("**生成情報**\\n待機中...")
                
                # プロンプト表示とコピー
                with gr.Accordion("📋 プロンプト編集", open=False):
                    prompt_display = gr.Textbox(
                        label="生成されたプロンプト（編集可能）",
                        placeholder="プロンプトがここに表示されます",
                        lines=8,
                        interactive=True,
                        show_copy_button=True
                    )
                    with gr.Row():
                        regenerate_btn = gr.Button("🔄 編集したプロンプトで再生成", variant="primary")
                        reset_prompt_btn = gr.Button("↩️ 元に戻す", variant="secondary")
                
                # 対話型編集機能
                with gr.Accordion("💬 対話型編集", open=False):
                    gr.Markdown("""
                    ### 生成画像との対話
                    - **継続編集**: 現在の画像をベースに追加の変更を指示
                    - **コンテキスト保持**: 前回の生成内容を覚えて改善
                    - **インタラクティブ**: 細かい調整や追加要求が可能
                    
                    """)
                    
                    interactive_prompt = gr.Textbox(
                        label="追加の指示・変更点",
                        placeholder="例: 空を青空に変更、建物を追加、色調を暖かくする",
                        lines=3
                    )
                    
                    with gr.Row():
                        continue_btn = gr.Button("🔄 対話型編集", variant="primary")
                        reset_context_btn = gr.Button("🗑️ 履歴リセット", variant="secondary")
                    
                    interactive_status = gr.Markdown("💬 画像を生成後、対話型編集が利用可能になります")
                
                # アクションボタン
                with gr.Row():
                    pass  # ダウンロードボタンを削除
            
            # 右側：履歴（25%）
            with gr.Column(scale=25):
                with gr.Accordion("📚 履歴", open=True):
                    history_gallery = gr.Gallery(
                        label="最近の生成",
                        columns=1,
                        rows=3,
                        height="auto"
                    )
                    refresh_btn = gr.Button("🔄 更新", size="sm")
        
        # イベントハンドリング
        
        # プロンプト直接生成
        direct_btn.click(
            generate_from_prompt_fast,
            inputs=[api_key, prompt, size, quality, format_option, transparent_bg, compression_slider, moderation_dropdown, image_count_slider, enable_responses_api],
            outputs=[output_image, status_display, cost_info, prompt_display]
        ).then(
            get_history_images,
            outputs=[history_gallery]
        )
        
        # 参照画像生成
        reference_generate_btn.click(
            generate_with_reference_image_fast,
            inputs=[api_key, reference_image, reference_prompt, size, quality, format_option, transparent_bg, compression_slider, moderation_dropdown],
            outputs=[output_image, status_display, cost_info, prompt_display]
        ).then(
            get_history_images,
            outputs=[history_gallery]
        )
        
        # AIチャット機能
        def ai_chat_simple(api_key, message, chat_history, current_size):
            app_state['current_size'] = current_size
            return ai_chat_response(api_key, message, chat_history)
        
        ai_send_btn.click(
            ai_chat_simple,
            inputs=[api_key, ai_message_input, ai_chatbot, size],
            outputs=[ai_chatbot, ai_message_input]
        )
        
        ai_message_input.submit(
            ai_chat_simple,
            inputs=[api_key, ai_message_input, ai_chatbot, size],
            outputs=[ai_chatbot, ai_message_input]
        )
        
        # AIチャットクリア
        ai_clear_btn.click(
            lambda: [],
            outputs=[ai_chatbot]
        )
        
        # AIチャット再開
        def ai_chat_restart():
            welcome_msg = """🎨 **AI画像生成アシスタント** へようこそ！

段階的な対話で、あなたの理想の画像プロンプトを作成します。

**今日は何を作りますか？**

1: YouTubeサムネイル画像
2: Instagram投稿用画像  
3: ブログアイキャッチ画像
4: ロゴデザイン
5: 自由にテーマを入力

番号を選ぶか、作りたいものを自由に入力してください！"""
            
            # Gradio標準形式で初期メッセージを返す
            return [["", welcome_msg]]
        
        ai_restart_btn.click(
            ai_chat_restart,
            outputs=[ai_chatbot]
        )
        
        # 履歴更新
        refresh_btn.click(get_history_images, outputs=[history_gallery])
        
        # プロンプト再生成
        def regenerate_with_edited_prompt(api_key, edited_prompt, enable_responses_api):
            """編集されたプロンプトで再生成"""
            if not edited_prompt.strip():
                return None, "❌ プロンプトを入力してください", "", ""
            
            # 現在の設定を使用して再生成（サイズ、品質等は最後の設定を使用）
            return generate_from_prompt_fast(api_key, edited_prompt, "1024x1024 (正方形)", "auto", "png", False, APP_CONFIG['default_compression'], "auto", 1, enable_responses_api)
        
        regenerate_btn.click(
            regenerate_with_edited_prompt,
            inputs=[api_key, prompt_display, enable_responses_api],
            outputs=[output_image, status_display, cost_info, prompt_display]
        ).then(
            get_history_images,
            outputs=[history_gallery]
        )
        
        # プロンプトを元に戻す
        def reset_to_original_prompt():
            """元のプロンプトに戻す"""
            if app_state.get('original_prompt'):
                return app_state['original_prompt']
            return "元のプロンプトがありません"
        
        reset_prompt_btn.click(
            reset_to_original_prompt,
            outputs=[prompt_display]
        )
        
        
        # 対話型編集機能
        def interactive_edit(api_key, user_instruction, size, quality, format_opt, transparent, compression, moderation):
            """対話型編集実行"""
            try:
                # デバッグ情報
                current_response_id = app_state.get('last_response_id', 'None')
                debug_info = f"デバッグ: last_response_id = {current_response_id}"
                print(f"[DEBUG] 対話型編集開始: {debug_info}")
                
                if not current_response_id or current_response_id == 'None':
                    error_msg = f"""💬 **対話型編集を使用するには：**
                    
1. いずれかのタブで「💬 対話型有効」をチェックしてください
2. その状態で画像を生成してください
3. 生成後に対話型編集が利用可能になります

{debug_info}"""
                    return None, error_msg, ""
                
                if not user_instruction.strip():
                    return None, "💬 変更したい内容を入力してください", ""
                
                valid, error_msg = validate_api_key(api_key)
                if not valid:
                    return None, f"❌ {error_msg}", ""
                
                # Responses APIで継続生成を実行
                responses_api = ResponsesAPI(api_key)
                
                # 継続生成を実行
                print(f"[DEBUG] 継続生成実行: previous_response_id={current_response_id[:8]}...")
                
                result = responses_api.continue_generation(
                    previous_response_id=current_response_id,
                    prompt=user_instruction,
                    size=SIZE_MAP.get(size, size),
                    quality=quality,
                    format=format_opt,
                    background="transparent" if transparent else "auto",
                    output_compression=compression if format_opt in ["jpeg", "webp"] else None,
                    moderation=moderation
                )
                
                # 画像を処理
                image = Image.open(BytesIO(result['image_data']))
                
                # 履歴に追加
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_opt}")
                image.save(temp_file.name, format=format_opt.upper() if format_opt != 'jpeg' else 'JPEG')
                
                history_item = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'image_data': result['image_data'],
                    'prompt': f"対話編集: {user_instruction}",
                    'purpose': "対話型編集",
                    'style': "継続生成",
                    'temp_file': temp_file.name
                }
                app_state['generation_history'].append(history_item)
                
                # 最新3件に制限
                if len(app_state['generation_history']) > 3:
                    app_state['generation_history'] = app_state['generation_history'][-3:]
                
                # 状態更新（重要：新しいresponse_idに更新）
                app_state['current_image'] = {'image_data': result['image_data'], 'prompt': user_instruction}
                new_response_id = result['response_id']
                app_state['last_response_id'] = new_response_id
                print(f"[DEBUG] 対話型編集完了: 新response_id={new_response_id[:8]}...")
                
                # コスト情報
                cost_info = f"""**対話型編集完了**
⏱️ 生成時間: {result['generation_time']}秒
🔄 前回ID: {result['previous_response_id'][:8]}...
🆕 新規ID: {result['response_id'][:8]}..."""
                
                return image, "💬 対話型編集が完了しました！", cost_info
                
            except Exception as e:
                error_detail = f"""❌ **対話型編集エラー**

**エラー詳細**: {str(e)}
**last_response_id**: {app_state.get('last_response_id', 'None')}

**解決方法**: 「💬 対話型有効」をチェックして画像を生成してから再試行してください"""
                return None, error_detail, ""
        
        # 対話型編集イベント
        continue_btn.click(
            interactive_edit,
            inputs=[api_key, interactive_prompt, size, quality, format_option, transparent_bg, compression_slider, moderation_dropdown],
            outputs=[output_image, interactive_status, cost_info]
        ).then(
            get_history_images,
            outputs=[history_gallery]
        )
        
        # 対話履歴リセット
        def reset_interactive_context():
            """対話コンテキストをリセット"""
            app_state['last_response_id'] = None
            app_state['generation_context'] = []
            return "💬 対話履歴をリセットしました。新しい画像を生成してください。"
        
        reset_context_btn.click(
            reset_interactive_context,
            outputs=[interactive_status]
        )
        
        # フォーマット変更時の制御
        def toggle_compression_slider(format_value):
            """JPEG/WebP選択時のみ圧縮スライダーを表示"""
            return gr.update(visible=format_value in ["jpeg", "webp"])
        
        def enforce_png_for_responses(format_value):
            """PNG以外を選んだら対話型有効を自動OFF"""
            if format_value != "png":
                return gr.update(
                    value=False, 
                    interactive=False,
                    label="💬 対話型有効（PNG形式のみ対応）",
                    info="⚠️ JPEG/WebP選択時は対話型編集が無効になります"
                )
            return gr.update(
                interactive=True,
                label="💬 対話型有効（Responses API使用）",
                info="⚠️ 制限: PNG形式のみ対応 | 参照画像生成では利用不可"
            )
        
        format_option.change(
            toggle_compression_slider,
            inputs=[format_option],
            outputs=[compression_slider]
        )
        
        format_option.change(
            enforce_png_for_responses,
            inputs=[format_option],
            outputs=[enable_responses_api]
        )
        
        # タブ選択による対話型有効の制御
        def disable_responses_for_ref_tab():
            """参照画像タブ選択時は対話型有効を自動OFF"""
            return gr.update(
                value=False,
                interactive=False,
                label="💬 対話型有効（参照画像タブでは利用不可）",
                info="⚠️ 参照画像生成では対話型編集は利用できません"
            )
        
        def enable_responses_for_other_tabs():
            """他のタブ選択時は対話型有効を復活"""
            return gr.update(
                interactive=True,
                label="💬 対話型有効（Responses API使用）",
                info="⚠️ 制限: PNG形式のみ対応 | 参照画像生成では利用不可"
            )
        
        # タブ選択イベント
        tab_ref.select(
            disable_responses_for_ref_tab,
            outputs=[enable_responses_api]
        )
        
        tab_direct.select(
            enable_responses_for_other_tabs,
            outputs=[enable_responses_api]
        )
        
        tab_ai.select(
            enable_responses_for_other_tabs,
            outputs=[enable_responses_api]
        )
    
    return app

# HuggingFace Spaces用のメイン実行部
if __name__ == "__main__":
    app = create_optimized_app()
    # Hugging Face Spaces用の設定
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False  # Hugging Face Spacesでは不要
    )