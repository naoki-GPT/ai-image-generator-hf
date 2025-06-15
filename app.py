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
except:
    pass

# Gradioのバグ修正用猿パッチ（ChatGPT推奨）
try:
    import gradio_client.utils as _gcu
    _orig = _gcu._json_schema_to_python_type  # keep ref

    def _patched(schema, defs=None):
        if isinstance(schema, bool):          # ← 追加
            schema = {}                       # bool → 空dict に変換
        return _orig(schema, defs or {})

    _gcu._json_schema_to_python_type = _patched
    print("猿パッチを適用しました")
except Exception as e:
    print(f"猿パッチの適用に失敗: {e}")

# HuggingFace Spaces用パス設定
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

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

def convert_to_yaml_prompt(text_prompt, api_key, current_size="1024x1024 (正方形)"):
    """通常のプロンプトをYAML形式に変換（完全な構造保持）"""
    try:
        # HuggingFace Spaces用パス設定
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
            fallback_path = BASE_DIR / "prompts" / "base_square.yaml"
            with open(fallback_path, 'r', encoding='utf-8') as f:
                base_yaml = f.read()
        
        # 行数カウント
        total_lines = len(base_yaml.split('\n'))
        
        # OpenAI APIでYAML変換
        from openai import OpenAI
        import os
        
        # Hugging Face Spacesの環境変数によるプロキシ設定を無効化
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(key, None)
        
        # openai==1.51.0のバグ対策（猿パッチ）
        try:
            import functools
            OpenAI.__init__ = functools.partialmethod(OpenAI.__init__, proxies=None)
        except:
            pass
        
        client = OpenAI(api_key=api_key)
        
        # システムプロンプト（完全保持型）
        system_prompt = f"""あなたは高度なYAMLプロンプト生成スペシャリストです。以下のベースYAML構造を使用して、ユーザーの要求を完全なYAMLプロンプトに変換してください。

## ベースYAML構造
```yaml
{base_yaml}
```

## 重要な変換ルール

### 1. 構造完全保持（必須）
- ベースYAMLの**全{total_lines}行の構造を完全に保持**
- インデント、配列、オブジェクト構造を一切変更しない
- キー名、階層構造を完全に維持

### 2. プレースホルダー置換
- `{{AUTO_*}}`を具体的な値に置換
- ユーザー要求に最適化された内容を生成
- 全てのAUTO_プレースホルダーを必ず置換

### 3. 内容の最適化
- ユーザーの要求（用途、スタイル、テーマ）を反映
- 高品質なプロ仕様の記述
- 詳細で具体的な要素指定

### 4. 品質保証
- offset、scale、rotation値は論理的に設定
- 色指定は具体的なカラーコード使用
- effect、lighting等は現実的な値を設定

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
            temperature=0.1,
            max_tokens=4000
        )
        
        yaml_result = response.choices[0].message.content
        
        # コードブロックから抽出
        if "```yaml" in yaml_result:
            yaml_result = yaml_result.split("```yaml")[1].split("```")[0]
        elif "```" in yaml_result:
            yaml_result = yaml_result.split("```")[1].split("```")[0]
        
        yaml_result = yaml_result.strip()
        
        return yaml_result
        
    except Exception as e:
        print(f"YAML変換エラー: {e}")
        return text_prompt

def create_optimized_app():
    # アプリケーション状態の初期化
    app_state = {
        'generation_history': [],
        'current_image': None,
        'current_prompt': '',
        'original_prompt': '',
        'interactive_context': [],
        'ai_chat_step': 0,
        'current_size': '1024x1024 (正方形)'
    }
    
    def generate_from_prompt_fast(api_key, prompt, size, quality, format_option, transparent_bg, compression, moderation, image_count, enable_responses_api):
        """プロンプト直接生成（高速化版）"""
        try:
            if not api_key or not api_key.strip():
                return None, "❌ APIキーを入力してください", "", ""
            
            if not prompt or not prompt.strip():
                return None, "❌ プロンプトを入力してください", "", ""
            
            # サイズ情報をapp_stateに保存
            app_state['current_size'] = size
            
            # プロンプトがYAML形式かチェック
            if prompt.strip().startswith('style:') or 'main_texts:' in prompt:
                final_prompt = prompt
            else:
                final_prompt = convert_to_yaml_prompt(prompt, api_key, size)
            
            # Responses API使用判定
            if enable_responses_api:
                responses_api = ResponsesAPI(api_key)
                result = responses_api.generate_with_responses(
                    prompt=final_prompt,
                    size=SIZE_MAP.get(size, "1024x1024"),
                    quality=quality,
                    format=format_option,
                    background="transparent" if transparent_bg else "auto",
                    output_compression=compression if format_option in ["jpeg", "webp"] else None,
                    moderation=moderation
                )
            else:
                generator = ImageGenerator(api_key)
                result = generator.generate_image(
                    prompt=final_prompt,
                    size=SIZE_MAP.get(size, "1024x1024"),
                    quality=quality,
                    format=format_option,
                    transparent_bg=transparent_bg,
                    output_compression=compression if format_option in ["jpeg", "webp"] else None,
                    moderation=moderation,
                    n=image_count
                )
            
            if result and 'image_data' in result:
                image = Image.open(BytesIO(result['image_data']))
                
                cost_info = f"""**生成完了** 📝
**時間**: {result.get('generation_time', 'N/A')}秒
**API**: {'Responses API' if enable_responses_api else 'Image API'}
**YAML変換**: {'適用済み' if final_prompt != prompt else 'なし'}"""
                
                return image, "✅ 生成完了！", cost_info, final_prompt
            else:
                return None, "❌ 画像生成に失敗しました", "", ""
                
        except Exception as e:
            return None, f"❌ 生成エラー: {str(e)}", "", ""
    
    def ai_chat_response(api_key, message, chat_history):
        """AIチャット機能"""
        try:
            if not message.strip():
                return chat_history, ""
            
            if not api_key or not api_key.strip():
                error_msg = "❌ APIキーを入力してください"
                chat_history.append([message, error_msg])
                return chat_history, ""
            
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Gradio形式からOpenAI messages形式に変換
            messages = []
            for pair in chat_history:
                if len(pair) >= 2:
                    messages.append({"role": "user", "content": pair[0]})
                    messages.append({"role": "assistant", "content": pair[1]})
            
            # 現在のメッセージを追加
            messages.append({"role": "user", "content": message})
            
            current_step = min(len([msg for msg in messages if msg["role"] == "assistant"]), 5)
            system_prompt = get_system_prompt_for_step(current_step)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content
            
            if "YAML_GENERATE:" in ai_response:
                yaml_part = ai_response.split("YAML_GENERATE:")[1].strip()
                yaml_result = convert_to_yaml_prompt(yaml_part, api_key, app_state.get('current_size', '1024x1024 (正方形)'))
                ai_response = ai_response.replace(f"YAML_GENERATE:{yaml_part}", f"YAML_GENERATE:\n\n```yaml\n{yaml_result}\n```")
            
            # Gradio形式で返す
            chat_history.append([message, ai_response])
            return chat_history, ""
            
        except Exception as e:
            error_msg = f"❌ AI応答エラー: {str(e)}"
            chat_history.append([message, error_msg])
            return chat_history, ""
    
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
                            value="png"
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
                            step=1
                        )
                    
                    enable_responses_api = gr.Checkbox(
                        label="💬 対話型有効（Responses API使用）",
                        value=False
                    )
                
                # タブ
                with gr.Tabs():
                    with gr.Tab("✏️ プロンプト直接"):
                        prompt = gr.Textbox(
                            label="プロンプト",
                            placeholder="美しい山の夕日、フォトリアリスティック、高品質",
                            lines=6
                        )
                        
                        direct_btn = gr.Button("🚀 画像生成", variant="primary", size="lg")
                    
                    with gr.Tab("🤖 AIチャット"):
                        gr.Markdown("""
                        ### 🤖 AI画像生成アシスタント
                        対話形式でプロンプトを生成します。段階的にテーマ→ペルソナ→コピーを決めていきます。
                        """)
                        
                        ai_chatbot = gr.Chatbot(
                            label="AI対話",
                            height=400
                        )
                        
                        with gr.Row():
                            ai_message_input = gr.Textbox(
                                label="メッセージを入力",
                                placeholder="何を作りたいですか？（例：YouTubeサムネイル、Instagram投稿画像）",
                                scale=4,
                                lines=2
                            )
                            ai_send_btn = gr.Button("送信", variant="primary", scale=1)
                        
                        with gr.Row():
                            ai_clear_btn = gr.Button("🗑️ チャットクリア", variant="secondary")
                            ai_restart_btn = gr.Button("🔄 最初から", variant="secondary")
            
            # 中央：画像表示（40%）
            with gr.Column(scale=40):
                output_image = gr.Image(label="生成画像", height=400)
                status_display = gr.Markdown("📸 画像生成の準備完了")
                cost_info = gr.Markdown("**生成情報**\\n待機中...")
                
                # プロンプト表示
                with gr.Accordion("📋 プロンプト編集", open=False):
                    prompt_display = gr.Textbox(
                        label="生成されたプロンプト（編集可能）",
                        placeholder="プロンプトがここに表示されます",
                        lines=8,
                        interactive=True,
                        show_copy_button=True
                    )
                    regenerate_btn = gr.Button("🔄 編集したプロンプトで再生成", variant="primary")
            
            # 右側：履歴（25%）
            with gr.Column(scale=25):
                with gr.Accordion("📚 履歴", open=True):
                    gr.Markdown("履歴機能は簡略化版です")
        
        # イベントハンドリング
        
        # プロンプト直接生成
        direct_btn.click(
            generate_from_prompt_fast,
            inputs=[api_key, prompt, size, quality, format_option, transparent_bg, compression_slider, moderation_dropdown, image_count_slider, enable_responses_api],
            outputs=[output_image, status_display, cost_info, prompt_display]
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
        
        # プロンプト再生成
        def regenerate_with_edited_prompt(api_key, edited_prompt, enable_responses_api):
            if not edited_prompt.strip():
                return None, "❌ プロンプトを入力してください", "", ""
            
            return generate_from_prompt_fast(api_key, edited_prompt, "1024x1024 (正方形)", "auto", "png", False, APP_CONFIG['default_compression'], "auto", 1, enable_responses_api)
        
        regenerate_btn.click(
            regenerate_with_edited_prompt,
            inputs=[api_key, prompt_display, enable_responses_api],
            outputs=[output_image, status_display, cost_info, prompt_display]
        )
        
        # 形式変更時のUI更新
        def update_compression_visibility(format_opt):
            return gr.update(visible=(format_opt in ["jpeg", "webp"]))
        
        format_option.change(
            update_compression_visibility,
            inputs=[format_option],
            outputs=[compression_slider]
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