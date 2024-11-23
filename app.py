# 导入必要的库
import gradio as gr  # Gradio用于创建Web界面
from together import Together  # Together AI的Python客户端
import tempfile  # 用于创建临时文件
from PIL import Image  # 用于图像处理
import os  # 用于文件和目录操作
import requests  # 用于发送HTTP请求
from io import BytesIO  # 用于处理二进制数据
import logging  # 用于日志记录
import base64  # 用于Base64编码/解码
import zipfile  # 用于创建ZIP文件
import time  # 用于时间相关操作
import random  # 用于生成随机数

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Together AI客户端
client = Together()

# 定义可用的图生图模型
IMG2IMG_MODELS = {
    "FLUX.1-depth": "black-forest-labs/FLUX.1-depth",  # 深度图引导的图像生成
    "FLUX.1-canny": "black-forest-labs/FLUX.1-canny",  # 边缘检测引导的图像生成
    "FLUX.1-redux": "black-forest-labs/FLUX.1-redux"   # 通用图像生成
}

# 定义可用的文生图模型
TEXT2IMG_MODELS = {
    "FLUX.1-dev": "black-forest-labs/FLUX.1-dev",        # 开发版模型
    "FLUX.1-schnell": "black-forest-labs/FLUX.1-schnell",  # 快速生成模型
    "FLUX.1.1-pro": "black-forest-labs/FLUX.1.1-pro"      # 专业版模型
}

# 各模型的特定参数设置
MODEL_PARAMS = {
    # 图生图模型参数
    "black-forest-labs/FLUX.1-depth": {"min_steps": 4, "max_steps": 50, "default_steps": 28},
    "black-forest-labs/FLUX.1-canny": {"min_steps": 4, "max_steps": 50, "default_steps": 28},
    "black-forest-labs/FLUX.1-redux": {"min_steps": 4, "max_steps": 50, "default_steps": 28},
    
    # 文生图模型参数
    "black-forest-labs/FLUX.1-dev": {"min_steps": 25, "max_steps": 50, "default_steps": 28},
    "black-forest-labs/FLUX.1-schnell": {"min_steps": 4, "max_steps": 50, "default_steps": 12},
    "black-forest-labs/FLUX.1.1-pro": {"min_steps": 1, "max_steps": 10, "default_steps": 4}
}

def encode_image_to_base64(image):
    """
    将PIL图像对象转换为base64编码的字符串
    
    Args:
        image: PIL Image对象
        
    Returns:
        str: base64编码的图像数据URL
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def save_images_as_zip(images):
    """
    将多张图片保存为ZIP文件
    
    Args:
        images: PIL Image对象列表
        
    Returns:
        str: ZIP文件的路径，如果出错则返回None
    """
    # 创建输出目录
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用时间戳创建唯一的文件名
    timestamp = int(time.time())
    zip_filename = f"generated_images_{timestamp}.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i, img in enumerate(images):
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                zipf.writestr(f"image_{i+1}.png", img_byte_arr)
        
        return zip_path
    except Exception as e:
        logger.error(f"Error creating ZIP file: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None

def process_image2image(image, model_name, prompt, width, height, steps, num_images, seed):
    """
    处理图生图请求，将输入图片转换为新的图片
    
    Args:
        image: 输入图片（PIL Image对象）
        model_name: 选择的模型名称
        prompt: 文本提示词
        width: 输出图片宽度
        height: 输出图片高度
        steps: 生成步数
        num_images: 生成图片数量
        seed: 随机种子
        
    Returns:
        tuple: (生成的图片列表, ZIP文件路径, 状态消息)
    """
    # 检查必要的输入
    if image is None:
        return [None] * num_images, None, "请上传图片"
    if model_name != "FLUX.1-redux" and not prompt:
        return [None] * num_images, None, "请输入提示词（仅FLUX.1-redux模型可选）"
    
    try:
        # 将图片转换为base64格式
        image_base64 = encode_image_to_base64(image)
        logger.info("图片已转换为base64格式")
        
        model_id = IMG2IMG_MODELS[model_name]
        
        # 准备API调用参数
        api_params = {
            "model": model_id,
            "width": width,
            "height": height,
            "steps": min(steps, MODEL_PARAMS[model_id]["max_steps"]),
            "n": num_images,
            "image_url": image_base64,
            "prompt": prompt if (prompt or model_name != "FLUX.1-redux") else "[]",
            "seed": seed if seed > 0 else random.randint(1, 10000)
        }
        
        logger.info(f"使用模型 {model_name} 生成 {num_images} 张图片")
        imageCompletion = client.images.generate(**api_params)
        
        # 处理生成的图片
        generated_images = []
        for i, data in enumerate(imageCompletion.data):
            result_url = data.url
            logger.info(f"下载生成的图片 {i+1}/{num_images}")
            
            response = requests.get(result_url)
            if response.status_code == 200:
                generated_image = Image.open(BytesIO(response.content))
                generated_images.append(generated_image)
            else:
                logger.error(f"下载图片 {i+1} 失败。状态码: {response.status_code}")
        
        if not generated_images:
            return [None] * num_images, None, "生成图片失败"
        
        # 保存为ZIP文件
        zip_path = save_images_as_zip(generated_images)
        
        # 确保返回列表长度符合要求
        while len(generated_images) < num_images:
            generated_images.append(None)
        
        return (
            generated_images,
            zip_path,
            f"成功使用 {model_name} 生成了 {len(generated_images)} 张图片！"
        )
            
    except Exception as e:
        error_msg = f"错误: {str(e)}"
        logger.error(error_msg)
        return [None] * num_images, None, error_msg

def process_text2image(prompt, model_name, width, height, steps, num_images, seed):
    """
    处理文生图请求，根据文本提示词生成图片
    
    Args:
        prompt: 文本提示词
        model_name: 选择的模型名称
        width: 输出图片宽度
        height: 输出图片高度
        steps: 生成步数
        num_images: 生成图片数量
        seed: 随机种子
        
    Returns:
        tuple: (生成的图片列表, ZIP文件路径, 状态消息)
    """
    # 检查必要的输入
    if not prompt:
        return [None] * num_images, None, "请输入提示词"
    
    try:
        model_id = TEXT2IMG_MODELS[model_name]
        model_params = MODEL_PARAMS[model_id]
        
        # 根据模型调整步数
        if steps == 28:  # 如果是默认值
            steps = model_params["default_steps"]
        else:
            # 确保步数在模型的限制范围内
            steps = max(model_params["min_steps"], min(steps, model_params["max_steps"]))
        
        # 准备API调用参数
        api_params = {
            "model": model_id,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "n": num_images,
            "seed": seed if seed > 0 else random.randint(1, 10000)
        }
        
        logger.info(f"使用模型 {model_name} 生成 {num_images} 张图片（步数: {steps}）")
        try:
            imageCompletion = client.images.generate(**api_params)
        except Exception as api_error:
            error_msg = str(api_error)
            if "credit card" in error_msg.lower():
                raise Exception(
                    "此模型需要信用卡验证。"
                    "请在 https://api.together.xyz/settings/billing 添加信用卡"
                )
            raise  # 重新抛出其他API错误
        
        # 处理生成的图片
        generated_images = []
        for i, data in enumerate(imageCompletion.data):
            result_url = data.url
            logger.info(f"下载生成的图片 {i+1}/{num_images}")
            
            response = requests.get(result_url)
            if response.status_code == 200:
                generated_image = Image.open(BytesIO(response.content))
                generated_images.append(generated_image)
            else:
                logger.error(f"下载图片 {i+1} 失败。状态码: {response.status_code}")
        
        if not generated_images:
            return [None] * num_images, None, "生成图片失败"
        
        # 保存为ZIP文件
        zip_path = save_images_as_zip(generated_images)
        
        # 确保返回列表长度符合要求
        while len(generated_images) < num_images:
            generated_images.append(None)
        
        return (
            generated_images,
            zip_path,
            f"成功使用 {model_name} 生成了 {len(generated_images)} 张图片！"
        )
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"错误: {error_msg}")
        return [None] * num_images, None, f"错误: {error_msg}"

def get_image_prompts(image):
    """
    分析图片并生成相似的AI艺术提示词
    
    Args:
        image: PIL Image对象
        
    Returns:
        str: 生成的提示词或错误消息
    """
    if image is None:
        return "请先上传图片"
    
    try:
        # 将图片转换为base64格式
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 准备对话消息
        client = Together()
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的AI艺术专家。你的任务是分析图片并提供能重新创建类似图片的AI艺术提示词。请始终提供3个提示词，分别关注不同方面如风格、构图和细节。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "分析这张图片并写出3个不同的AI艺术提示词，每个提示词关注不同的方面。格式：1. [提示词1] 2. [提示词2] 3. [提示词3]"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ]

        # 调用API生成提示词
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            messages=messages,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            max_tokens=512
        )
        
        result = response.choices[0].message.content.strip()
        
        # 检查响应是否包含拒绝或错误信息
        error_phrases = [
            "i'm not", "cannot", "sorry", "unable", 
            "don't", "do not", "won't", "will not",
            "inappropriate", "not appropriate"
        ]
        
        if any(phrase in result.lower() for phrase in error_phrases):
            return "无法分析此图片，请尝试上传其他图片。提示：上传AI生成的图片效果更好。"
            
        # 如果响应太短，可能不是有效的提示词
        if len(result.split()) < 10:
            return "生成的提示词无效，请重试或上传其他图片"
            
        return result
    except Exception as e:
        return f"生成提示词时出错: {str(e)}"

# 定义界面样式
css = """
.gradio-container {
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
}
.gallery {
    margin: 20px 0;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.gallery img {
    cursor: pointer;
}
.app-title {
    text-align: center;
    margin-bottom: 2em;
    color: #2a4858;
}
.app-title h1 {
    font-size: 2.5em;
    margin-bottom: 0.2em;
}
.app-title p {
    font-size: 1.1em;
    color: #666;
}
"""

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    # 添加页面标题
    with gr.Column():
        gr.HTML(
            """
            <div class="app-title">
                <h1>🌊 Together Flux Studio</h1>
                <p>Powered by Together AI</p>
            </div>
            """
        )
    
    # 用于在标签页之间传递选中图片的状态变量
    selected_image = gr.State(None)
    
    with gr.Tabs():
        # 文生图标签页
        with gr.Tab("Text to Image"):
            with gr.Row():
                # 左侧控制面板
                with gr.Column():
                    # 文本输入框
                    text2img_prompt = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=2
                    )
                    # 模型选择下拉框
                    text2img_model = gr.Dropdown(
                        choices=list(TEXT2IMG_MODELS.keys()),
                        value="FLUX.1-dev",
                        label="Select Model (Credit Card Required)"
                    )
                    
                    # 图片尺寸控制
                    with gr.Row():
                        text2img_width = gr.Slider(
                            minimum=0,
                            maximum=1792,
                            value=1024,
                            step=64,
                            label="Width"
                        )
                        text2img_height = gr.Slider(
                            minimum=0,
                            maximum=1792,
                            value=1024,
                            step=64,
                            label="Height"
                        )
                    
                    # 生成参数控制
                    with gr.Row():
                        text2img_steps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=28,
                            step=1,
                            label="Steps (will adjust based on model)"
                        )
                        text2img_num = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1,
                            label="Number of Images"
                        )
                    
                    # 随机种子控制
                    text2img_seed = gr.Slider(
                        minimum=0,
                        maximum=10000,
                        value=0,
                        step=1,
                        label="Seed (0 for random)"
                    )
                    
                    # 生成按钮
                    text2img_btn = gr.Button("Generate Images", variant="primary")
                
                # 右侧图片展示区
                with gr.Column():
                    # 图片画廊
                    text2img_gallery = gr.Gallery(
                        label="生成的图片（选择一张想要处理的图片）",
                        show_label=True,
                        elem_id="text2img_gallery",
                        columns=2,
                        rows=2,
                        height="auto",
                        preview=True,
                        allow_preview=True,
                        selected_index=None
                    )
                    # 操作按钮和下载区
                    with gr.Row():
                        send_to_img2img = gr.Button("发送到图片生成页面", variant="secondary", scale=2)
                        text2img_download = gr.File(
                            label="下载所有图片(ZIP)",
                            file_count="single",
                            scale=1
                        )
                    # 状态显示
                    text2img_status = gr.Textbox(label="状态", interactive=False)
        
        # 图生图标签页
        with gr.Tab("Image to Image"):
            with gr.Row():
                # 左侧控制面板
                with gr.Column():
                    # 图片输入
                    input_image = gr.Image(
                        type="pil",
                        label="上传图片或从文字生成的图片中选择（先在文字生成页面生成图片，点击想要的图片后会自动传到这里）"
                    )
                    # 模型选择
                    img2img_model = gr.Dropdown(
                        choices=list(IMG2IMG_MODELS.keys()),
                        value="FLUX.1-depth",
                        label="Select Model"
                    )
                    # 提示词输入
                    img2img_prompt = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Describe how you want to transform the image... (optional for FLUX.1-redux)",
                        lines=2
                    )
                    
                    # 图片尺寸控制
                    with gr.Row():
                        img2img_width = gr.Slider(
                            minimum=0,
                            maximum=1792,
                            value=1024,
                            step=64,
                            label="Width"
                        )
                        img2img_height = gr.Slider(
                            minimum=0,
                            maximum=1792,
                            value=1024,
                            step=64,
                            label="Height"
                        )
                    
                    # 生成参数控制
                    with gr.Row():
                        img2img_steps = gr.Slider(
                            minimum=4,
                            maximum=50,
                            value=28,
                            step=1,
                            label="Steps"
                        )
                        img2img_num = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=1,
                            step=1,
                            label="Number of Images"
                        )
                    
                    # 随机种子控制
                    img2img_seed = gr.Slider(
                        minimum=0,
                        maximum=10000,
                        value=0,
                        step=1,
                        label="Seed (0 for random)"
                    )
                    
                    # 生成按钮
                    img2img_btn = gr.Button("Generate Images", variant="primary")
                
                # 右侧图片展示区
                with gr.Column():
                    # 图片画廊
                    img2img_gallery = gr.Gallery(
                        label="Generated Images",
                        show_label=True,
                        elem_id="img2img_gallery",
                        columns=[2],
                        rows=[2],
                        height="auto"
                    )
                    # 下载区
                    img2img_download = gr.File(
                        label="Download All Images (ZIP)",
                        file_count="single"
                    )
                    # 状态显示
                    img2img_status = gr.Textbox(label="Status", interactive=False)
        
        # 提示词分析标签页
        with gr.Tab("提示词识别"):
            with gr.Row():
                # 左侧图片上传区
                with gr.Column(scale=1):
                    prompt_image = gr.Image(
                        type="pil",
                        label="上传图片",
                        sources=["upload"],
                        height=400
                    )
                    analyze_btn = gr.Button("分析图片", variant="primary")
                
                # 右侧结果显示区
                with gr.Column(scale=1):
                    prompt_output = gr.Textbox(
                        label="可能的提示词",
                        placeholder="上传图片并点击分析按钮获取提示词...",
                        lines=6,
                        interactive=False
                    )
                    prompt_status = gr.Textbox(
                        label="状态",
                        visible=False,
                        interactive=False
                    )
    
    # 事件处理函数
    def on_select_text2img(gallery):
        """
        处理从图库中选择图片的事件
        
        Args:
            gallery: 图库组件的当前状态
            
        Returns:
            选中的图片或None
        """
        if gallery is not None and isinstance(gallery, list) and len(gallery) > 0:
            selected = gallery[0]  # 获取选中的第一张图片
            if isinstance(selected, tuple):
                # 如果是元组，返回第一个元素（图片数据）
                return selected[0]
            return selected
        return None

    # 注册事件处理器
    # 从文生图页面发送图片到图生图页面
    send_to_img2img.click(
        fn=on_select_text2img,
        inputs=[text2img_gallery],
        outputs=[input_image],
    )
    
    # 图生图生成按钮点击事件
    img2img_btn.click(
        fn=process_image2image,
        inputs=[
            input_image,
            img2img_model,
            img2img_prompt,
            img2img_width,
            img2img_height,
            img2img_steps,
            img2img_num,
            img2img_seed
        ],
        outputs=[img2img_gallery, img2img_download, img2img_status]
    )
    
    # 文生图生成按钮点击事件
    text2img_btn.click(
        fn=process_text2image,
        inputs=[
            text2img_prompt,
            text2img_model,
            text2img_width,
            text2img_height,
            text2img_steps,
            text2img_num,
            text2img_seed
        ],
        outputs=[text2img_gallery, text2img_download, text2img_status]
    )
    
    def on_analyze_click(image, status):
        """
        处理图片分析按钮点击事件
        
        Args:
            image: 输入图片
            status: 当前状态
            
        Returns:
            tuple: (生成的提示词, 状态消息)
        """
        if image is None:
            return None, "请先上传图片"
        try:
            prompts = get_image_prompts(image)
            return prompts, None
        except Exception as e:
            return None, f"分析出错: {str(e)}"

    # 图片分析按钮点击事件
    analyze_btn.click(
        fn=on_analyze_click,
        inputs=[prompt_image, prompt_status],
        outputs=[prompt_output, prompt_status],
        api_name=False
    )

# 程序入口点
if __name__ == "__main__":
    # 启动Gradio应用
    demo.launch(
        server_name="127.0.0.1",  # 本地服务器
        show_error=True,          # 显示错误信息
        share=False              # 不创建公共链接
    )