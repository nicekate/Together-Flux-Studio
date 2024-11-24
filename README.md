# Together Flux Studio

🌊 一个基于Together AI的强大图像生成工具，支持文生图、图生图和提示词分析功能。

## 预览

![预览图1](https://r2.kateviews.com/20241124_665.png)
![预览图2](https://r2.kateviews.com/20241124_26.png)
![预览图3](https://r2.kateviews.com/20241124_639.png)

## 视频教程

📺 观看详细的使用教程：[Together Flux Studio 使用指南](https://www.bilibili.com/video/BV1pCBHYqEaT/)

## 功能特点

### 🎨 文生图 (Text to Image)
- 支持多个专业级AI模型：
  - FLUX.1-dev：开发版模型
  - FLUX.1-schnell：快速生成模型
  - FLUX.1.1-pro：专业版模型
- 可调整参数：
  - 图像尺寸（宽度/高度）
  - 生成步数
  - 随机种子
  - 批量生成数量

### 🖼️ 图生图 (Image to Image)
- 支持多种转换模型：
  - FLUX.1-depth：深度图引导
  - FLUX.1-canny：边缘检测引导
  - FLUX.1-redux：通用图像生成
- 特点：
  - 上传本地图片
  - 从文生图结果导入
  - 自定义转换提示词

### 🔍 提示词识别
- 分析图片生成AI艺术提示词
- 提供多个角度的提示词建议
- 支持上传本地图片分析

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/nicekate/Together-Flux-Studio
cd Together-Flux-Studio
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行应用：
```bash
python app.py
```

4. 在浏览器中访问：
```
http://127.0.0.1:7860
```

## 使用要求

- Python 3.9+
- Together AI API访问权限

## Together API密钥配置

### Windows系统
1. 打开命令提示符（CMD）或PowerShell
2. 设置环境变量：
```powershell
setx TOGETHER_API_KEY "你的API密钥"
```
3. 重启命令提示符或PowerShell使设置生效

### macOS/Linux系统
1. 打开终端
2. 编辑shell配置文件（根据你使用的shell选择）：
   - Bash: `~/.bash_profile` 或 `~/.bashrc`
   - Zsh: `~/.zshrc`
3. 添加以下行：
```bash
export TOGETHER_API_KEY="你的API密钥"
```
4. 保存文件并重新加载配置：
```bash
source ~/.bash_profile  # 或 source ~/.zshrc
```

获取API密钥：
1. 访问 [Together AI](https://api.together.xyz)
2. 注册/登录账号
3. 在设置页面获取API密钥

## 使用说明

### 文生图
1. 选择模型
2. 输入详细的图像描述
3. 调整生成参数
4. 点击"Generate Images"
5. 等待图像生成完成
6. 下载生成的图片或发送到图生图功能继续处理

### 图生图
1. 上传图片或从文生图结果选择
2. 选择转换模型
3. 输入转换提示词（FLUX.1-redux模型可选）
4. 调整参数
5. 点击生成
6. 下载转换后的图片

### 提示词识别
1. 上传想要分析的图片
2. 点击"分析图片"
3. 查看生成的提示词建议

## 常见问题

### Q: 为什么图片生成速度较慢？
A: 生成速度受多个因素影响：
- 网络连接质量
- 选择的模型类型
- 服务器负载情况
- 图片尺寸和生成参数

### Q: 如何提高生成质量？
A: 可以尝试以下方法：
- 使用更详细的提示词描述
- 增加生成步数
- 尝试不同的随机种子
- 选择更高级的模型

### Q: API密钥无法使用怎么办？
A: 请检查：
- API密钥是否正确配置
- 账户余额是否充足
- 是否完成了必要的身份验证
- Together AI服务是否正常运行

## 注意事项

- 确保Together AI API密钥配置正确
- 部分高级模型需要信用卡验证
- 建议使用较新版本的现代浏览器
- 图片生成可能需要一定时间，请耐心等待

## 许可证

[MIT License](LICENSE)

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 致谢

- [Together AI](https://www.together.ai/) - 提供强大的AI模型和API支持
- [Gradio](https://gradio.app/) - 提供优秀的Web界面框架
