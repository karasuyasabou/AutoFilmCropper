# AutoFilmCropper

AutoFilmCropper 是一个用于胶片扫描工作流的桌面工具，帮助用户预览、校正并导出裁切后的 16-bit TIFF 胶片扫描图像。

## 功能概览

- 批量打开包含 TIFF / TIF 扫描文件的文件夹。
- 自动检测画面区域并生成初始裁切框。
- 支持负片与反转片两种片型检测模式。
- 支持常用比例（3:2、1:1、6:7）以及自定义裁切比例。
- 支持手动调整裁切框、旋转角度、曝光预览和逐张确认。
- 保留 16-bit 图像数据导出裁切结果。

## 运行环境

- Python 3.11+
- macOS / Windows / Linux（开发运行）

安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动应用：

```bash
python main.py
```

## 基本使用

1. 点击“打开文件夹”，选择包含 TIFF 扫描文件的目录。
2. 根据胶片类型选择“负片”或“反转片”。
3. 选择目标比例，必要时调整裁切框和角度。
4. 确认需要导出的照片。
5. 执行导出，生成裁切后的 TIFF 文件。

## 构建发布包

项目包含 GitHub Actions 发布流程：推送 `v*` 标签或手动触发 workflow 后，会构建 macOS 与 Windows 发布包。

也可以在本地使用 PyInstaller 相关脚本进行构建：

```bash
python scripts/build_release.py --platform macos-arm64
```

## 项目结构

```text
AutoFilmCropper/
├── .github/workflows/      # GitHub Actions 发布流程
├── assets/                 # 应用图标等静态资源
├── build_support/          # 构建元数据
├── scripts/                # 构建辅助脚本
├── image_core.py           # 图像处理与裁切检测核心
├── main.py                 # 应用入口与主窗口控制逻辑
├── ui.py                   # 图形视图与交互组件
└── requirements.txt        # Python 依赖
```

## 版本发布

创建并推送版本标签即可触发自动构建与 GitHub Release：

```bash
git tag v1.0.0
git push origin v1.0.0
```
