# LBM_post_process Win64 构建说明

最终用户不需要阅读本文件。它仅用于生成 Win64 便携包和安装包。

## 为什么必须在 Windows 构建

PyInstaller 会绑定当前操作系统的 Python 解释器和本地动态库，因此它不是交叉编译器。Win64 发行包必须在 Windows x64 或 Windows 云端构建机上生成。

## 方法一：Windows 电脑一键构建

### 小白方式（不使用 GitHub）

在 Windows 10/11 64 位电脑中完整解压项目，然后直接双击：

~~~text
一键生成Win64软件.bat
~~~

该脚本会使用 Windows 自带的 WinGet 自动安装缺少的 Python 3.12 和 Inno Setup，并完成全部构建步骤。首次运行需要联网下载数百 MB 的三维渲染和界面组件，通常需要 10–30 分钟。最终文件位于 `release` 文件夹。

如果提示找不到 WinGet，请在微软商店安装或更新“应用安装程序（App Installer）”，然后重新运行脚本。

### 开发者方式

准备：

- Windows 10/11 64 位
- [Python 3.12 x64](https://www.python.org/downloads/windows/)，安装时勾选 Python Launcher
- 可选：[Inno Setup 6](https://jrsoftware.org/isdl.php)，用于生成安装版 EXE

双击：

~~~text
tools\build_win64.bat
~~~

脚本会创建隔离的构建环境、安装构建依赖、运行 PyInstaller、自检并生成：

~~~text
release\LBM_post_process_win64_portable.zip
release\LBM_post_process_Setup_win64.exe   （安装了 Inno Setup 时）
~~~

## 方法二：GitHub Actions 云端构建

项目已包含 `.github/workflows/build-win64.yml`：

1. 将完整项目放入 GitHub 仓库。
2. 打开仓库的 Actions 页面。
3. 选择 `Build LBM_post_process Win64`。
4. 点击 `Run workflow`。
5. 构建完成后下载 `LBM_post_process-win64` Artifact。

云端流程会同时生成安装版 EXE 和便携版 ZIP。最终发行前，应在一台干净的 Windows 10/11 x64 电脑上测试打开 TIFF、3D 渲染、截图及批量处理。

## 发行方案说明

默认使用 PyInstaller one-folder 模式。VTK 和 PySide6 包含大量 DLL 与插件；one-folder 模式启动更快、故障率更低。Inno Setup 会把整个目录封装成一个面向用户的安装程序，因此安装者仍只需要双击一个 Setup EXE。

软件不要求用户安装 Python、VTK、PySide6 或 Visual Studio。发行包体积不会很小，因为三维渲染和 Qt 运行库必须随软件分发；`LBM_post_process.spec` 已排除 Matplotlib、Pandas、SciPy、Jupyter 和其他未使用的大型模块。

## 代码签名

当前脚本不包含商业代码签名证书。公开分发时建议为安装程序和主 EXE 配置可信的 Windows Authenticode 证书，以减少 SmartScreen 警告。证书私钥不应提交到源码仓库。
