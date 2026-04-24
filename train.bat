@echo off
echo ================================================
echo nnU-Net 训练脚本 (Windows)
echo ================================================
echo.

REM 设置环境变量
set nnUNet_raw=data\nnUNet_raw
set nnUNet_preprocessed=data\nnUNet_preprocessed
set nnUNet_results=models

REM 检查 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查 nnU-Net
python -c "import nnunetv2" >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] nnU-Net 未安装
    echo 请运行：pip install nnunetv2
    pause
    exit /b 1
)

REM 检查数据
if not exist "data\nnUNet_raw\Dataset501_BUSI\dataset.json" (
    echo [错误] 数据未准备好
    echo 请先运行：python scripts\01_preprocess.py
    pause
    exit /b 1
)

echo [1/2] 预处理数据...
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

if %errorlevel% neq 0 (
    echo [错误] 预处理失败
    pause
    exit /b 1
)

echo.
echo [2/2] 训练模型...
echo 训练时间：约 1-2 小时（GPU）
echo.

REM 运行训练
nnUNetv2_train 501 2d 0

echo.
echo ================================================
echo 训练完成！
echo ================================================
echo.
echo 下一步：
echo   启动 Web 界面：python scripts\05_web_app.py
echo   单图预测：python scripts\04_predict.py <图像路径>
echo.

pause
