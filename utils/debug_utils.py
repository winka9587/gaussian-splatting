from PIL import Image
import torch

def save_rendered_image(render_tensor, file_path='rendered_image.png'):
    """
    将渲染的 3D 图像保存为 PNG 文件。

    参数:
        render_tensor (torch.Tensor): 渲染的图像，形状为 (3, H, W)。
        file_path (str): 保存 PNG 文件的路径，默认为 'rendered_image.png'。
    """
    # 确保 tensor 在 CPU 上
    render_tensor = render_tensor.cpu()

    # 转换为 (H, W, 3) 格式
    render_tensor = render_tensor.permute(1, 2, 0)

    # 将值转换为 [0, 255] 范围并转为 uint8 类型
    render_tensor = (render_tensor * 255).byte()

    # 使用 PIL 保存为 PNG 文件
    image = Image.fromarray(render_tensor.numpy())
    image.save(file_path)
    print(f"Image saved as {file_path}")