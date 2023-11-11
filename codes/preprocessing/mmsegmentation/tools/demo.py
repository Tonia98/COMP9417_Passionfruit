from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
# IMG_4862.jpg
# D:\\unsw\\seg\\1\\mmsegmentation-0.20.2\\mmsegmentation-0.20.2\\configs\\deeplabv3plus\\1my_deeplabv3plus_r50-d8_480x480_40k_pascal_context.py
# D:\\unsw\\seg\\1\\mmsegmentation-0.20.2\\mmsegmentation-0.20.2\\tools\\work_dirs\\deeplabv3plus_r50-d8_480x480_40k_pascal_context\\iter_1000.pth
config_file = r'D:\unsw\9417\hw3\mmsegmentation-0.20.2\configs\deeplabv3plus\my_deeplabv3plus_r50-d8_480x480_40k_pascal_context.py'
checkpoint_file = r'D:\unsw\seg\1\mmsegmentation-0.20.2\mmsegmentation-0.20.2\tools\work_dirs\deeplabv3plus_r50-d8_480x480_40k_pascal_context\latest.pth'

# 从一个 config 配置文件和 checkpoint 文件里创建分割模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
import os

folder_path = r"D:\unsw\9417\data\archive\RiceLeafs\validation\BrownSpot"
for filename in os.listdir(folder_path):
    abs_path = os.path.join(folder_path, filename)
    # 测试一张样例图片并得到结果
    img = abs_path
    result = inference_segmentor(model, img)
    # 在新的窗口里可视化结果
    model.show_result(img, result, show=True)
    # 或者保存图片文件的可视化结果
    # 您可以改变 segmentation map 的不透明度(opacity)，在(0, 1]之间。
    model.show_result(img, result, out_file=f'./output/validation1/BrownSpot/{filename}', opacity=0.5)



