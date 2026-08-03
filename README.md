# cvkit

`cvkit` 是一个面向计算机视觉数据处理的轻量 Python 工具库，提供图像读写、
TXT/XML 文档操作、YOLO/VOC 标注处理、数据集检查与划分、二值 mask 生成与
图像修复，以及基于 Albumentations 的数据增强。当前版本为 `0.0.3`。

## 环境要求

- Python 3.10+
- NumPy 1.24.4+
- OpenCV Headless 4.9.0.80+
- Albumentations 2.x

在项目根目录安装开发版本：

```bash
python -m pip install -e .
```

`MaskImageUtils` 的图像修复功能还依赖
[`simple-lama-inpainting`](https://pypi.org/project/simple-lama-inpainting/)。
该依赖目前未写入 `pyproject.toml`，使用 mask 模块前需要在同一 Python 或 Conda
环境中额外安装：

```bash
python -m pip install simple-lama-inpainting
```

## 功能概览

| 模块 | 功能 |
| --- | --- |
| `TxtDocument` | TXT 读取、写入、追加和另存为 |
| `XMLDocument` | XML 创建、查询、修改节点和保存 |
| `YOLOAnnotationUtils` | YOLO HBB 解析、类别映射、删除类别、去重和统计 |
| `YOLOSegmentationUtils` | YOLO 分割多边形转换为 YOLO HBB |
| `VOCAnnotationUtils` | VOC XML 创建、解析、重命名和格式转换 |
| `Datasets` | YOLO 数据集检查和训练集/验证集划分 |
| `ImageIO` | OpenCV 图像读取和保存 |
| `MaskImageUtils` | 根据分割或矩形标签生成 mask、保存 mask 外接框、膨胀 mask 和 LaMa 修复 |
| `Augmenter` | 链式构建图像增强，并支持多进程批量增强 |

`cvkit.ocr` 当前为预留模块，暂未提供公开功能。

## 数据集目录约定

数据集相关功能默认使用以下目录结构：

```text
dataset/
├── images/
│   ├── image001.jpg
│   └── image002.png
└── labels/
    ├── image001.txt
    └── image002.txt
```

图片和标签通过相同的文件名主干匹配，例如 `image001.jpg` 对应
`image001.txt`。

## TXT 文档

导入：

```python
from cvkit.core.annotation.io.txt import TxtDocument
```

### 读取

```python
document = TXTDocument("labels/image001.txt")

content = document.read()
lines = document.readlines()
```

### 写入和追加

`write()` 和 `append()` 只修改内存内容，调用 `save()` 后才写入文件。

```python
(
    TXTDocument("output.txt")
    .write("first line\n")
    .append("second line\n")
    .save()
)
```

### 另存为

```python
(
    TXTDocument("source.txt")
    .write("new content\n")
    .save("outputs/result.txt")
)
```

不传 `save_path` 时保存到对象原路径。

## XML 文档

导入：

```python
from cvkit.core.annotation.io.xmlio import XMLDocument
```

### 打开已有 XML

```python
document = XMLDocument("annotations/image001.xml")

xml_text = document.read()
width = document.gettext("size/width")
objects = document.findall("object")
```

### 创建 XML

`create()` 在内存中创建文档，`save()` 负责写入文件并自动创建父目录。

```python
(
  XMLDocument.new(
    "outputs/example.xml",
    root_tag="annotation",
  )
  .append_node(".", "filename", text="image001.jpg")
  .append_node(".", "size")
  .append_node("size", "width", text="1920")
  .append_node("size", "height", text="1080")
  .save()
)
```

`.` 表示 XML 根节点。

### 查询和修改

```python
document = XMLDocument("annotations/image001.xml")

node = document.find("size/width")
all_objects = document.findall("object")
text = document.gettext("filename")

(
    document
    .update_text("filename", "renamed.jpg")
    .update_attr("object", "verified", "true")
    .save()
)
```

删除节点：

```python
XMLDocument("example.xml").remove_node("object").save()
```

## YOLO HBB 标签

YOLO HBB 每行格式：

```text
class_id x_center y_center width height
```

坐标为相对于图片宽高的归一化值。

导入：

```python
from cvkit.core.annotation.hbb.yolo import YOLOAnnotationUtils
```

### 解析标签

```python
bboxes, classes = YOLOAnnotationUtils(
  "labels/image001.txt"
).parse_label()

print(bboxes)
print(classes)
```

返回值示例：

```python
bboxes = [[0.5, 0.5, 0.2, 0.3]]
classes = [0]
```

### 类别映射

```python
YOLOAnnotationUtils(
    "labels/image001.txt"
).remap_classes({
    0: 2,
    1: 0,
})
```

`remap_classes()` 会直接覆盖原标签文件。

### 删除类别

```python
YOLOAnnotationUtils(
    "labels/image001.txt"
).del_cls(2)
```

`del_cls()` 会直接覆盖原标签文件。

### 删除空标签文件

```python
YOLOAnnotationUtils(
    "labels/image001.txt"
).remove_empty()
```

标签文件完全为空时，`remove_empty()` 会删除该文件。

### 去除重复标注

```python
YOLOAnnotationUtils(
    "labels/image001.txt"
).remove_duplicate()
```

该方法按整行内容去重，并覆盖原标签文件。

### 统计类别

```python
counts = YOLOAnnotationUtils(
    "labels/image001.txt"
).count_classes()

print(counts)
```

返回 `collections.Counter`，例如：

```python
Counter({0: 5, 1: 2})
```

## YOLO 分割标签

YOLO 分割标签每行格式：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

导入：

```python
from cvkit.core.annotation.seg.yolo import YOLOSegmentationUtils
```

### 多边形转 HBB

```python
(
    YOLOSegmentationUtils("segment/image001.txt")
    .polygon_2_bbox()
    .save("labels/image001.txt")
)
```

该方法计算每个多边形的水平外接矩形，并输出 YOLO
`class_id x_center y_center width height` 格式。

不向 `save()` 传入新路径时会覆盖原分割标签，应根据数据保留需求选择保存位置。

## VOC 标注

导入：

```python
from cvkit.core.annotation.hbb.voc import VOCAnnotationUtils
```

### 创建 VOC XML

`build_annotation()` 根据 `bboxes` 创建一个或多个同类别的 VOC `object`。该方法
只构建内存文档，必须继续调用 `save()` 才会写入 XML。

```python
(
  VOCAnnotationUtils.build_annotation(
    img_name="image001.jpg",
    img_size=(1920, 1080),
    bboxes=[
      [100, 200, 500, 600],
      [700, 300, 900, 650],
    ],
    save_path="annotations/image001.xml",
    class_name="defect",
    depth=3,
  )
  .save()
)
```

### 读取类别和矩形框

```python
document = VOCAnnotationUtils(
    "annotations/image001.xml"
)

names = document.get_voc_label_names()
exists = document.is_label_in_voc("defect")
image_size, bboxes = document.parse_voc()
```

`parse_voc()` 返回：

```python
image_size = (1920, 1080)
bboxes = [
    [100, 200, 500, 600, "defect"],
    [700, 300, 900, 650, "defect"],
]
```

### 重命名类别

```python
VOCAnnotationUtils(
    "annotations/image001.xml"
).rename_voc_label(
    new_label="scratch",
    old_label="defect",
)
```

该方法会直接保存并覆盖原 XML。

### VOC 转 YOLO

```python
VOCAnnotationUtils(
    "annotations/image001.xml"
).save_as_yolo("labels")
```

如果不传保存目录，默认写入 XML 所在数据集根目录下的 `labels/`。

类别 ID 根据当前 XML 文件中的类别名称排序生成。批量转换多个 XML 时，应确认不同
文件生成的类别映射是否满足训练配置要求。

### VOC 转 JSON

```python
VOCAnnotationUtils(
    "annotations/image001.xml"
).save_as_json(
    save_path="json",
    image_suffix="jpg",
)
```

输出内容包含矩形 shape、图片宽高和图片文件名。

## 数据集检查与划分

导入：

```python
from cvkit.core.dataset.base import Datasets
```

### 检查数据集

```python
Datasets("dataset").check()
```

检查内容：

- OpenCV 无法读取的坏图片；
- 图片缺少同名标签；
- 空标签；
- 标签缺少同名图片。

无法读取的坏图片会被移动到：

```text
dataset/images_bad/
```

因此 `check()` 不完全是只读操作。

### 划分训练集和验证集

```python
Datasets(
    "dataset",
    ratio=0.9,
).split()
```

输出目录：

```text
dataset/split/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

`ratio` 必须位于 `0` 和 `1` 之间。划分以标签文件为入口，支持
`.jpg`、`.jpeg` 和 `.png` 图片。当前划分使用全局随机状态且没有固定随机种子；
重复运行前还应自行清理旧的 `split/`，避免上一次结果残留。

## 图像读写

导入：

```python
from cvkit.core.image.io import ImageIO
```

### 读取图像

```python
document = ImageIO("images/image001.jpg").read()

image = document.image
height = document.height
width = document.width
```

OpenCV 默认以 BGR 格式读取彩色图片。

转换为 RGB：

```python
import cv2

rgb = cv2.cvtColor(
    document.image,
    cv2.COLOR_BGR2RGB,
)
```

转换为灰度图：

```python
gray = cv2.cvtColor(
    document.image,
    cv2.COLOR_BGR2GRAY,
)
```

### 保存图像

保存 `self.image`：

```python
ImageIO("images/image001.jpg").read().save(
    save_path="outputs/image001.png"
)
```

保存指定的 NumPy 图像：

```python
document.save(
    image=gray,
    save_path="outputs/image001-gray.png",
)
```

## 二值 Mask

导入：

```python
from cvkit.core.image.mask import MaskImageUtils
```

`MaskImageUtils` 按下面的目录结构自动查找标签，构造时不接收 `label_path`：

```text
dataset/
├── images/
│   └── image001.jpg
└── labels/
    └── image001.txt
```

初始化时会立即读取原图。mask 使用以下像素约定：

- `0`：黑色背景，不擦除；
- `255`：白色目标区域，需要擦除。

`cls` 指定参与生成的类别，`erase_num` 指定从该类别中随机选择多少个标注区域。
生成方法只更新内存中的 mask；调用 `save_mask()` 后默认保存为 PNG：

```text
dataset/mask/<image_stem>.png
```

### 根据 YOLO 分割标签生成

```python
(
    MaskImageUtils(
        image_path="dataset/images/image001.jpg",
        cls=0,
        erase_num=2,
    )
    .generate_from_seg()
    .save_mask()
)
```

该方法读取 YOLO 分割标签，从类别 `0` 的所有多边形中随机选择两个并填充为白色。

### 根据 YOLO HBB 标签生成

```python
(
    MaskImageUtils(
        image_path="dataset/images/image001.jpg",
        cls=0,
        erase_num=2,
    )
    .generate_from_hbb()
    .save_mask()
)
```

该方法从类别 `0` 的所有矩形框中随机选择两个，并将它们填充为白色。

当 `erase_num=0` 时生成全黑 mask；当 `erase_num` 大于目标类别标注数量时抛出
`ValueError`。

### 保存 mask 外接框

```python
(
  MaskImageUtils(
    image_path="dataset/images/image001.jpg",
    cls=0,
    erase_num=2,
  )
  .generate_from_seg()
  .save_mask_bbox_as_yolo()
)
```

`save_mask_bbox_as_yolo()` 使用 `cv2.RETR_EXTERNAL` 提取所有外部轮廓，并为每个有效轮廓
生成一个 YOLO HBB。标签保存到：

```text
dataset/cachu/labels/image001.txt
```

多个互不相连的 mask 会生成多行标注；相互接触或重叠的区域可能被识别为一个轮廓。
输出标签使用构造 `MaskImageUtils` 时传入的 `cls` 作为类别 ID。可以通过
`save_mask_bbox_as_yolo(save_path=...)` 指定其他保存位置。

### 膨胀 mask

```python
utils = (
  MaskImageUtils(
    image_path="dataset/images/image001.jpg",
    cls=0,
    erase_num=1,
  )
  .generate_from_seg()
  .expand_mask(kernel_size=5)
)
```

`expand_mask()` 使用椭圆形结构元素执行一次膨胀。`kernel_size=1` 使用
`1 x 1` 核，不会产生实际膨胀效果。

### LaMa 图像修复

推荐在膨胀前保存标注，让标注描述原始目标区域；膨胀后的 mask 仅供图像修复使用：

```python
from pathlib import Path

from cvkit.core.image.mask import MaskImageUtils

image_path = Path("dataset/images/image001.jpg")
result_path = (
        image_path.parents[1]
        / "cachu"
        / "images"
        / f"{image_path.stem}.png"
)

(
  MaskImageUtils(
    image_path=image_path,
    cls=0,
    erase_num=1,
  )
  .generate_from_seg()
  .save_mask_bbox_as_yolo()
  .expand_mask(kernel_size=5)
  .repair()
  .save_result(result_path)
)
```

`repair()` 使用 `simple_lama_inpainting.SimpleLama` 修复白色 mask 区域。LaMa 会把
输入宽高补齐到 `8` 的倍数，当前实现不会自动裁剪回原始尺寸；当原图宽高不是
`8` 的倍数时，输出尺寸可能增大。

### 保存 mask 分割标注

`save_mask_seg_as_yolo()` 会提取 mask 的所有有效外部轮廓，将轮廓坐标按原图宽高
归一化，并保存为 YOLO 分割标签：

```python
(
  MaskImageUtils(
    image_path="dataset/images/image001.jpg",
    cls=0,
    erase_num=2,
  )
  .generate_from_seg()
  .save_mask_seg_as_yolo()
)
```

标签默认保存到 `dataset/cachu/labels/<image_stem>.txt`。多个互不相连的 mask
轮廓会生成多行分割标注，也可以通过 `save_path` 指定其他保存位置。

## 数据增强

导入：

```python
from cvkit.augment.aug import Augmenter
```

输入数据集应包含：

```text
dataset/
├── images/
└── labels/
```

标签格式为 YOLO HBB。

### 构建 Albumentations 流程

```python
transform = (
    Augmenter("dataset")
    .horizontal_flip(p=0.5)
    .rotate(limit=10, p=0.5)
    .brightness(p=0.5)
    .build(bbox=True)
)
```

可用增强方法：

- `horizontal_flip()`：水平翻转；
- `vertical_flip()`：垂直翻转；
- `rotate()`：随机旋转；
- `brightness()`：随机亮度和对比度；
- `blur()`：模糊；
- `gauss_noise()`：高斯噪声；
- `image_compression()`：JPEG/WebP 压缩模拟；
- `clahe()`：局部直方图均衡；
- `to_gray()`：随机转换为灰度图；
- `motion_blur()`：运动模糊；
- `one_of_sharp_blur()`：锐化或模糊组合；
- `one_of_affine()`：横向或纵向缩放组合。

### 批量增强

```python
(
    Augmenter(
        "dataset",
        repeat=10,
    )
    .horizontal_flip()
    .brightness()
    .gauss_noise()
    .augment(worker=1)
)
```

输出目录：

```text
dataset/aug/
├── images/
└── labels/
```

每张输入图片生成 `repeat` 份增强结果，图片与 YOLO 标签保持同名对应。

### 多进程增强

```python
from cvkit.augment.aug import Augmenter


if __name__ == "__main__":
    (
        Augmenter(
            "dataset",
            repeat=10,
        )
        .horizontal_flip()
        .rotate(limit=10)
        .augment(worker=4)
    )
```

`worker<=1` 使用串行处理，`worker>1` 使用 `ProcessPoolExecutor`。使用多进程时，
入口脚本必须添加 `if __name__ == "__main__":`。

## 构建 Wheel

安装构建工具：

```bash
python -m pip install build
```

构建：

```bash
python -m build --wheel
```

或者执行：

```bash
bash build.sh
```

生成的 wheel 位于 `dist/`。

## 注意事项

- 部分标注处理方法会直接覆盖或删除原文件，批量执行前应先备份数据。
- `Datasets.check()` 会把坏图片移动到 `images_bad/`。
- mask 生成和数据增强包含随机选择，每次运行结果可能不同。
- `VOCAnnotationUtils.build_annotation()` 支持多个框，但一次调用中的所有框共用同一个
  `class_name`。
- `VOCAnnotationUtils.save_as_yolo()` 按单个 XML 内的类别名称排序并从 `0` 生成
  类别 ID，不维护跨文件的全局类别映射。
- `XMLDocument.append_node()` 通过 XPath 查找父节点；存在多个同名父节点时会使用
  第一个匹配节点。需要操作多个同名节点时应使用带索引的路径，例如 `object[2]`。
- 项目当前处于 Alpha 阶段，接口仍可能调整。
