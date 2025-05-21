import json

def generate_yolov8_yaml(input_json_path: str, output_yaml_path: str,
                         num_classes: int = 80, reduce_channels_pct: float = 0.0):
    """
    Generate a custom YOLOv8 YAML configuration file from a JSON input with optional channel reduction.

    Args:
        input_json_path (str): Path to the input JSON file (e.g., model_best.json)
        output_yaml_path (str): Path to save the generated YAML file
        num_classes (int): Number of classes for the model (default: 80)
        reduce_channels_pct (float): Percentage (0–1) to reduce each channel (e.g., 0.3 for 30% reduction)
    """
    def reduce(channels: int) -> int:
        reduced = int(channels * (1.0 - reduce_channels_pct))
        return max(reduced, 16)  # Ensure minimum channel count

    # Load the JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    params = data['params']
    depth_multiple = params['depth_multiple']
    width_multiple = params['width_multiple']
    img_size = params['img_size']
    kernel_size = params['kernel_size']

    # YAML content
    yaml_content = f"""# Ultralytics YOLOv8 AGPL-3.0 license
# Generated from: {input_json_path}

# Parameters
nc: {num_classes}  # number of classes
img_size: [{img_size}, {img_size}]  # input image size

# Model scaling parameters
depth_multiple: {depth_multiple}  # model depth scaling factor
width_multiple: {width_multiple}  # model width scaling factor
kernel_size: {kernel_size}        # convolution kernel size
reduce_channels_pct: {reduce_channels_pct}  # reduce each channel by this percentage

# YOLOv8n backbone
backbone:
  - [-1, 1, Conv, [{reduce(64)}, {kernel_size}, 2]]  # 0-P1/2
  - [-1, 1, Conv, [{reduce(128)}, {kernel_size}, 2]]  # 1-P2/4
  - [-1, 3, C2f, [{reduce(128)}, True]]
  - [-1, 1, Conv, [{reduce(256)}, {kernel_size}, 2]]  # 3-P3/8
  - [-1, 6, C2f, [{reduce(256)}, True]]
  - [-1, 1, Conv, [{reduce(512)}, {kernel_size}, 2]]  # 5-P4/16
  - [-1, 6, C2f, [{reduce(512)}, True]]
  - [-1, 1, Conv, [{reduce(1024)}, {kernel_size}, 2]]  # 7-P5/32
  - [-1, 3, C2f, [{reduce(1024)}, True]]
  - [-1, 1, SPPF, [{reduce(1024)}, 5]]  # 9

# YOLOv8n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [{reduce(512)}]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [{reduce(256)}]]
  - [-1, 1, Conv, [{reduce(256)}, {kernel_size}, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [{reduce(512)}]]
  - [-1, 1, Conv, [{reduce(512)}, {kernel_size}, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [{reduce(1024)}]]
  - [[15, 18, 21], 1, Detect, [nc]]
"""

    # Write YAML file
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"YOLOv8 config with reduced channels saved to: {output_yaml_path}")
