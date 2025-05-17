import json

def generate_yolov8_yaml(input_json_path, output_yaml_path, num_classes=80):
    """
    Generate a custom YOLOv8 YAML configuration file from a JSON input.
    
    Args:
        input_json_path (str): Path to the input JSON file (e.g., model_best.json)
        output_yaml_path (str): Path to save the generated YAML file
        num_classes (int): Number of classes for the model (default: 80)
    """
    # Load the JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Extract parameters
    params = data['params']
    depth_multiple = params['depth_multiple']
    width_multiple = params['width_multiple']
    img_size = params['img_size']
    kernel_size = params['kernel_size']
    
    # YOLOv8 YAML template with dynamic parameters
    yaml_content = f"""# Ultralytics YOLOv8 AGPL-3.0 license
# Generated from: {input_json_path}

# Parameters
nc: {num_classes}  # number of classes
img_size: [{img_size}, {img_size}]  # input image size

# Model scaling parameters
depth_multiple: {depth_multiple}  # model depth scaling factor
width_multiple: {width_multiple}   # model width scaling factor
kernel_size: {kernel_size}        # convolution kernel size

# YOLOv8n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, {kernel_size}, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, {kernel_size}, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, {kernel_size}, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, {kernel_size}, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, {kernel_size}, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)
  - [-1, 1, Conv, [256, {kernel_size}, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)
  - [-1, 1, Conv, [512, {kernel_size}, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
"""

    # Write to YAML file with UTF-8 encoding
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"Successfully generated YOLOv8 config at: {output_yaml_path}")

# Example usage
"""
generate_yolov8_yaml(
    input_json_path="model_best.json",
    output_yaml_path="yolov8_custom.yaml",
    num_classes=10  # Replace with your dataset's class count
)
"""