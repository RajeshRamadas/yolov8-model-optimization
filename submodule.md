# YOLOv8 Model Optimization

This repository contains implementation and optimization techniques for YOLOv8 object detection models.

## Repository Structure

- `02-Implementation/`: Implementation code and submodules
  - `highway-vehicle-tracker/`: Submodule for highway vehicle tracking implementation

## Submodule Details

This repository includes the [highway-vehicle-tracker](https://github.com/RajeshRamadas/highway-vehicle-tracker.git) as a submodule in the `02-Implementation` directory. The submodule provides specialized functionality for tracking vehicles on highways using the YOLOv8 model.

## Setup Instructions

### Cloning the Repository with Submodules

To clone this repository along with its submodules, use:

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/RajeshRamadas/yolov8-model-optimization.git

# Navigate to the repository
cd yolov8-model-optimization
```

If you've already cloned the repository without the submodules, initialize and update them:

```bash
git submodule init
git submodule update
```

### Updating Submodules

To update the submodules to their latest versions:

```bash
# Navigate to the submodule directory
cd 02-Implementation/highway-vehicle-tracker

# Update to the latest commit on main branch
git checkout main
git pull origin main

# Go back to the main repository
cd ../..

# Commit the submodule update
git add 02-Implementation/highway-vehicle-tracker
git commit -m "Update highway-vehicle-tracker submodule"
git push origin main
```

### Configuration for Easier Submodule Management

For easier ongoing management, you can configure Git to always update submodules:

```bash
git config --global submodule.recurse true
```

With this configuration, operations like `git pull` will automatically update submodules too.

## Usage

Details on how to use the YOLOv8 model optimization techniques and the highway vehicle tracker will be provided in their respective directories.

## Requirements

- Git 2.13 or newer
- Python 3.8 or newer (for YOLOv8)
- Additional requirements are specified in the individual implementation directories

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

[Specify your license here]

## Contact

[Your contact information]
