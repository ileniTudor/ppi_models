import torch
import torch.nn as nn
from torchvision.models import resnet18

# Assuming your model setup as described:
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet18(pretrained=False, num_classes=2)
model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming 2 classes
state_dict = torch.load('../flask_app_DL/best_model_resnet18.pth', map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)  # Batch size 1, 3 channels, 224x224 resolution

# Export to ONNX
onnx_path = "../flask_app_DL/best_model_resnet18.onnx"
torch.onnx.export(
    model,                      # Model being converted
    dummy_input,                # Example input tensor
    onnx_path,                  # Output file name
    export_params=True,         # Store the trained parameters in the model file
    opset_version=11,           # ONNX version (11 is widely supported)
    do_constant_folding=True,   # Optimize constant folding for inference
    input_names=["input"],      # Model input name
    output_names=["output"],    # Model output name
    # dynamic_axes={              # Enable dynamic axes for variable batch sizes
    #     "input": {0: "batch_size"},
    #     "output": {0: "batch_size"}
    # }
)

print(f"Model has been successfully exported to {onnx_path}")
