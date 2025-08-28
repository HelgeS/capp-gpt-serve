from optimum.onnxruntime import ORTModelForCausalLM
import os
import sys

#  TODO Conversion need to provide token_mapping.json from initial dataset

if len(sys.argv) != 2:
    print("Specify model input path")
    sys.exit(1)

model_path = sys.argv[1]
onnx_path = "model"

model = ORTModelForCausalLM.from_pretrained(model_path, export=True)
model.save_pretrained(onnx_path)

open(os.path.join(onnx_path, "info.txt"), "w").write(model_path)
print(f"ONNX Model saved to {onnx_path}.")
