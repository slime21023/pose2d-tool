# pose-estimation tools
Based on the google `movenet-singlepose-lightning`

### Download the model
```bash
python script/save_movenet.py
```

### Convert the tf saved model to the onnx model

```bash
python -m tf2onnx.convert --saved-model dist/movenet-lightning --output dist/onnx/movenet-lightning.onnx --opset 13
```

### Test the onnx model inference

```bash
python script/onnx/inference.py
```