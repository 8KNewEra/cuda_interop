import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "E:/cuda_interop/project/models/FSRCNN_x2.onnx"
)

x = np.random.rand(
    1,3,256,256
).astype(np.float32)

y = sess.run(
    None,
    {"input":x}
)[0]

print(y.shape)