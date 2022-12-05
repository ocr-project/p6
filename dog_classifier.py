import onnxruntime
import numpy as np
import time
import sys
from PIL import Image

from class_names import class_names

def cleanBreed(string):
  return string.split('-', 1)[1].replace('_', ' ')

ort_session = onnxruntime.InferenceSession("ImageDogClassifier.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def transpose(input, mean_std={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}):
    input = input.transpose((2, 0, 1))
    if mean_std != None:
        mean = mean_std['mean']
        std = mean_std['std']
        input[0] -= mean[0]
        input[1] -= mean[1]
        input[2] -= mean[2]
        input[0] /= std[0]
        input[1] /= std[1]
        input[2] /= std[2]
    return input

def get_input_batch_from_pil(filename):
    image = Image.open(filename)
    (w, h) = (image.width, image.height)
    if w > h:
        m = (w - h) /2
        x0, y0, x1, y1 = (m, 0, m+h, h)
    else:
        m = (h - w) /2
        x0, y0, x1, y1 = (0, m, w, m+w)

    image = image.crop((x0, y0, x1, y1))
    image = image.resize((224, 224))
    pix = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    pix = pix / 255
    pix = transpose(pix)
    pix = pix.astype('float32')
    pix_batch = np.expand_dims(pix,axis=0)

    return pix_batch

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cleanFile(string):
  return string.split('/')[-1].replace('.jpg', '')

latency = []
def run_sample(session, image_file, categories, inputs):
    start = time.time()
    input_arr = inputs
    ort_outputs = session.run([], {'input':input_arr})[0]
    latency.append(time.time() - start)
    output = ort_outputs.flatten()

    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:5]
    image = cleanFile(image_file)
    breed = ""
    print("{} {}".format(image, breed))
    for catid in top5_catid:
        print(" - {:<30} {:2f}".format(cleanBreed(categories[catid]), output[catid]))
    return ort_outputs
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Un ou plusieurs fichiers doivent être fourni en argument")
        sys.exit(0)
    filenames = sys.argv[1::]


for filename in filenames:
    session_fp32 = onnxruntime.InferenceSession("ImageDogClassifier.onnx", providers=['CPUExecutionProvider'])
    ort_output = run_sample(session_fp32, "" + filename, class_names, get_input_batch_from_pil(filename))
    
