import torch
from flask import Flask, request, jsonify

from models.vgg16 import VGG16

app = Flask(__name__)

# cifar 100
model = VGG16(100, 3, 32, 32)
model.load_state_dict(
    torch.load(f'saved/best_cloud_vgg16_cifar100.pt', map_location='cpu'))
model.eval()
model.to('cuda:6')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        image = torch.tensor(image_data).permute(2, 0, 1)  # -> (C, H, W)
        image = image.unsqueeze(0)  # add B=1 (1,C,H,W)

        with torch.no_grad():
            outputs, _ = model(image)
            _, predicted = outputs.max(1) 

        return jsonify({"prediction": int(predicted.item())})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# this should be run on server side
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)