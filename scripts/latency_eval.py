import time
import requests
import torch
import torchvision

# this should be run on the local edge side
SERVER_URL = "http://127.0.0.1:3000/predict"

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
test_dataset = torchvision.datasets.CIFAR100(root='.', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

start_time = time.time()
predictions = []
for idx, (image, label) in enumerate(test_loader):
    image_data = image.squeeze(0).permute(1, 2, 0).numpy().tolist()
    payload = {"image": image_data}

    response = requests.post(SERVER_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Sample {idx}: Prediction = {result['prediction']}")
    else:
        print(f"Sample {idx}: Failed with error {response.text}")

end_time = time.time() 
total_time = end_time - start_time
print(f"Average processed time for cifar sample: {total_time / len(test_dataset) * 1000:.2f}ms.")
