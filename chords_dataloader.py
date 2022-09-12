import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

classes = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
    "E" : 4,
    "F" : 5,
    "G" : 6
}

class ChordClassificationDataset(Dataset):
    def __init__(self, csv_file, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(csv_file, encoding="ISO-8859-1")
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples-length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self.annotations.iloc[index, 0]
        y_label = (self.annotations.iloc[index, 1])
        y_label = torch.Tensor([classes[y_label]])
        signal, sr = torchaudio.load(audio_path)
        signal = signal.to(self.device)
        signal = self.resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self.cut_if_necessary(signal)
        signal = self.right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, y_label

    def _get_audio_sample_path(self, index):
        return self.annotations.iloc[index, 0]

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]



""""
transform = transforms.Compose([transforms.ToTensor()])

dataset = CancerClassificationDataset("train_custom.csv",
                                      "/home/geobots/project/classifier-2/sample_single_output/cropped",
                                      transform=transform)


dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

for batch_idx, (data, y, image) in enumerate(dataloader):
    x = data
    print(image.shape, type(image), y)
    y = y
    for i in range(len(x)):
        print(x[i], y[i])
    if batch_idx == 10:
        break
"""