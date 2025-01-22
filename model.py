import torch.nn as nn
import torch.nn.functional as F

# class SpeechEmotionRecognitionModel(nn.Module):
#     def __init__(self):
#         super(SpeechEmotionRecognitionModel, self).__init__()
#         self.fc1 = nn.Linear(13, 64)  # 13 = Number of MFCC features
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 4)  # 4 = Number of emotion classes

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
class SpeechEmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(SpeechEmotionRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x
