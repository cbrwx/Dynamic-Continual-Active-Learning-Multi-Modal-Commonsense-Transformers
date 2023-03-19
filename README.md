# Dynamic, Continual, and Active Learning with Multi-Modal and Commonsense-Enhanced Transformers

This repository contains the implementation of a Dynamic Transformer model that can adapt its architecture based on the complexity of the task it is being trained on. The model is built on PyTorch and can be used in a variety of learning settings, such as continual learning, multi-modal learning, commonsense reasoning, and active learning.

# Requirements
- Python 3.6 or later
- PyTorch 1.9.0 or later
- NumPy
- Pandas

# Configuration
The model configuration should be provided as a dictionary. The following keys are used in the configuration:

- 'vocab_size': The size of the vocabulary.
- 'embedding_dim': The size of the embeddings.
- 'initial_layers': The initial number of layers in the transformer.
- 'initial_attention_heads': The initial number of attention heads in the transformer.
- 'feedforward_dim': The dimension of the feedforward layers in the transformer.
Additional configurations can be added for specific types of learners, such as 'image_input_dim' and 'audio_input_dim' for the MultiModalLearner.

# Training
To train the model, create a DataLoader from the dataset and follow the training loop provided in the train method of each learner class. The training loop can be customized depending on your specific needs.

For some learners like the MetaLearner, you can optimize hyperparameters using the optimize_hyperparameters method. This method takes a dataset, the number of epochs, and the number of trials for the search. You can customize the search space for hyperparameters by modifying the search_space dictionary.

To train the model using a learning rate schedule, the learn_rate_schedule method can be used, which takes a dataset, the number of epochs, and the type of schedule as input. The available schedule types are 'step', 'cosine', and 'reduce_on_plateau'.

# Dataset Format and Structure
The dataset should be in a CSV file format, with two columns named 'text' and 'label'. Each row represents a single data point, with the 'text' column containing the input text and the 'label' column containing the corresponding label or target value.

Here's an example of how the CSV file should be formatted:
```
text,label
"Once upon a time, in a faraway land, there was a young prince.",0
"The prince was kind and generous to all the people in his kingdom.",0
"The kingdom was protected by a fierce dragon that lived in a cave.",1
"Many brave knights tried to defeat the dragon, but all of them failed.",1
"The prince decided to embark on a journey to find a way to defeat the dragon.",0
```
In this example, the 'text' column contains sentences, while the 'label' column contains binary labels (0 or 1). Depending on your specific task, the labels can be integers, strings, or other formats. Just make sure to update the CsvDataset class accordingly to handle the label type you are using.

To use this dataset with the CsvDataset class and DataLoader, first create an instance of CsvDataset, passing the file path as an argument, and then create a DataLoader instance with the CsvDataset object:
```
from torch.utils.data import DataLoader
from custom_classes import CsvDataset

csv_file_path = "path/to/your/csvfile.csv"
dataset = CsvDataset(csv_file_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```
Make sure to adjust the batch_size, shuffle, and num_workers parameters to fit your specific requirements.

# Custom Categories and Labels
The dataset can be tailored to any classification task by simply adjusting the labels and their corresponding meanings. The labels in the provided dataset are arbitrary, meaning that they can represent any category or classification target that you wish to use. In the previous example, we used binary labels (0 or 1) without specifying their meanings. To adapt the dataset to your specific task, you should provide a clear description of what each label represents.
```
text,label
"I love this product!",positive
"This is the worst purchase I have ever made.",negative
"I had an amazing time at the park.",positive
"I am so disappointed with this service.",negative
```
# Label Descriptions
For this sentiment analysis task, we use the following labels:

- positive: Represents a sentence that expresses a positive sentiment or emotion, such as happiness, satisfaction, or excitement.
- negative: Represents a sentence that expresses a negative sentiment or emotion, such as sadness, anger, or disappointment.
To use these custom labels with the CsvDataset class, you will need to update the class to handle string labels. Modify the __getitem__ method as follows:
```
def __getitem__(self, idx):
    row = self.data.iloc[idx]
    text = row["text"]
    label = row["label"]

    if self.tokenizer:
        text = self.tokenizer(text)

    return text, label
```
This change will ensure that the CsvDataset class returns the labels as strings. Note that you may need to adjust the model and training loop to accommodate these changes. For example, you might need to update the loss function and the output layer of your model to handle a multi-class classification problem.

# Main Function
The main function (main) is provided as an example of how to use the ActiveLearner class. In this example, a CsvDataset object is created from a CSV file, and an ActiveLearner is instantiated with a specified configuration. The most informative data points are selected from the dataset, and the model is trained using the selected data points. After training, the model is saved to a specified path.

You can modify the main function to use different learners or adapt it to your specific needs.

# CsvDataset
The CsvDataset class is a custom dataset class that loads data from a CSV file, which should have two columns named 'text' and 'label'. It inherits from PyTorch's Dataset class and implements the __init__, __len__, and __getitem__ methods. This class can be used with PyTorch's DataLoader to load the data in batches during training. If you have a custom dataset, you can modify the class to read data from your specific file format.

# MetaLearner
The MetaLearner class is designed for optimizing hyperparameters of the DynamicTransformer model using random search. It receives the model and optimizer as input during initialization. The optimize_hyperparameters method performs the random search over a specified search space (currently defined for learning rate and weight decay), trains the model for each set of hyperparameters, and saves the best model based on the evaluation metric (currently using negative average loss).

# ContinualLearner
The ContinualLearner class extends the DynamicTransformer class and is designed for continual learning. It incorporates Elastic Weight Consolidation (EWC) for mitigating catastrophic forgetting when learning multiple tasks sequentially. The class implements methods to compute Fisher information matrix, update model's memory, and calculate EWC loss. The train method is overridden to include EWC loss during the optimization.

# MultiModalLearner
The MultiModalLearner class extends the DynamicTransformer class and is designed for multi-modal learning. It includes additional image and audio embeddings and a modal_merge layer to combine embeddings from all modalities. The forward method is overridden to handle input from multiple modalities and merge their embeddings before passing them through the Transformer layers.

  To use the MultiModalLearner along with text, you'll need to modify the CsvDataset class to accommodate image and audio data. You can store the paths of the image and audio files in the CSV file and then load the actual data within the dataset class. You will also need to preprocess the image and audio data before feeding it into the model. Here's an example of how to modify the CsvDataset class and update the main function:

  - Modify the CsvDataset class:
  ```
  import torchvision.transforms as transforms
  from PIL import Image
  from torchaudio.transforms import Spectrogram, MelScale

  class CsvDataset(Dataset):
      def __init__(self, file_path, text_transform=None, image_transform=None, audio_transform=None):
          self.data = pd.read_csv(file_path)
          self.text_transform = text_transform
          self.image_transform = image_transform or transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.ToTensor()
          ])
          self.audio_transform = audio_transform or transforms.Compose([
              Spectrogram(),
              MelScale()
          ])

      def __getitem__(self, idx):
          text = self.data.iloc[idx]['text']
          image_path = self.data.iloc[idx]['image_path']
          audio_path = self.data.iloc[idx]['audio_path']
          label = self.data.iloc[idx]['label']

          if self.text_transform:
              text = self.text_transform(text)

          image = Image.open(image_path).convert('RGB')
          image = self.image_transform(image)

          audio, _ = torchaudio.load(audio_path)
          audio = self.audio_transform(audio)

          return text, image, audio, label
  ```
  - Update the main function to use MultiModalLearner:
  ```
  def main():
      # Load your data pool here
      data_pool = CsvDataset("path/to/your/csvfile.csv")

      # Configuration for the MultiModalLearner
      config = {
          'vocab_size': 10000,
          'embedding_dim': 512,
          'initial_layers': 6,
          'initial_attention_heads': 8,
          'feedforward_dim': 2048,
          'image_input_dim': 2048,
          'audio_input_dim': 128,
      }

      # Create the MultiModalLearner
      multimodal_learner = MultiModalLearner(config)

      # Train the model using the data pool
      n_epochs = 10
      optimizer = torch.optim.Adam(multimodal_learner.parameters(), lr=0.001)
      criterion = nn.CrossEntropyLoss()
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      dataloader = torch.utils.data.DataLoader(data_pool, batch_size=64, shuffle=True)

      for epoch in range(n_epochs):
          multimodal_learner.train()
          total_loss = 0
          num_batches = 0

          for text_inputs, image_inputs, audio_inputs, targets in dataloader:
              text_inputs, image_inputs, audio_inputs, targets = text_inputs.to(device), image_inputs.to(device), audio_inputs.to(device), targets.to(device)

              optimizer.zero_grad()
              outputs = multimodal_learner(text_inputs, image_inputs, audio_inputs)
              loss = criterion(outputs.view(-1), targets.long())
              loss.backward()
              optimizer.step()

              total_loss += loss.item()
              num_batches += 1

          avg_loss = total_loss / num_batches
          print("Epoch:", epoch, "Loss:", avg_loss)

      # Save the model after training
      multimodal_learner.save_model("path/to/save/model.pt")
  ```
# CommonsenseReasoner
The CommonsenseReasoner class extends the DynamicTransformer class and is designed for incorporating external knowledge from a graph-based knowledge base. The incorporate_external_knowledge method is provided as an example implementation, which updates node embeddings in the knowledge base by considering their neighbors' embeddings. The forward method is overridden to include the incorporation of external knowledge if a knowledge base is provided.

# ActiveLearner
The ActiveLearner class extends the DynamicTransformer class and is designed for active learning. It implements a select_data_points method that selects the most informative data points from the data pool using entropy as the uncertainty measure. In the main function, the data pool is loaded, the most informative data points are selected, and the model is trained using the selected data points. The model is saved after training.

In this repository im trying to provide a comprehensive implementation of a Dynamic Transformer model that aims to cover various learning scenarios, such as continual learning, multi-modal learning, commonsense reasoning, and active learning. Developed using PyTorch, the model can be customized and extended to cater to specific requirements. The code is designed to be easily adaptable, featuring examples of how to use different learner classes, modify dataset formats, and create custom categories and labels. This versatile and capable implementation aspires to assist in rapidly developing and experimenting with new learning techniques, ultimately contributing to the creation of more robust and intelligent models capable of tackling complex real-world problems.
