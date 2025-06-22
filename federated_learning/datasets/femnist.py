# # # # # # # # from .dataset import Dataset
# # # # # # # # from torchvision import transforms
# # # # # # # # from torch.utils.data import DataLoader
# # # # # # # # from PIL import Image
# # # # # # # # import torch
# # # # # # # # import os
# # # # # # # # import shutil
# # # # # # # # from torchvision.datasets.utils import download_and_extract_archive
# # # # # # # # from typing import Any

# # # # # # # # class FEMNISTDataset(Dataset):
# # # # # # # #     def __init__(self, args: Any) -> None:
# # # # # # # #         # Initialize attributes before calling the parent class's __init__
# # # # # # # #         self.root = args.get_data_path()  # Dataset root path
# # # # # # # #         self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
# # # # # # # #         self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'

# # # # # # # #         # Define file paths
# # # # # # # #         self.training_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_train.pt')
# # # # # # # #         self.test_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_test.pt')
# # # # # # # #         self.user_list = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_user_keys.pt')

# # # # # # # #         # Call the parent class's __init__ AFTER initializing attributes
# # # # # # # #         super(FEMNISTDataset, self).__init__(args)

# # # # # # # #         # Ensure dataset files are available (download if necessary)
# # # # # # # #         self.dataset_download()

# # # # # # # #     def load_train_dataset(self):
# # # # # # # #         self.get_args().get_logger().debug("Loading FEMNIST train data")

# # # # # # # #         # Check if both training and test data exist before downloading
# # # # # # # #         if not os.path.exists(self.training_file) or not os.path.exists(self.test_file):
# # # # # # # #             self.dataset_download()

# # # # # # # #         # Load data and targets after ensuring download
# # # # # # # #         if not os.path.exists(self.training_file):
# # # # # # # #             raise RuntimeError(f'Training data not found at {self.training_file}. Please ensure it is downloaded.')

# # # # # # # #         data_targets_users = torch.load(self.training_file)
# # # # # # # #         self.data, self.targets, self.users = torch.Tensor(data_targets_users[0]), torch.Tensor(data_targets_users[1]), data_targets_users[2]

# # # # # # # #         transform = transforms.Compose([transforms.ToTensor()])
# # # # # # # #         train_dataset = torch.utils.data.TensorDataset(self.data, self.targets, self.users)
# # # # # # # #         train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # # # # # # #         train_data = self.get_tuple_from_data_loader(train_loader)
# # # # # # # #         self.get_args().get_logger().debug("Finished loading FEMNIST train data")
# # # # # # # #         return train_data

# # # # # # # #     def load_test_dataset(self):
# # # # # # # #         self.get_args().get_logger().debug("Loading FEMNIST test data")

# # # # # # # #         if not os.path.exists(self.test_file):
# # # # # # # #             raise RuntimeError(f'Test data not found at {self.test_file}. Please ensure it is downloaded.')

# # # # # # # #         data_targets_users = torch.load(self.test_file)
# # # # # # # #         self.data, self.targets, self.users = torch.Tensor(data_targets_users[0]), torch.Tensor(data_targets_users[1]), data_targets_users[2]

# # # # # # # #         transform = transforms.Compose([transforms.ToTensor()])
# # # # # # # #         test_dataset = torch.utils.data.TensorDataset(self.data, self.targets, self.users)
# # # # # # # #         test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # # # # # # #         test_data = self.get_tuple_from_data_loader(test_loader)
# # # # # # # #         self.get_args().get_logger().debug("Finished loading FEMNIST test data")
# # # # # # # #         return test_data

# # # # # # # #     def __getitem__(self, index: int):
# # # # # # # #         img, target, user = self.data[index], int(self.targets[index]), self.users[index]
# # # # # # # #         img = Image.fromarray(img.numpy(), mode='F')
# # # # # # # #         return img, target, user

# # # # # # # #     def dataset_download(self):
# # # # # # # #         # Create required directories
# # # # # # # #         paths = [os.path.join(self.root, 'FEMNIST', 'raw'), os.path.join(self.root, 'FEMNIST', 'processed')]
# # # # # # # #         for path in paths:
# # # # # # # #             if not os.path.exists(path):
# # # # # # # #                 os.makedirs(path)

# # # # # # # #         # Download and extract dataset
# # # # # # # #         filename = self.download_link.split('/')[-1]
# # # # # # # #         download_and_extract_archive(self.download_link, download_root=os.path.join(self.root, 'FEMNIST', 'raw'), filename=filename, md5=self.file_md5)

# # # # # # # #         # Move extracted files to the processed directory
# # # # # # # #         files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
# # # # # # # #         for file in files:
# # # # # # # #             shutil.move(os.path.join(self.root, 'FEMNIST', 'raw', file), os.path.join(self.root, 'FEMNIST', 'processed', file))

# # # # # # # #         print("FEMNIST dataset downloaded and processed successfully.")
        
        
        
        
        
# # # # # # # from loguru import logger
# # # # # # # import pathlib
# # # # # # # import os
# # # # # # # import shutil
# # # # # # # from typing import Any
# # # # # # # import torch
# # # # # # # from torchvision import transforms
# # # # # # # from torchvision.datasets.utils import download_and_extract_archive
# # # # # # # from torch.utils.data import DataLoader, TensorDataset
# # # # # # # from PIL import Image
# # # # # # # from .dataset import Dataset  # Assuming Dataset is a base class in your codebase


# # # # # # # class FEMNISTDataset(Dataset):
# # # # # # #     def __init__(self, args: Any) -> None:
# # # # # # #         """
# # # # # # #         FEMNIST dataset class for federated learning experiments.
# # # # # # #         Handles downloading, loading, and preprocessing of FEMNIST data.
# # # # # # #         """
# # # # # # #         # Initialize attributes
# # # # # # #         self.root = args.get_data_path()  # Dataset root path
# # # # # # #         self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
# # # # # # #         self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'

# # # # # # #         # Define file paths
# # # # # # #         self.training_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_train.pt')
# # # # # # #         self.test_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_test.pt')
# # # # # # #         self.user_list = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_user_keys.pt')

# # # # # # #         # Ensure dataset files are available (download if necessary)
# # # # # # #         if not self._check_files_exist():
# # # # # # #             self.dataset_download()

# # # # # # #         # Load datasets
# # # # # # #         self.train_dataset = self.load_train_dataset()
# # # # # # #         self.test_dataset = self.load_test_dataset()

# # # # # # #         # Call parent class initializer
# # # # # # #         super(FEMNISTDataset, self).__init__(args)

# # # # # # #     def _check_files_exist(self) -> bool:
# # # # # # #         """
# # # # # # #         Check if the required dataset files exist.
# # # # # # #         """
# # # # # # #         return all(os.path.exists(path) for path in [self.training_file, self.test_file, self.user_list])

# # # # # # #     def dataset_download(self) -> None:
# # # # # # #         """
# # # # # # #         Download and extract the FEMNIST dataset if not already present.
# # # # # # #         """
# # # # # # #         # Create required directories
# # # # # # #         paths = [os.path.join(self.root, 'FEMNIST', 'raw'), os.path.join(self.root, 'FEMNIST', 'processed')]
# # # # # # #         for path in paths:
# # # # # # #             os.makedirs(path, exist_ok=True)

# # # # # # #         # Download and extract dataset
# # # # # # #         raw_dir = os.path.join(self.root, 'FEMNIST', 'raw')
# # # # # # #         filename = self.download_link.split('/')[-1]
# # # # # # #         try:
# # # # # # #             logger.info("Downloading FEMNIST dataset...")
# # # # # # #             download_and_extract_archive(
# # # # # # #                 self.download_link, download_root=raw_dir, filename=filename, md5=self.file_md5
# # # # # # #             )
# # # # # # #             logger.info("Download and extraction completed.")
# # # # # # #         except Exception as e:
# # # # # # #             raise RuntimeError(f"Failed to download or extract FEMNIST dataset: {e}")

# # # # # # #         # Move extracted files to the processed directory
# # # # # # #         files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
# # # # # # #         for file in files:
# # # # # # #             src = os.path.join(raw_dir, file)
# # # # # # #             dest = os.path.join(self.root, 'FEMNIST', 'processed', file)
# # # # # # #             if os.path.exists(src):
# # # # # # #                 shutil.move(src, dest)
# # # # # # #             else:
# # # # # # #                 raise RuntimeError(f"Expected file {file} not found in raw directory after extraction.")

# # # # # # #         logger.info("FEMNIST dataset downloaded and processed successfully.")

# # # # # # #     def load_train_dataset(self):
# # # # # # #         """
# # # # # # #         Load and preprocess the training dataset.
# # # # # # #         """
# # # # # # #         logger.debug("Loading FEMNIST train data")

# # # # # # #         if not os.path.exists(self.training_file):
# # # # # # #             raise RuntimeError(f"Training data not found at {self.training_file}. Please ensure it is downloaded.")

# # # # # # #         # Load data and targets
# # # # # # #         data_targets_users = torch.load(self.training_file)
# # # # # # #         self.data = torch.Tensor(data_targets_users[0])
# # # # # # #         self.targets = torch.Tensor(data_targets_users[1])
# # # # # # #         self.users = data_targets_users[2]

# # # # # # #         # Create DataLoader
# # # # # # #         transform = transforms.Compose([transforms.ToTensor()])
# # # # # # #         train_dataset = TensorDataset(self.data, self.targets, self.users)
# # # # # # #         train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # # # # # #         logger.debug("Finished loading FEMNIST train data")
# # # # # # #         return self.get_tuple_from_data_loader(train_loader)

# # # # # # #     def load_test_dataset(self):
# # # # # # #         """
# # # # # # #         Load and preprocess the test dataset.
# # # # # # #         """
# # # # # # #         logger.debug("Loading FEMNIST test data")

# # # # # # #         if not os.path.exists(self.test_file):
# # # # # # #             raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

# # # # # # #         # Load data and targets
# # # # # # #         data_targets_users = torch.load(self.test_file)
# # # # # # #         self.data = torch.Tensor(data_targets_users[0])
# # # # # # #         self.targets = torch.Tensor(data_targets_users[1])
# # # # # # #         self.users = data_targets_users[2]

# # # # # # #         # Create DataLoader
# # # # # # #         transform = transforms.Compose([transforms.ToTensor()])
# # # # # # #         test_dataset = TensorDataset(self.data, self.targets, self.users)
# # # # # # #         test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # # # # # #         logger.debug("Finished loading FEMNIST test data")
# # # # # # #         return self.get_tuple_from_data_loader(test_loader)

# # # # # # #     def __getitem__(self, index: int):
# # # # # # #         """
# # # # # # #         Get a single sample from the dataset.
# # # # # # #         """
# # # # # # #         img, target, user = self.data[index], int(self.targets[index]), self.users[index]
# # # # # # #         img = Image.fromarray(img.numpy(), mode='F')
# # # # # # #         return img, target, user



# # # # # # from loguru import logger
# # # # # # import os
# # # # # # import shutil
# # # # # # import torch
# # # # # # from torch.utils.data import DataLoader, TensorDataset
# # # # # # from torchvision import transforms
# # # # # # from torchvision.datasets.utils import download_and_extract_archive
# # # # # # from typing import Any
# # # # # # from PIL import Image

# # # # # # # Base Dataset Class
# # # # # # class Dataset:
# # # # # #     def __init__(self, args: Any) -> None:
# # # # # #         self.args = args
# # # # # #         self.train_dataset = self.load_train_dataset()
# # # # # #         self.test_dataset = self.load_test_dataset()

# # # # # #     def get_args(self):
# # # # # #         return self.args

# # # # # #     def get_tuple_from_data_loader(self, data_loader):
# # # # # #         data, targets = [], []
# # # # # #         for batch in data_loader:
# # # # # #             data.append(batch[0])
# # # # # #             targets.append(batch[1])
# # # # # #         return torch.cat(data), torch.cat(targets)

# # # # # # # FEMNISTDataset Class
# # # # # # class FEMNISTDataset(Dataset):
# # # # # #     def __init__(self, args: Any) -> None:
# # # # # #         # Initialize attributes before calling parent class
# # # # # #         self.root = args.get_data_path()  # Dataset root path
# # # # # #         self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
# # # # # #         self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'

# # # # # #         # Define file paths
# # # # # #         self.training_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_train.pt')
# # # # # #         self.test_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_test.pt')
# # # # # #         self.user_list = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_user_keys.pt')

# # # # # #         # Ensure dataset is downloaded
# # # # # #         self.dataset_download()

# # # # # #         # Call the parent class's __init__
# # # # # #         super(FEMNISTDataset, self).__init__(args)

# # # # # #     def load_train_dataset(self):
# # # # # #         """
# # # # # #         Load and preprocess the training dataset.
# # # # # #         """
# # # # # #         logger.debug("Loading FEMNIST train data")

# # # # # #         if not os.path.exists(self.training_file):
# # # # # #             raise RuntimeError(f"Training data not found at {self.training_file}. Please ensure it is downloaded.")

# # # # # #         # Load data and targets
# # # # # #         data_targets_users = torch.load(self.training_file)
# # # # # #         self.data = torch.Tensor(data_targets_users[0])
# # # # # #         self.targets = torch.Tensor(data_targets_users[1])
# # # # # #         self.users = data_targets_users[2]

# # # # # #         # Ensure all tensors are consistent in size
# # # # # #         logger.debug(f"Shapes - Data: {self.data.shape}, Targets: {self.targets.shape}, Users: {len(self.users)}")

# # # # # #         min_size = min(self.data.size(0), self.targets.size(0), len(self.users))
# # # # # #         if self.data.size(0) != min_size or self.targets.size(0) != min_size or len(self.users) != min_size:
# # # # # #             logger.warning(f"Size mismatch detected. Truncating tensors to min size: {min_size}")
# # # # # #             self.data = self.data[:min_size]
# # # # # #             self.targets = self.targets[:min_size]
# # # # # #             self.users = self.users[:min_size]

# # # # # #         # Create DataLoader
# # # # # #         transform = transforms.Compose([transforms.ToTensor()])
# # # # # #         train_dataset = TensorDataset(self.data, self.targets, torch.tensor(self.users))
# # # # # #         train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # # # # #         logger.debug("Finished loading FEMNIST train data")
# # # # # #         return self.get_tuple_from_data_loader(train_loader)

# # # # # #     def load_test_dataset(self):
# # # # # #         """
# # # # # #         Load and preprocess the test dataset.
# # # # # #         """
# # # # # #         logger.debug("Loading FEMNIST test data")

# # # # # #         if not os.path.exists(self.test_file):
# # # # # #             raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

# # # # # #         # Load data and targets
# # # # # #         data_targets_users = torch.load(self.test_file)
# # # # # #         self.data = torch.Tensor(data_targets_users[0])
# # # # # #         self.targets = torch.Tensor(data_targets_users[1])
# # # # # #         self.users = data_targets_users[2]

# # # # # #         # Ensure all tensors are consistent in size
# # # # # #         transform = transforms.Compose([transforms.ToTensor()])
# # # # # #         test_dataset = TensorDataset(self.data, self.targets, torch.tensor(self.users))
# # # # # #         test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # # # # #         logger.debug("Finished loading FEMNIST test data")
# # # # # #         return self.get_tuple_from_data_loader(test_loader)

# # # # # #     def __getitem__(self, index: int):
# # # # # #         """
# # # # # #         Get a specific sample by index.
# # # # # #         """
# # # # # #         img, target, user = self.data[index], int(self.targets[index]), self.users[index]
# # # # # #         img = Image.fromarray(img.numpy(), mode='F')
# # # # # #         return img, target, user

# # # # # #     def dataset_download(self):
# # # # # #         """
# # # # # #         Download and extract the dataset if it's not already available.
# # # # # #         """
# # # # # #         logger.info("Downloading FEMNIST dataset...")

# # # # # #         # Create required directories
# # # # # #         paths = [os.path.join(self.root, 'FEMNIST', 'raw'), os.path.join(self.root, 'FEMNIST', 'processed')]
# # # # # #         for path in paths:
# # # # # #             if not os.path.exists(path):
# # # # # #                 os.makedirs(path)

# # # # # #         # Download and extract dataset
# # # # # #         filename = self.download_link.split('/')[-1]
# # # # # #         download_and_extract_archive(self.download_link, download_root=os.path.join(self.root, 'FEMNIST', 'raw'), filename=filename, md5=self.file_md5)

# # # # # #         # Move extracted files to the processed directory
# # # # # #         files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
# # # # # #         for file in files:
# # # # # #             shutil.move(os.path.join(self.root, 'FEMNIST', 'raw', file), os.path.join(self.root, 'FEMNIST', 'processed', file))

# # # # # #         logger.info("FEMNIST dataset downloaded and processed successfully.")


# # # # # from loguru import logger
# # # # # import pathlib
# # # # # import os
# # # # # import shutil
# # # # # import torch
# # # # # from torch.utils.data import Dataset, DataLoader, TensorDataset
# # # # # from torchvision import transforms
# # # # # from torchvision.datasets.utils import download_and_extract_archive
# # # # # from PIL import Image
# # # # # from typing import Any

# # # # # class FEMNISTDataset(Dataset):
# # # # #     def __init__(self, args: Any) -> None:
# # # # #         """
# # # # #         Initialize the FEMNIST dataset.
# # # # #         """
# # # # #         # Initialize attributes
# # # # #         self.root = args.get_data_path()  # Dataset root path
# # # # #         self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
# # # # #         self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'

# # # # #         # Define file paths
# # # # #         self.training_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_train.pt')
# # # # #         self.test_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_test.pt')
# # # # #         self.user_list = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_user_keys.pt')

# # # # #         # Ensure dataset files are available (download if necessary)
# # # # #         self.dataset_download()

# # # # #         # Load train dataset
# # # # #         self.train_dataset = self.load_train_dataset()

# # # # #         # Call the parent class's __init__
# # # # #         super(FEMNISTDataset, self).__init__()

# # # # #     def load_train_dataset(self):
# # # # #         """
# # # # #         Load and preprocess the training dataset.
# # # # #         """
# # # # #         logger.debug("Loading FEMNIST train data")

# # # # #         if not os.path.exists(self.training_file):
# # # # #             raise RuntimeError(f"Training data not found at {self.training_file}. Please ensure it is downloaded.")

# # # # #         # Load data and targets
# # # # #         data_targets_users = torch.load(self.training_file)
# # # # #         self.data = torch.Tensor(data_targets_users[0])
# # # # #         self.targets = torch.Tensor(data_targets_users[1])
# # # # #         self.users = data_targets_users[2]

# # # # #         # Ensure all tensors are consistent in size
# # # # #         logger.debug(f"Shapes - Data: {self.data.shape}, Targets: {self.targets.shape}, Users: {len(self.users)}")

# # # # #         min_size = min(self.data.size(0), self.targets.size(0), len(self.users))
# # # # #         if self.data.size(0) != min_size or self.targets.size(0) != min_size or len(self.users) != min_size:
# # # # #             logger.warning(f"Size mismatch detected. Truncating tensors to min size: {min_size}")
# # # # #             self.data = self.data[:min_size]
# # # # #             self.targets = self.targets[:min_size]
# # # # #             self.users = self.users[:min_size]

# # # # #         # Convert users to numerical indices
# # # # #         user_to_index = {user: idx for idx, user in enumerate(set(self.users))}
# # # # #         numeric_users = torch.tensor([user_to_index[user] for user in self.users])

# # # # #         # Create DataLoader
# # # # #         train_dataset = TensorDataset(self.data, self.targets, numeric_users)
# # # # #         train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # # # #         logger.debug("Finished loading FEMNIST train data")
# # # # #         return self.get_tuple_from_data_loader(train_loader)

# # # # #     def load_test_dataset(self):
# # # # #         """
# # # # #         Load and preprocess the test dataset.
# # # # #         """
# # # # #         logger.debug("Loading FEMNIST test data")

# # # # #         if not os.path.exists(self.test_file):
# # # # #             raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

# # # # #         data_targets_users = torch.load(self.test_file)
# # # # #         self.data = torch.Tensor(data_targets_users[0])
# # # # #         self.targets = torch.Tensor(data_targets_users[1])
# # # # #         self.users = data_targets_users[2]

# # # # #         transform = transforms.Compose([transforms.ToTensor()])
# # # # #         test_dataset = TensorDataset(self.data, self.targets, self.users)
# # # # #         test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # # # #         logger.debug("Finished loading FEMNIST test data")
# # # # #         return self.get_tuple_from_data_loader(test_loader)

# # # # #     def dataset_download(self):
# # # # #         """
# # # # #         Download and extract the FEMNIST dataset.
# # # # #         """
# # # # #         logger.info("Downloading FEMNIST dataset...")

# # # # #         # Create required directories
# # # # #         paths = [os.path.join(self.root, 'FEMNIST', 'raw'), os.path.join(self.root, 'FEMNIST', 'processed')]
# # # # #         for path in paths:
# # # # #             if not os.path.exists(path):
# # # # #                 os.makedirs(path)

# # # # #         # Download and extract dataset
# # # # #         filename = self.download_link.split('/')[-1]
# # # # #         download_and_extract_archive(self.download_link, download_root=os.path.join(self.root, 'FEMNIST', 'raw'), filename=filename, md5=self.file_md5)

# # # # #         # Move extracted files to the processed directory
# # # # #         files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
# # # # #         for file in files:
# # # # #             shutil.move(os.path.join(self.root, 'FEMNIST', 'raw', file), os.path.join(self.root, 'FEMNIST', 'processed', file))

# # # # #         logger.info("FEMNIST dataset downloaded and processed successfully.")

# # # # #     def get_tuple_from_data_loader(self, loader: DataLoader):
# # # # #         """
# # # # #         Convert DataLoader to tuple of (data, targets, users).
# # # # #         """
# # # # #         data, targets, users = [], [], []
# # # # #         for batch in loader:
# # # # #             data.append(batch[0])
# # # # #             targets.append(batch[1])
# # # # #             users.append(batch[2])

# # # # #         return torch.cat(data), torch.cat(targets), torch.cat(users)

# # # # #     def __getitem__(self, index: int):
# # # # #         """
# # # # #         Get a single data sample.
# # # # #         """
# # # # #         img, target, user = self.data[index], int(self.targets[index]), self.users[index]
# # # # #         img = Image.fromarray(img.numpy(), mode='F')
# # # # #         return img, target, user


# # # # from loguru import logger
# # # # import os
# # # # import shutil
# # # # import torch
# # # # from torch.utils.data import Dataset, DataLoader, TensorDataset
# # # # from torchvision import transforms
# # # # from torchvision.datasets.utils import download_and_extract_archive
# # # # from PIL import Image
# # # # from typing import Any

# # # # class FEMNISTDataset(Dataset):
# # # #     def __init__(self, args: Any) -> None:
# # # #         """
# # # #         Initialize the FEMNIST dataset.
# # # #         """
# # # #         # Initialize attributes
# # # #         self.root = args.get_data_path()  # Dataset root path
# # # #         self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
# # # #         self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'

# # # #         # Define file paths
# # # #         self.training_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_train.pt')
# # # #         self.test_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_test.pt')
# # # #         self.user_list = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_user_keys.pt')

# # # #         # Ensure dataset files are available (download if necessary)
# # # #         self.dataset_download()

# # # #         # Load train dataset
# # # #         self.train_dataset = self.load_train_dataset()

# # # #         # Call the parent class's __init__
# # # #         super(FEMNISTDataset, self).__init__()

# # # #     def load_train_dataset(self):
# # # #         """
# # # #         Load and preprocess the training dataset.
# # # #         """
# # # #         logger.debug("Loading FEMNIST train data")

# # # #         if not os.path.exists(self.training_file):
# # # #             raise RuntimeError(f"Training data not found at {self.training_file}. Please ensure it is downloaded.")

# # # #         # Load data and targets
# # # #         data_targets_users = torch.load(self.training_file)
# # # #         self.data = torch.Tensor(data_targets_users[0])
# # # #         self.targets = torch.Tensor(data_targets_users[1])
# # # #         self.users = data_targets_users[2]

# # # #         # Ensure all tensors are consistent in size
# # # #         logger.debug(f"Shapes - Data: {self.data.shape}, Targets: {self.targets.shape}, Users: {len(self.users)}")

# # # #         min_size = min(self.data.size(0), self.targets.size(0), len(self.users))
# # # #         if self.data.size(0) != min_size or self.targets.size(0) != min_size or len(self.users) != min_size:
# # # #             logger.warning(f"Size mismatch detected. Truncating tensors to min size: {min_size}")
# # # #             self.data = self.data[:min_size]
# # # #             self.targets = self.targets[:min_size]
# # # #             self.users = self.users[:min_size]

# # # #         # Convert users to numerical indices
# # # #         user_to_index = {user: idx for idx, user in enumerate(set(self.users))}
# # # #         numeric_users = torch.tensor([user_to_index[user] for user in self.users])

# # # #         # Create DataLoader
# # # #         train_dataset = TensorDataset(self.data, self.targets, numeric_users)
# # # #         train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # # #         logger.debug("Finished loading FEMNIST train data")
# # # #         return self.get_tuple_from_data_loader(train_loader)

# # # #     def load_test_dataset(self):
# # # #         """
# # # #         Load and preprocess the test dataset.
# # # #         """
# # # #         logger.debug("Loading FEMNIST test data")

# # # #         if not os.path.exists(self.test_file):
# # # #             raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

# # # #         data_targets_users = torch.load(self.test_file)
# # # #         self.data = torch.Tensor(data_targets_users[0])
# # # #         self.targets = torch.Tensor(data_targets_users[1])
# # # #         self.users = data_targets_users[2]

# # # #         transform = transforms.Compose([transforms.ToTensor()])
# # # #         test_dataset = TensorDataset(self.data, self.targets, self.users)
# # # #         test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # # #         logger.debug("Finished loading FEMNIST test data")
# # # #         return self.get_tuple_from_data_loader(test_loader)

# # # #     def dataset_download(self):
# # # #         """
# # # #         Download and extract the FEMNIST dataset.
# # # #         """
# # # #         logger.info("Downloading FEMNIST dataset...")

# # # #         # Create required directories
# # # #         paths = [os.path.join(self.root, 'FEMNIST', 'raw'), os.path.join(self.root, 'FEMNIST', 'processed')]
# # # #         for path in paths:
# # # #             if not os.path.exists(path):
# # # #                 os.makedirs(path)

# # # #         # Download and extract dataset
# # # #         filename = self.download_link.split('/')[-1]
# # # #         download_and_extract_archive(self.download_link, download_root=os.path.join(self.root, 'FEMNIST', 'raw'), filename=filename, md5=self.file_md5)

# # # #         # Move extracted files to the processed directory
# # # #         files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
# # # #         for file in files:
# # # #             shutil.move(os.path.join(self.root, 'FEMNIST', 'raw', file), os.path.join(self.root, 'FEMNIST', 'processed', file))

# # # #         logger.info("FEMNIST dataset downloaded and processed successfully.")

# # # #     def get_tuple_from_data_loader(self, loader: DataLoader):
# # # #         """
# # # #         Convert DataLoader to tuple of (data, targets, users).
# # # #         """
# # # #         data, targets, users = [], [], []
# # # #         for batch in loader:
# # # #             data.append(batch[0])
# # # #             targets.append(batch[1])
# # # #             users.append(batch[2])

# # # #         return torch.cat(data), torch.cat(targets), torch.cat(users)

# # # #     def get_train_dataset(self):
# # # #         """
# # # #         Return the train dataset. This method wraps around load_train_dataset() to provide the expected interface.
# # # #         """
# # # #         return self.train_dataset

# # # #     def __getitem__(self, index: int):
# # # #         """
# # # #         Get a single data sample.
# # # #         """
# # # #         img, target, user = self.data[index], int(self.targets[index]), self.users[index]
# # # #         img = Image.fromarray(img.numpy(), mode='F')
# # # #         return img, target, user



# # from loguru import logger
# # import os
# # import shutil
# # import torch
# # from torch.utils.data import Dataset, DataLoader, TensorDataset
# # from torchvision import transforms
# # from torchvision.datasets.utils import download_and_extract_archive
# # from PIL import Image
# # from typing import Any

# # class FEMNISTDataset(Dataset):
# #     def __init__(self, args: Any) -> None:
# #         """
# #         Initialize the FEMNIST dataset.
# #         """
# #         # Initialize attributes
# #         self.root = args.get_data_path()  # Dataset root path
# #         self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
# #         self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'

# #         # Define file paths
# #         self.training_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_train.pt')
# #         self.test_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_test.pt')
# #         self.user_list = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_user_keys.pt')

# #         # Ensure dataset files are available (download if necessary)
# #         self.dataset_download()

# #         # Load train dataset
# #         self.train_dataset = self.load_train_dataset()

# #         # Call the parent class's __init__
# #         super(FEMNISTDataset, self).__init__()

# #     def load_train_dataset(self):
# #         """
# #         Load and preprocess the training dataset.
# #         """
# #         logger.debug("Loading FEMNIST train data")

# #         if not os.path.exists(self.training_file):
# #             raise RuntimeError(f"Training data not found at {self.training_file}. Please ensure it is downloaded.")

# #         # Load data and targets
# #         data_targets_users = torch.load(self.training_file)
# #         self.data = torch.Tensor(data_targets_users[0])
# #         self.targets = torch.Tensor(data_targets_users[1])
# #         self.users = data_targets_users[2]

# #         # Ensure all tensors are consistent in size
# #         logger.debug(f"Shapes - Data: {self.data.shape}, Targets: {self.targets.shape}, Users: {len(self.users)}")

# #         min_size = min(self.data.size(0), self.targets.size(0), len(self.users))
# #         if self.data.size(0) != min_size or self.targets.size(0) != min_size or len(self.users) != min_size:
# #             logger.warning(f"Size mismatch detected. Truncating tensors to min size: {min_size}")
# #             self.data = self.data[:min_size]
# #             self.targets = self.targets[:min_size]
# #             self.users = self.users[:min_size]

# #         # Convert users to numerical indices
# #         user_to_index = {user: idx for idx, user in enumerate(set(self.users))}
# #         numeric_users = torch.tensor([user_to_index[user] for user in self.users])

# #         # Create DataLoader
# #         train_dataset = TensorDataset(self.data, self.targets, numeric_users)
# #         train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# #         logger.debug("Finished loading FEMNIST train data")
# #         return self.get_tuple_from_data_loader(train_loader)

# #     def load_test_dataset(self):
# #         """
# #         Load and preprocess the test dataset.
# #         """
# #         logger.debug("Loading FEMNIST test data")

# #         if not os.path.exists(self.test_file):
# #             raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

# #         data_targets_users = torch.load(self.test_file)
# #         self.data = torch.Tensor(data_targets_users[0])
# #         self.targets = torch.Tensor(data_targets_users[1])
# #         self.users = data_targets_users[2]

# #         transform = transforms.Compose([transforms.ToTensor()])
# #         test_dataset = TensorDataset(self.data, self.targets, self.users)
# #         test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# #         logger.debug("Finished loading FEMNIST test data")
# #         return self.get_tuple_from_data_loader(test_loader)

# #     def dataset_download(self):
# #         """
# #         Download and extract the FEMNIST dataset.
# #         """
# #         logger.info("Downloading FEMNIST dataset...")

# #         # Create required directories
# #         paths = [os.path.join(self.root, 'FEMNIST', 'raw'), os.path.join(self.root, 'FEMNIST', 'processed')]
# #         for path in paths:
# #             if not os.path.exists(path):
# #                 os.makedirs(path)

# #         # Download and extract dataset
# #         filename = self.download_link.split('/')[-1]
# #         download_and_extract_archive(self.download_link, download_root=os.path.join(self.root, 'FEMNIST', 'raw'), filename=filename, md5=self.file_md5)

# #         # Move extracted files to the processed directory
# #         files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
# #         for file in files:
# #             shutil.move(os.path.join(self.root, 'FEMNIST', 'raw', file), os.path.join(self.root, 'FEMNIST', 'processed', file))

# #         logger.info("FEMNIST dataset downloaded and processed successfully.")

# #     def get_tuple_from_data_loader(self, loader: DataLoader):
# #         """
# #         Convert DataLoader to tuple of (data, targets, users).
# #         """
# #         data, targets, users = [], [], []
# #         for batch in loader:
# #             data.append(batch[0])
# #             targets.append(batch[1])
# #             users.append(batch[2])

# #         return torch.cat(data), torch.cat(targets), torch.cat(users)

# #     def get_train_dataset(self):
# #         """
# #         Return the train dataset. This method wraps around load_train_dataset() to provide the expected interface.
# #         """
# #         return self.train_dataset

# #     def get_data_loader_from_data(self, batch_size: int, X: torch.Tensor, Y: torch.Tensor):
# #         """
# #         Return a DataLoader from the provided data (X) and targets (Y) tensors.
# #         """
# #         dataset = TensorDataset(X, Y)
# #         return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# #     def __getitem__(self, index: int):
# #         """
# #         Get a single data sample.
# #         """
# #         img, target, user = self.data[index], int(self.targets[index]), self.users[index]
# #         img = Image.fromarray(img.numpy(), mode='F')
# #         return img, target, user

# from loguru import logger
# import os
# import shutil
# import torch
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torchvision import transforms
# from torchvision.datasets.utils import download_and_extract_archive
# from PIL import Image
# from typing import Any

# class FEMNISTDataset(Dataset):
#     def __init__(self, args: Any) -> None:
#         """
#         Initialize the FEMNIST dataset.
#         """
#         # Initialize attributes
#         self.root = args.get_data_path()  # Dataset root path
#         self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
#         self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'

#         # Define file paths
#         self.training_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_train.pt')
#         self.test_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_test.pt')
#         self.user_list = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_user_keys.pt')

#         # Ensure dataset files are available (download if necessary)
#         self.dataset_download()

#         # Load train dataset
#         self.train_dataset = self.load_train_dataset()

#         # Call the parent class's __init__
#         super(FEMNISTDataset, self).__init__()

#     def load_train_dataset(self):
#         """
#         Load and preprocess the training dataset.
#         """
#         logger.debug("Loading FEMNIST train data")

#         if not os.path.exists(self.training_file):
#             raise RuntimeError(f"Training data not found at {self.training_file}. Please ensure it is downloaded.")

#         # Load data and targets
#         data_targets_users = torch.load(self.training_file)
#         self.data = torch.Tensor(data_targets_users[0])
#         self.targets = torch.Tensor(data_targets_users[1])
#         self.users = data_targets_users[2]

#         # Ensure all tensors are consistent in size
#         logger.debug(f"Shapes - Data: {self.data.shape}, Targets: {self.targets.shape}, Users: {len(self.users)}")

#         min_size = min(self.data.size(0), self.targets.size(0), len(self.users))
#         if self.data.size(0) != min_size or self.targets.size(0) != min_size or len(self.users) != min_size:
#             logger.warning(f"Size mismatch detected. Truncating tensors to min size: {min_size}")
#             self.data = self.data[:min_size]
#             self.targets = self.targets[:min_size]
#             self.users = self.users[:min_size]

#         # Convert users to numerical indices
#         user_to_index = {user: idx for idx, user in enumerate(set(self.users))}
#         numeric_users = torch.tensor([user_to_index[user] for user in self.users])

#         # Create DataLoader
#         train_dataset = TensorDataset(self.data, self.targets, numeric_users)
#         train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#         logger.debug("Finished loading FEMNIST train data")
#         return self.get_tuple_from_data_loader(train_loader)

#     def load_test_dataset(self):
#         """
#         Load and preprocess the test dataset.
#         """
#         logger.debug("Loading FEMNIST test data")

#         if not os.path.exists(self.test_file):
#             raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

#         data_targets_users = torch.load(self.test_file)
#         self.data = torch.Tensor(data_targets_users[0])
#         self.targets = torch.Tensor(data_targets_users[1])
#         self.users = data_targets_users[2]

#         transform = transforms.Compose([transforms.ToTensor()])
#         test_dataset = TensorDataset(self.data, self.targets, self.users)
#         test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#         logger.debug("Finished loading FEMNIST test data")
#         return self.get_tuple_from_data_loader(test_loader)

#     def dataset_download(self):
#         """
#         Download and extract the FEMNIST dataset.
#         """
#         logger.info("Downloading FEMNIST dataset...")

#         # Create required directories
#         paths = [os.path.join(self.root, 'FEMNIST', 'raw'), os.path.join(self.root, 'FEMNIST', 'processed')]
#         for path in paths:
#             if not os.path.exists(path):
#                 os.makedirs(path)

#         # Download and extract dataset
#         filename = self.download_link.split('/')[-1]
#         download_and_extract_archive(self.download_link, download_root=os.path.join(self.root, 'FEMNIST', 'raw'), filename=filename, md5=self.file_md5)

#         # Move extracted files to the processed directory
#         files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
#         for file in files:
#             shutil.move(os.path.join(self.root, 'FEMNIST', 'raw', file), os.path.join(self.root, 'FEMNIST', 'processed', file))

#         logger.info("FEMNIST dataset downloaded and processed successfully.")

#     def get_tuple_from_data_loader(self, loader: DataLoader):
#         """
#         Convert DataLoader to tuple of (data, targets, users).
#         """
#         data, targets, users = [], [], []
#         for batch in loader:
#             data.append(batch[0])
#             targets.append(batch[1])
#             users.append(batch[2])

#         return torch.cat(data), torch.cat(targets), torch.cat(users)

#     def get_train_dataset(self):
#         """
#         Return the train dataset. This method wraps around load_train_dataset() to provide the expected interface.
#         """
#         return self.train_dataset

#     # def get_data_loader_from_data(self, batch_size: int, X: torch.Tensor, Y: torch.Tensor):
#     #     """
#     #     Return a DataLoader from the provided data (X) and targets (Y) tensors.
#     #     """
#     #     # Debugging output to check the shape of X and Y
#     #     logger.debug(f"X size: {X.shape}, Y size: {Y.shape}")
#     #     dataset = TensorDataset(X, Y)
#     #     return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # def get_data_loader_from_data(self, batch_size: int, X: torch.Tensor, Y: torch.Tensor):
#     #     logger.debug(f"X size: {X.shape}, Y size: {Y.shape}")
#     #     # Get the minimum size (assuming you want to keep as much data as possible)
#     #     min_size = min(X.size(0), Y.size(0))
#     #     # Truncate tensors to the minimum size
#     #     X = X[:min_size]
#     #     Y = Y[:min_size]
#     #     dataset = TensorDataset(X, Y)
#     #     return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     def get_data_loader_from_data(self, batch_size: int, X: torch.Tensor, Y: torch.Tensor):
#         logger.debug(f"X size: {X.shape}, Y size: {Y.shape}")
#         # Get the minimum size (assuming you want to keep as much data as possible)
#         min_size = min(X.shape[0], Y.shape[0])  # Use shape[0] instead of size(0)
#         # Truncate tensors to the minimum size
#         X = X[:min_size]
#         Y = Y[:min_size]
#         dataset = TensorDataset(X, Y)
#         return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    

#     def __getitem__(self, index: int):
#         """
#         Get a single data sample.
#         """
#         img, target, user = self.data[index], int(self.targets[index]), self.users[index]
#         img = Image.fromarray(img.numpy(), mode='F')
#         return img, target, user

from loguru import logger
import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
from typing import Any

class FEMNISTDataset(Dataset):
    def __init__(self, args: Any) -> None:
        """
        Initialize the FEMNIST dataset.
        """
        # Initialize attributes
        self.root = args.get_data_path()  # Dataset root path
        self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
        self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'

        # Define file paths
        self.training_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_train.pt')
        self.test_file = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_test.pt')
        self.user_list = os.path.join(self.root, 'FEMNIST', 'processed', 'femnist_user_keys.pt')

        # Ensure dataset files are available (download if necessary)
        self.dataset_download()

        # Load train dataset
        self.train_dataset = self.load_train_dataset()

        # Call the parent class's __init__
        super(FEMNISTDataset, self).__init__()

    def load_train_dataset(self):
        """
        Load and preprocess the training dataset.
        """
        logger.debug("Loading FEMNIST train data")

        if not os.path.exists(self.training_file):
            raise RuntimeError(f"Training data not found at {self.training_file}. Please ensure it is downloaded.")

        # Load data and targets
        data_targets_users = torch.load(self.training_file)
        self.data = torch.Tensor(data_targets_users[0])
        self.targets = torch.Tensor(data_targets_users[1])
        self.users = data_targets_users[2]

        # Ensure all tensors are consistent in size
        logger.debug(f"Shapes - Data: {self.data.shape}, Targets: {self.targets.shape}, Users: {len(self.users)}")

        min_size = min(self.data.size(0), self.targets.size(0), len(self.users))
        if self.data.size(0) != min_size or self.targets.size(0) != min_size or len(self.users) != min_size:
            logger.warning(f"Size mismatch detected. Truncating tensors to min size: {min_size}")
            self.data = self.data[:min_size]
            self.targets = self.targets[:min_size]
            self.users = self.users[:min_size]

        # Convert users to numerical indices
        user_to_index = {user: idx for idx, user in enumerate(set(self.users))}
        numeric_users = torch.tensor([user_to_index[user] for user in self.users])

        # Create DataLoader
        train_dataset = TensorDataset(self.data, self.targets, numeric_users)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        logger.debug("Finished loading FEMNIST train data")
        return self.get_tuple_from_data_loader(train_loader)

    # def load_test_dataset(self):
    #     """
    #     Load and preprocess the test dataset.
    #     """
    #     logger.debug("Loading FEMNIST test data")

    #     if not os.path.exists(self.test_file):
    #         raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

    #     data_targets_users = torch.load(self.test_file)
    #     self.data = torch.Tensor(data_targets_users[0])
    #     self.targets = torch.Tensor(data_targets_users[1])
    #     self.users = data_targets_users[2]

    #     transform = transforms.Compose([transforms.ToTensor()])
    #     test_dataset = TensorDataset(self.data, self.targets, self.users)
    #     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    #     logger.debug("Finished loading FEMNIST test data")
    #     return self.get_tuple_from_data_loader(test_loader)
    
    # def load_test_dataset(self):
    #     """
    #     Load and preprocess the test dataset.
    #     """
    #     logger.debug("Loading FEMNIST test data")

    #     if not os.path.exists(self.test_file):
    #         raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

    #     data_targets_users = torch.load(self.test_file)
    #     self.data = torch.Tensor(data_targets_users[0])  # Ensure this is a tensor
    #     self.targets = torch.Tensor(data_targets_users[1])  # Ensure this is a tensor
    #     self.users = torch.tensor(data_targets_users[2])  # Ensure users are a tensor

    #     transform = transforms.Compose([transforms.ToTensor()])
    #     test_dataset = TensorDataset(self.data, self.targets, self.users)
    #     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    #     logger.debug("Finished loading FEMNIST test data")
    #     return self.get_tuple_from_data_loader(test_loader)
    
    def load_test_dataset(self):
        """
        Load and preprocess the test dataset.
        """
        logger.debug("Loading FEMNIST test data")

        if not os.path.exists(self.test_file):
            raise RuntimeError(f"Test data not found at {self.test_file}. Please ensure it is downloaded.")

        # Load data and targets
        data_targets_users = torch.load(self.test_file)
        self.data = torch.Tensor(data_targets_users[0])
        self.targets = torch.Tensor(data_targets_users[1])
        users = data_targets_users[2]

        # Ensure all tensors are consistent in size
        logger.debug(f"Shapes - Data: {self.data.shape}, Targets: {self.targets.shape}, Users: {len(users)}")

        min_size = min(self.data.size(0), self.targets.size(0), len(users))
        if self.data.size(0) != min_size or self.targets.size(0) != min_size or len(users) != min_size:
            logger.warning(f"Size mismatch detected. Truncating tensors to min size: {min_size}")
            self.data = self.data[:min_size]
            self.targets = self.targets[:min_size]
            users = users[:min_size]  # Truncate users list

        # Convert users (strings) to numerical indices (same way as train dataset)
        user_to_index = {user: idx for idx, user in enumerate(set(users))}
        self.users = torch.tensor([user_to_index[user] for user in users])

        # Create the TensorDataset
        test_dataset = TensorDataset(self.data, self.targets, self.users)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        logger.debug("Finished loading FEMNIST test data")
        return self.get_tuple_from_data_loader(test_loader)



    def dataset_download(self):
        """
        Download and extract the FEMNIST dataset.
        """
        logger.info("Downloading FEMNIST dataset...")

        # Create required directories
        paths = [os.path.join(self.root, 'FEMNIST', 'raw'), os.path.join(self.root, 'FEMNIST', 'processed')]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # Download and extract dataset
        filename = self.download_link.split('/')[-1]
        download_and_extract_archive(self.download_link, download_root=os.path.join(self.root, 'FEMNIST', 'raw'), filename=filename, md5=self.file_md5)

        # Move extracted files to the processed directory
        files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
        for file in files:
            shutil.move(os.path.join(self.root, 'FEMNIST', 'raw', file), os.path.join(self.root, 'FEMNIST', 'processed', file))

        logger.info("FEMNIST dataset downloaded and processed successfully.")

    def get_tuple_from_data_loader(self, loader: DataLoader):
        """
        Convert DataLoader to tuple of (data, targets, users).
        """
        data, targets, users = [], [], []
        for batch in loader:
            data.append(batch[0])
            targets.append(batch[1])
            users.append(batch[2])

        return torch.cat(data), torch.cat(targets), torch.cat(users)

    def get_train_dataset(self):
        """
        Return the train dataset. This method wraps around load_train_dataset() to provide the expected interface.
        """
        return self.train_dataset
    
    def get_test_dataset(self):
        """
        Return the test dataset.
        """
        return self.load_test_dataset()

    
    
    def get_data_loader_from_data(self, batch_size: int, X, Y):
        """
        Return a DataLoader from the provided data (X) and targets (Y) tensors.
        """
        logger.debug(f"Type of X: {type(X)}, Type of Y: {type(Y)}")
    
        # Convert X and Y to tensors if they are not already tensors
        if not isinstance(X, torch.Tensor):
            logger.warning("X is not a torch.Tensor. Attempting to convert.")
            X = torch.tensor(X)

        if not isinstance(Y, torch.Tensor):
            logger.warning("Y is not a torch.Tensor. Attempting to convert.")
            Y = torch.tensor(Y)

        logger.debug(f"Converted X shape: {X.shape}, Converted Y shape: {Y.shape}")

        # Ensure tensors have the same size
        min_size = min(X.shape[0], Y.shape[0])
        X = X[:min_size]
        Y = Y[:min_size]

        dataset = TensorDataset(X, Y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    # def get_data_loader_from_data(self, batch_size: int, X: torch.Tensor, Y: torch.Tensor):
    #     """
    #     Return a DataLoader from the provided data (X) and targets (Y) tensors.
    #     """
    #     # Log the type and shape of inputs for debugging
    #     logger.debug(f"Type of X: {type(X)}, Type of Y: {type(Y)}")
    #     logger.debug(f"X shape: {getattr(X, 'shape', None)}, Y shape: {getattr(Y, 'shape', None)}")
    
    #     # Ensure inputs are valid tensors
    #     if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
    #         raise ValueError("Both X and Y must be torch.Tensor objects.")

    #     # Use `.shape` instead of `.size()`
    #     min_size = min(X.shape[0], Y.shape[0])  # Determine the minimum size
    #     X = X[:min_size]  # Truncate tensors to the minimum size
    #     Y = Y[:min_size]

    #     # Create TensorDataset and DataLoader
    #     dataset = TensorDataset(X, Y)
    #     return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
            

    # def get_data_loader_from_data(self, batch_size: int, X: torch.Tensor, Y: torch.Tensor):
    #     """
    #     Return a DataLoader from the provided data (X) and targets (Y) tensors.
    #     """
    #     logger.debug(f"X size: {X.shape}, Y size: {Y.shape}")
    #     min_size = min(X.size(0), Y.size(0))  # Determine the minimum size
    #     X = X[:min_size]  # Truncate tensors to the minimum size
    #     Y = Y[:min_size]
    #     dataset = TensorDataset(X, Y)
    #     return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    

    def __getitem__(self, index: int):
        """
        Get a single data sample.
        """
        img, target, user = self.data[index], int(self.targets[index]), self.users[index]
        img = Image.fromarray(img.numpy(), mode='F')
        return img, target, user





