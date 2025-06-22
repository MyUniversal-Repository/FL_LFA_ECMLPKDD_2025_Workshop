import torch

def inspect_pt_file(file_path):
    try:
        # Load the .pt file
        data_targets_users = torch.load(file_path)
        
        # Verify that the loaded object is a tuple
        if not isinstance(data_targets_users, tuple) or len(data_targets_users) != 3:
            raise ValueError("The loaded .pt file is not in the expected (data, targets, users) tuple format.")
        
        # Extract data, targets, and users
        data, targets, users = data_targets_users
        
        # Display shapes and statistics
        print(f"Data Shape: {torch.tensor(data).shape}")
        print(f"Targets Shape: {torch.tensor(targets).shape}")
        print(f"Number of Users: {len(users)}")
        
        # Check class distribution in targets
        unique_classes, class_counts = torch.unique(torch.tensor(targets), return_counts=True)
        print(f"Class Distribution: {dict(zip(unique_classes.tolist(), class_counts.tolist()))}")
        
    except Exception as e:
        print(f"Error inspecting file: {e}")

# Replace with the path to your .pt file
file_path = "D:/SGD_TargetedDPA/data/FEMNIST/raw/femnist/femnist_test.pt"
inspect_pt_file(file_path)

