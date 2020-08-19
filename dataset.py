from torch.utils.data import Dataset
import torch
import numpy as np
import os

from simplereacherdimensions import img_res

class NumpyCsvDataSet(Dataset):
    def __init__(self, csv_file, device=None):
        self.csv_file = csv_file
        self.data = torch.as_tensor(np.loadtxt(csv_file, delimiter=',', dtype=np.float32)).to(device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class SimpleReacherTrajectoryDataset(Dataset):
    def __init__(self, root_folder, trajectory_ids, include_images=True, device=None):
        self.root_folder = root_folder
        self.device = device
        self.trajectory_ids = trajectory_ids
        self.include_images = include_images
        if isinstance(self.trajectory_ids, int):
            self.trajectory_ids = range(self.trajectory_ids)

    def __len__(self):
        return len(self.trajectory_ids)

    def _load_array(self, trajectoryId, fileName):
        filePath = os.path.join(self.root_folder, str(trajectoryId), fileName)
        if not os.path.exists(filePath):
            raise Exception('Data file {} not present in root folder'.format(filePath))
        return torch.as_tensor(np.load(filePath)).to(self.device)

    def __getitem__(self, index):
        result = {'states': self._load_array(index, 'trajectory.npy')}
        if self.include_images:
            result['images'] = self._load_array(index, 'images.npy')
        return result

class SimpleReacherBaseDataset(Dataset):
    def __init__(self, root_folder, trajectory_ids, device=None):
        self.root_folder = root_folder
        self.device = device
        self.trajectory_ids = trajectory_ids
        if isinstance(self.trajectory_ids, int):
            self.trajectory_ids = range(self.trajectory_ids)
        self.trajectory_data = None
        self.cumulative_index = torch.zeros(len(self.trajectory_ids))
        idx = 0
        for tId in self.trajectory_ids:
            curr_trajectory = self._load_array(tId, 'trajectory.npy')
            if self.trajectory_data is None:
                self.trajectory_data = curr_trajectory
            else:
                self.trajectory_data = torch.cat((self.trajectory_data, curr_trajectory), axis=0)
            # Allow child classes to process
            self._process_trajectory(tId, curr_trajectory)
            if idx+1 < len(self.cumulative_index):
                self.cumulative_index[idx+1] = self.cumulative_index[idx] + len(curr_trajectory) 
            idx += 1
        self.trajectory_data = torch.as_tensor(self.trajectory_data).to(self.device)

    def __len__(self):
        return len(self.trajectory_data)

    def _load_array(self, trajectoryId, fileName):
        filePath = os.path.join(self.root_folder, str(trajectoryId), fileName)
        if not os.path.exists(filePath):
            raise Exception('Data file {} not present in root folder'.format(filePath))
        return torch.as_tensor(np.load(filePath)).to(self.device)

    def _process_trajectory(self, trajectoryId, stateData):
        pass

    def __getitem__(self, index):
        return {'states': self.trajectory_data[index]}

class SimpleReacherPreLoadedDataset(SimpleReacherBaseDataset):
    def __init__(self, root_folder, trajectory_ids, device=None):
        self.image_data = None
        super().__init__(root_folder, trajectory_ids, device)
        self.image_data = torch.as_tensor(self.image_data).to(self.device)

    def _process_trajectory(self, trajectoryId, stateData):
        curr_img_data = self._load_array(trajectoryId, 'images.npy')
        if self.image_data is None:
            self.image_data = curr_img_data
        else:
            self.image_data = torch.cat((self.image_data, curr_img_data), axis=0)

    def __getitem__(self, index):
        return {'states': self.trajectory_data[index], 'images': self.image_data[index]}


class SimpleReacherOnDemandDataset(SimpleReacherBaseDataset):
    def __init__(self, root_folder, trajectory_ids,  device=None):
        self.trajectory_index = None
        super().__init__(root_folder, trajectory_ids, device)
        self.trajectory_index = torch.as_tensor(self.trajectory_index).to(self.device)

    def _process_trajectory(self, trajectoryId, stateData):
        curr_traj_index = torch.as_tensor([[trajectoryId, k] for k in range(len(stateData))]).to(self.device)
        if self.trajectory_index is None:
            self.trajectory_index = curr_traj_index
        else:
            self.trajectory_index = torch.cat((self.trajectory_index, curr_traj_index), axis=0)

    def __getitem__(self, index):
        if isinstance(index, int):
            filtered_index = self.trajectory_index[index]
            traj_image_data = self._load_array(filtered_index[0].item(), 'images.npy')
            images = traj_image_data[filtered_index[1]]
        else:
            filtered_index = self.trajectory_index[index]
            required_traj_ids = torch.unique(filtered_index[:, 0])
            images = torch.zeros((len(index), img_res * img_res)).to(self.device)
            for tId in required_traj_ids:
                trajectoryId = tId.item()
                traj_image_data = self._load_array(trajectoryId, 'images.npy')
                mask_dest = torch.where(filtered_index[:, 0] == trajectoryId)
                mask_source = filtered_index[mask_dest][:, 1]
                images[mask_dest] = traj_image_data[mask_source]

        return {'states': self.trajectory_data[index], 'images': images}