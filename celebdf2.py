
#%%
import csv
import torch
from torchvision.io.video import read_video
from torchvision.datasets import VisionDataset
from typing import Any, Callable, List, Optional, Tuple

#%%
class CelebDF2(VisionDataset):
    '''Celeb-DF v2 dataset. Inherits VisionDataset.'''
    def __init__(
            self,
            root: str,
            max_frames=int,
            n_frames = int,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.max_frames = max_frames  # Fixed number of frames to clip from each video
        self.n_frames = n_frames
        set_name = 'Celeb-DF-v2'
        self._data_path = root + '/' + set_name # eg data/Celeb-DF-v2

        # read path, verify, throw exception if nonexistent or other problem
        with open(self._data_path + '/List_of_testing_videos.txt',
                  encoding='utf-8') as data:
            reader = csv.reader(data, delimiter=' ')
            # read into list of lists. 0: fake, 1: real. eg
            # [['1', 'YouTube-real/00170.mp4'], ...]
            self._data_info = list(reader) # list[list[str, str]]

    def __len__(self) -> int:
        ''' return total number of clips '''
        num_clips = len(self._data_info)
        return num_clips

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        ''' read, return item at index after applying transform.
        returns tuple of (video, audio, class label) with format TCHW aka time,
        channels, height, width '''
        # target class and file path eg 1, YouTube-real/00170.mp4
        label, filepath = self._data_info[index][0], self._data_info[index][1]
        video, audio, vid_info = read_video(self._data_path + '/' + filepath,
                                            output_format='TCHW', pts_unit='sec') # time, channels, height, width
        class_index = int(label) # get label index

        # Clip frames to the fixed number `max_frames`
        video = self._clip_frames(video) 

        # sample n number of frames from clipped video
        video = self._sample_frames(video)

        if self.transform is not None:
            video = self.transform(video)

        video = self._convertBCHWtoCBHW(video)

        # audio omitted for our purposes
        return video, class_index
    
    def _clip_frames(self, frames):
        """Clip the video frames to a fixed number of frames `max_frames`."""
        # If the video has more frames than max_frames, clip it to the first `max_frames`
        if len(frames) > self.max_frames:
            return frames[:self.max_frames]
        
        # If the video has fewer frames than `max_frames`, pad with zeros
        pad_size = self.max_frames - len(frames)
        pad_frames = torch.zeros((pad_size, *frames.shape[1:]))  # Create padding frames with same size
        frames = torch.cat([frames, pad_frames], dim=0)
        
        return frames
    
    def _sample_frames(self, frames):
        """Sample n number of frames from cliped video"""
        if len(frames) > self.n_frames:
            indices = torch.linspace(0, len(frames)-1, steps=self.n_frames).long()
            frames = frames[indices]  
        
        return frames
    
    def _convertBCHWtoCBHW(self, vid: torch.Tensor) -> torch.Tensor:
        """Convert tensor from (B, C, H, W) to (C, B, H, W)"""
        return vid.permute(1, 0, 2, 3)
