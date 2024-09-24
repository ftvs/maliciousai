
#%%
import csv
from torchvision.io.video import read_video
from torchvision.datasets import VisionDataset
from typing import Any, Callable, List, Optional, Tuple

#%%
class CelebDF2(VisionDataset):
    '''Celeb-DF v2 dataset. Inherits VisionDataset. TODO use the transforms'''
    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(str, transforms, transform, target_transform)
        self.root = root
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

    def __getitem__(self, index: int):
        # read, return item at index
        # target class and file path eg 1, YouTube-real/00170.mp4
        label, filepath = self._data_info[index][0], self._data_info[index][1]
        video, audio, vid_info = read_video(self._data_path + '/' + filepath,
                                            output_format='TCHW') # time, channels, height, width
        class_index = int(label) # get label index
        return video, audio, class_index
