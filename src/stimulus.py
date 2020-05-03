from collections import deque
import cv2
from skimage.color import rgb2gray
import torch
from torch.utils.data import IterableDataset, DataLoader


def frameToGrayscaleTensor(frame):
    gray = rgb2gray(frame).astype("float32")
    return gray.reshape(1, *gray.shape)

class VideoStimulus(IterableDataset):
    def __init__(self, path, timesteps, preprocess_frame=frameToGrayscaleTensor):
        # TODO - implement support for multiple workers
        """ VideoStimulus iterates through video and returns spatio-temporal stimulus.

            Args:
                path - path to video
                timesteps - integer number of time steps population responds to
                preprocess_frame - frame preprocess function, default converts to grayscale
        """
        self.path = path
        self.timesteps = timesteps
        self.__stim = deque(maxlen=self.timesteps)
        self.__preprocessFrame = preprocess_frame
        self.__openVideo()

    def __openVideo(self):
        """ Open video referenced by self.path and set basic properties. """
        if hasattr(self, "__vc") and self.__vc.isOpened():
            self.__vc.release()

        self.__vc = cv2.VideoCapture(self.path)
        self.spatial_shape = (int(self.__vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.__vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = int(self.__vc.get(cv2.CAP_PROP_FPS))

    def __next__(self):
        """ Return next stimulus tensor (timesteps x channels(1) x *spatial_shape). """
        if len(self.__stim) < self.timesteps - 1:
            self.__initStim()

        success, frame = self.__vc.read()
        if not success:
            raise StopIteration
        
        self.__appendToStim(frame)
        stim_tensor = torch.tensor(self.__stim)
        return stim_tensor.permute(1,0,2,3) # permute channels and timesteps
        # channels = self.__stim[0].shape[0]
        # return stim_tensor.view(channes, self.timesteps, *self.spatial_shape) # permute channels and timesteps

    def __iter__(self):
        """ Return self object. """
        self.__openVideo()
        return self

    def __len__(self):
        """ Return total stimuli number in video.
            Note: this function relies on cv2 frame counting capabilities.
        """
        return int(self.__vc.get(cv2.CAP_PROP_FRAME_COUNT)) - (self.timesteps-1)

    def __initStim(self):
        """ Initialize self.__stim with first self.timesteps-1 frames. """
        for i in range(self.timesteps-1):
            success, frame = self.__vc.read()
            if not success:
                raise ValueError(f"Video {self.path} has less that {self.timesteps} frames.")
            self.__appendToStim(frame)
    
    def __appendToStim(self, frame):
        """ Append next frame to self.__stim and remove the oldest if needed. """
        self.__stim.append(self.__preprocessFrame(frame))

    def __del__(self):
        self.__vc.release()

def getVideoStimulusLoader(path, timesteps, batch_size=1):
    """ Return data loader and properties of VideoStimulus. """
    vs = VideoStimulus(path, timesteps)
    props = {"spatial_shape": vs.spatial_shape, "fps": vs.fps}
    return DataLoader(vs, batch_size=batch_size, shuffle=False), props

"""
ts = 20
vs = VideoStimulus("../../data/nat_stim_256.mkv", ts)
s = next(vs)
print(len(vs))
"""
