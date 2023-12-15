import random

class GridWorld:
    def __init__(self, world_size=(20, 20), num_landmarks=8, discrete=True):
        '''
        world_size: size of world (square)
        num_landmarks: number of landmarks
        '''
        self.world_size = world_size
        self.width = world_size[0]
        self.height = world_size[1]
        self.num_landmarks = num_landmarks
        self.discrete = discrete

    def get_landmarks(self, landmarks=None):
        if landmarks is None:
            self.landmarks = []
            for i in range(self.num_landmarks):
                x = random.random() * self.width
                y = random.random() * self.height
                if self.discrete:
                    self.landmarks.append([round(x), round(y), 0])
                else:
                    self.landmarks.append((x, y, 0)) # [x, y, theta], id
        return self.landmarks
    
    def get_world_size(self):
        return self.world_size
    
    def get_num_landmarks(self):
        return len(self.landmarks)
    
    def get_landmark(self, index):
        return self.landmarks[index]

    