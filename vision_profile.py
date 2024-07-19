import pickle

class Profile:
    def __init__(self, name):
        self.name = name
        self.descriptors = []

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)

    def remove_descriptor(self, descriptor):
        self.descriptors.remove(descriptor)

    def get_descriptors(self):
        return self.descriptors

class FaceDatabase:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {}

    def create_profile(self, name):
        if name not in self.data:
            self.data[name] = Profile(name)

    def remove_profile(self, name):
        if name in self.data:
            del self.data[name]

    def add_image(self, name, image):
        if name not in self.data:
            self.create_profile(name)
        self.data[name].add_descriptor(image)

    def remove_image(self, name, image):
        self.data[name].remove_descriptor(image)

    def get_profile(self, name):
        return self.data[name]

    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
