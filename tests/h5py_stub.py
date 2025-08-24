import sys
import types
import pickle
import numpy as np

class _FakeDataset:
    def __init__(self, array):
        self._array = np.array(array)

    def __getitem__(self, key):
        return self._array[key]


class _FakeFile:
    def __init__(self, path, mode='r'):
        self.path = path
        self.mode = mode
        if 'r' in mode and not path is None:
            try:
                with open(path, 'rb') as fh:
                    self._data = pickle.load(fh)
            except FileNotFoundError:
                self._data = {}
        else:
            self._data = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if 'w' in self.mode or 'a' in self.mode:
            with open(self.path, 'wb') as fh:
                pickle.dump(self._data, fh)

    def create_dataset(self, name, data):
        self._data[name] = np.array(data)

    def __getitem__(self, key):
        return _FakeDataset(self._data[key])

    def __contains__(self, key):
        return key in self._data

h5py = types.SimpleNamespace(File=_FakeFile)
# Ensure modules importing ``h5py`` receive this stub
sys.modules.setdefault('h5py', h5py)
