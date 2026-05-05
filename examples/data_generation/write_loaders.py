import pandas as pd
import numpy as np
import concurrent.futures
import itertools

import os

content = r'''import pandas as pd
import numpy as np
import torch
import h5py
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, List, Optional, Any
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# Import helper from utils (assuming it stays there for now)
from TBMD.core.utils.misc import extract_step_number

logger = logging.getLogger(__name__)

# =================================================================================================
# EXISTING CLASSES FROM data/loaders.py (Renamed DataLoader -> BaseDataLoader)
# =================================================================================================

class BaseDataLoader:
    """
    Базовый класс для загрузки данных (Generic Loader).
    
    Examples:
        >>> loader = BaseDataLoader('data.h5')
        >>> tensor = loader.load_tensor()
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            data_path: Путь к файлу данных
            device: Torch device
            dtype: Torch dtype
        """
        self.data_path = Path(data_path)
        self.device = torch.device(device)
        self.dtype = dtype
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")
    
    def load_tensor(
        self,
        key: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Загрузить тензор из файла
        
        Args:
            key: Ключ для HDF5 файлов
            **kwargs: Дополнительные параметры
            
        Returns:
            Загруженный тензор
        """
        suffix = self.data_path.suffix.lower()
        
        if suffix == '.h5' or suffix == '.hdf5':
            return self._load_hdf5(key, **kwargs)
        elif suffix in ['.npy', '.npz']:
            return self._load_numpy(key, **kwargs)
        elif suffix in ['.pt', '.pth']:
            return self._load_pytorch(key, **kwargs)
        else:
            raise ValueError(f"Неподдерживаемый формат: {suffix}")
    
    def _load_hdf5(
        self,
        key: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """Загрузить из HDF5"""
        with h5py.File(self.data_path, 'r') as f:
            if key is None:
                # Взять первый ключ
                key = list(f.keys())[0]
                logger.info(f"Ключ не указан, используется: {key}")
            
            data = f[key][:]
        
        return torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
    
    def _load_numpy(
        self,
        key: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """Загрузить из NumPy"""
        if self.data_path.suffix == '.npz':
            data_dict = np.load(self.data_path)
            if key is None:
                key = list(data_dict.keys())[0]
            data = data_dict[key]
        else:
            data = np.load(self.data_path)
        
        return torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
    
    def _load_pytorch(
        self,
        key: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """Загрузить из PyTorch"""
        data = torch.load(self.data_path, map_location=self.device)
        
        if isinstance(data, dict) and key is not None:
            data = data[key]
        
        return data.to(device=self.device, dtype=self.dtype)
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Загрузить метаданные из файла
        
        Returns:
            Словарь с метаданными
        """
        metadata = {
            'path': str(self.data_path),
            'format': self.data_path.suffix,
            'size_mb': self.data_path.stat().st_size / (1024 * 1024)
        }
        
        suffix = self.data_path.suffix.lower()
        
        if suffix in ['.h5', '.hdf5']:
            with h5py.File(self.data_path, 'r') as f:
                metadata['keys'] = list(f.keys())
                # Размеры первого dataset
                if metadata['keys']:
                    first_key = metadata['keys'][0]
                    metadata['shape'] = f[first_key].shape
                    metadata['dtype'] = str(f[first_key].dtype)
        
        elif suffix == '.npz':
            data = np.load(self.data_path)
            metadata['keys'] = list(data.keys())
        
        return metadata


class HDF5Loader(BaseDataLoader):
    """
    Специализированный загрузчик для HDF5
    
    Поддерживает дополнительные функции для HDF5 файлов
    """
    
    def list_keys(self) -> List[str]:
        """Получить список всех ключей в HDF5 файле"""
        with h5py.File(self.data_path, 'r') as f:
            return list(f.keys())
    
    def load_multiple(
        self,
        keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Загрузить несколько тензоров
        
        Args:
            keys: Список ключей
            
        Returns:
            Словарь {key: tensor}
        """
        result = {}
        with h5py.File(self.data_path, 'r') as f:
            for key in keys:
                if key in f:
                    data = f[key][:]
                    result[key] = torch.from_numpy(data).to(
                        device=self.device,
                        dtype=self.dtype
                    )
        
        return result
    
    def load_slice(
        self,
        key: str,
        slices: tuple
    ) -> torch.Tensor:
        """
        Загрузить срез данных (эффективно для больших файлов)
        
        Args:
            key: Ключ dataset
            slices: Tuple срезов, например (slice(0, 100), slice(0, 50))
            
        Returns:
            Срез данных
        """
        with h5py.File(self.data_path, 'r') as f:
            data = f[key][slices]
        
        return torch.from_numpy(data).to(device=self.device, dtype=self.dtype)


class TensorDataLoader:
    """
    Загрузчик для тензорных данных с pre-processing
    
    Examples:
        >>> loader = TensorDataLoader('data.h5')
        >>> tensor, metadata = loader.load_with_preprocessing(
        ...     normalize=True,
        ...     remove_mean=True
        ... )
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        self.loader = BaseDataLoader(data_path, device, dtype)
    
    def load_with_preprocessing(
        self,
        key: Optional[str] = None,
        normalize: bool = False,
        remove_mean: bool = False,
        remove_trend: bool = False,
        **kwargs
    ) -> tuple:
        """
        Загрузить тензор с предобработкой
        
        Args:
            key: Ключ для HDF5
            normalize: Нормализовать к [0, 1]
            remove_mean: Удалить среднее
            remove_trend: Удалить тренд
            
        Returns:
            (tensor, metadata) где metadata содержит параметры обработки
        """
        tensor = self.loader.load_tensor(key, **kwargs)
        metadata = {}
        
        if remove_mean:
            mean = tensor.mean(dim=-1, keepdim=True)
            tensor = tensor - mean
            metadata['mean'] = mean
        
        if remove_trend:
            # Простое удаление линейного тренда
            T = tensor.shape[-1]
            t = torch.linspace(0, 1, T, device=tensor.device)
            
            # Для каждой временной серии
            original_shape = tensor.shape
            tensor_2d = tensor.reshape(-1, T)
            
            for i in range(tensor_2d.shape[0]):
                # Линейная регрессия
                A = torch.stack([t, torch.ones_like(t)], dim=1)
                coeffs = torch.linalg.lstsq(A, tensor_2d[i]).solution
                trend = A @ coeffs
                tensor_2d[i] -= trend
            
            tensor = tensor_2d.reshape(original_shape)
            metadata['detrended'] = True
        
        if normalize:
            min_val = tensor.min()
            max_val = tensor.max()
            tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
            metadata['min'] = min_val
            metadata['max'] = max_val
        
        return tensor, metadata


# =================================================================================================
# MIGRATED CLASS FROM utils/data_loader.py
# =================================================================================================


class DataLoader:
    """A data loader for tensor-based datasets.

    This class provides methods to load different types of data, including
    static and dynamic tensors from CSV/Excel files, images, HDF5 files, and
    JSON files. It offers a consistent interface for data loading and can
    convert data to PyTorch tensors if needed.
    """
    
    @staticmethod
    def _read_tabular_file(file_path: Path) -> pd.DataFrame:
        """Reads a CSV or Excel file into a pandas DataFrame.

        Args:
            file_path (Path): The path to the file to read.

        Returns:
            pd.DataFrame: The data from the file as a pandas DataFrame.

        Raises:
            ValueError: If the file path is not valid or the file format
                is not supported.
        """
        if not file_path.is_file():
            raise ValueError(f"The provided path '{file_path}' is not a valid file.")

        file_suffix = file_path.suffix.lower()
        if file_suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_suffix in (".xls", ".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_suffix}. Please provide a CSV or Excel file.")

    def load_static_tensor(self, data_path: Union[str, Path], shape: tuple) -> Dict[str, np.ndarray]:
        """Loads static tensor data from CSV or Excel files in a directory.

        This method uses a thread pool to load files in parallel, improving performance
        for large datasets.

        Args:
            data_path (Union[str, Path]): The path to the directory containing
                the data files.
            shape (tuple): The shape to reshape the loaded data into.

        Returns:
            Dict[str, np.ndarray]: A dictionary of tensors, where keys are
            filenames and values are the corresponding numpy arrays.
        """
        data_path = Path(data_path)
        if not data_path.is_dir():
            raise ValueError(f"The provided path '{data_path}' is not a valid directory.")

        files = sorted(itertools.chain(data_path.glob("*.csv"), data_path.glob("*.xls"), data_path.glob("*.xlsx")), key=lambda f: extract_step_number(f.name))
        if not files:
            raise ValueError(f"No CSV or Excel files found in directory: {data_path}")

        # Use ThreadPoolExecutor for parallel file loading to improve performance
        def _load_single_file(file_path):
            try:
                data = self._read_tabular_file(file_path).fillna(0)
                tensor = data.iloc[:, 4:].to_numpy(dtype=np.float32).reshape(shape)
                return file_path.stem, tensor
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                return file_path.stem, None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(_load_single_file, files), total=len(files), desc="Loading static tensor files"))

        tensors_dict = defaultdict(lambda: None)
        for stem, tensor in results:
            if tensor is not None:
                tensors_dict[stem] = tensor

        return tensors_dict
    
    def load_images_tensor(self, dataset_path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Loads images from subject directories into tensors.

        Args:
            dataset_path (Union[str, Path]): The path to the dataset directory,
                which contains subject subdirectories.

        Returns:
            Tuple[Dict[str, np.ndarray], List[str]]: A tuple containing a
            dictionary of image tensors (where keys are subject IDs) and a list
            of subject directory names.
        """
        dataset_path = Path(dataset_path)
        subject_images = defaultdict(lambda: None)
        subject_dir_list = []

        for subject_dir in tqdm(dataset_path.iterdir(), desc="Load images as tensors"):
            if not subject_dir.is_dir():
                continue

            image_files = list(subject_dir.glob("*.png"))
            if not image_files:
                print(f"Warning: Subject directory '{subject_dir.name}' contains no PNG files.")
                continue

            subject_id = subject_dir.name
            subject_dir_list.append(subject_id)

            image_files_sorted = sorted(image_files, key=lambda f: extract_step_number(f.name))

            def load_image(image_file):
                with Image.open(image_file) as img:
                    img = img.convert("RGB")
                    img_array = np.array(img, dtype=np.uint8)
                return img_array


            with concurrent.futures.ThreadPoolExecutor() as executor:
                image_list = list(tqdm(executor.map(load_image, image_files_sorted), total=len(image_files_sorted), desc=f"Loading {subject_id}", leave=False))

            # Stack images first (uint8), then convert to float32 and normalize
            # This is more memory efficient and faster than converting each image individually
            subject_images[subject_id] = np.stack(image_list, axis=-1).astype(np.float32)
            subject_images[subject_id] /= 255.0

        if not subject_dir_list:
            raise ValueError(f"No subjects with PNG images found in the directory: {dataset_path}")

        return subject_images, subject_dir_list
    
    def load_dynamic_tensor(self, directory: Union[str, Path], target_shape: tuple) -> Dict[str, np.ndarray]:
        """Loads and processes dynamic tensor data from CSV or Excel files.

        This function reads each file, reshapes the data into the specified
        target shape, and then checks if the resulting tensor is 4D. If so, it
        splits the tensor into multiple 3D tensors by slicing along the third
        dimension.

        Args:
            directory (Union[str, Path]): The directory containing the CSV or
                Excel files.
            target_shape (tuple): The shape to which each tensor should be
                reshaped.

        Returns:
            Dict[str, np.ndarray]: A dictionary where keys are file stems (or
            file stem with slice index) and values are the corresponding
            tensors.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"The provided path '{directory}' is not a valid directory.")

        # Collect CSV and Excel files
        file_paths = sorted(itertools.chain(directory.glob("*.csv"), directory.glob("*.xls*")), key=lambda f: extract_step_number(f.name))

        if not file_paths:
            raise ValueError(f"No CSV or Excel files found in directory: {directory}")

        def _load_single_file(file_path):
            try:
                df = self._read_tabular_file(file_path).fillna(0)
                tensor = df.iloc[:, 4:].to_numpy(dtype=np.float32).reshape(target_shape)
                return file_path.stem, tensor
            except Exception as err:
                print(f"Error loading file {file_path}: {err}")
                return file_path.stem, None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(_load_single_file, file_paths), total=len(file_paths), desc="Loading dynamic tensor files"))

        loaded_tensors = defaultdict(lambda: None)
        for stem, tensor in results:
            if tensor is not None:
                loaded_tensors[stem] = tensor

        # Post-process: split 4D tensors with shape[2] != 0 into multiple 3D tensors
        processed_tensors = defaultdict(lambda: None)
        for key, tensor in loaded_tensors.items():
            if tensor is not None and tensor.ndim == 4 and tensor.shape[2] != 0:
                # Iterate over the 3rd dimension and extract 3D slices.
                processed_tensors.update(zip(
                    (f"{key}_slice_{i}" for i in range(tensor.shape[2])),
                    tensor.swapaxes(0, 2)
                ))
            else:
                processed_tensors[key] = tensor

        return processed_tensors
    
    def load_data(self, path: Union[str, Path], data_type: str, shape: Optional[tuple] = None, tensor_type: str = "np") -> Any:
        """Loads data of different types.

        This function provides a unified interface to load various data types,
        including 'static', 'images', and 'dynamic'. It can also convert the
        loaded data to PyTorch tensors.

        Args:
            path (Union[str, Path]): The path to the data file or directory.
            data_type (str): The type of data to load ('static', 'images', or
                'dynamic').
            shape (Optional[tuple], optional): The shape for reshaping tensor
                data. Required for 'static' and 'dynamic' data types.
                Defaults to None.
            tensor_type (str, optional): The tensor type to return. Can be 'np'
                for numpy arrays or 'pt' for PyTorch tensors. Defaults to "np".

        Returns:
            Any: The loaded data, which can be a dictionary of numpy arrays or
            PyTorch tensors, depending on the `tensor_type`.
        """
        path = Path(path)
        
        if data_type == 'static':
            if shape is None:
                raise ValueError("Shape must be provided for static tensor data")
            data = self.load_static_tensor(path, shape)
            
        elif data_type == 'images':
            data = self.load_images_tensor(path)
            
        elif data_type == 'dynamic':
            if shape is None:
                raise ValueError("Shape must be provided for dynamic tensor data")
            data = self.load_dynamic_tensor(path, shape)
            
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Convert numpy arrays to PyTorch tensors if requested
        if tensor_type == "pt":
            try:
                import torch
            except ImportError:
                raise ImportError("PyTorch is not installed. Please install it to convert numpy arrays to PyTorch tensors.")
            
            if data_type in ("static", "dynamic", "images"):
                target_dict = data[0] if data_type == "images" else data

                if isinstance(target_dict, dict) and target_dict:
                    # Filter out None values and check for shape consistency
                    valid_items = {k: v for k, v in target_dict.items() if v is not None}

                    if valid_items:
                        # Attempt to batch conversion if all shapes are identical
                        first_val = next(iter(valid_items.values()))
                        first_shape = getattr(first_val, 'shape', None)

                        if len(valid_items) > 1 and all(getattr(v, 'shape', None) == first_shape for v in valid_items.values()):
                            # Batch conversion: stack into a single contiguous array first.
                            # While this adds one copy, it significantly reduces the number of
                            # separate PyTorch tensor creations and potential future device transfers.
                            keys = list(valid_items.keys())
                            stacked = np.stack([valid_items[k] for k in keys])
                            batched_torch = torch.as_tensor(stacked, dtype=torch.float32)

                            for i, key in enumerate(keys):
                                target_dict[key] = batched_torch[i]
                        else:
                            # Fallback to individual conversion
                            for key, value in valid_items.items():
                                target_dict[key] = torch.as_tensor(value, dtype=torch.float32)
        
        return data

    @staticmethod
    def load_h5_tensors(h5_path: Union[str, Path]) -> dict:
        """Loads tensors from an HDF5 file.

        This function loads pressure, soil, and names from an HDF5 file and
        returns them as dictionaries, similar to the notebook example.

        Args:
            h5_path (Union[str, Path]): The path to the HDF5 file.

        Returns:
            dict: A dictionary containing 'all', 'pressure', and 'soil' tensors.
        """
        h5_path = str(h5_path)  # h5py.File expects string path
        temp_tensors_all = {}
        temp_tensors_pressure = {}
        temp_tensors_soil = {}
        with h5py.File(h5_path, "r") as f:
            pressure_loaded = f["pressure"][:]
            soil_loaded = f["soil"][:]
            names_loaded = [n.decode() for n in f["names"][:]]
        for i, name in enumerate(names_loaded):
            temp_tensors_all[name] = np.transpose(
                np.concatenate([
                    pressure_loaded[i][..., None],
                    soil_loaded[i][..., None]
                ], axis=-1), (0, 1, 3, 2)
            )
            temp_tensors_pressure[name] = pressure_loaded[i]
            temp_tensors_soil[name] = soil_loaded[i]
        return {
            'all': temp_tensors_all,
            'pressure': temp_tensors_pressure,
            'soil': temp_tensors_soil
        }

    @staticmethod
    def load_wells_from_json(json_path: Union[str, Path]) -> dict:
        """Loads wells data from a JSON file.

        This is similar to `load_all_wells_from_json`.

        Args:
            json_path (Union[str, Path]): The path to the JSON file.

        Returns:
            dict: The wells data loaded from the JSON file.
        """
        json_path = str(json_path)  # json.load expects string path when using open()
        with open(json_path, 'r') as f:
            return json.load(f)
'''

with open('src/TBMD/core/data/loaders.py', 'w') as f:
    f.write(content)
