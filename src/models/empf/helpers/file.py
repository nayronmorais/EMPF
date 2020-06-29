"""Module contains functions for management of files."""

import os
from os import path

import msgpack as mpk
import requests
from tqdm import tqdm


_user_dir = path.expanduser('~')

DEFAULT_DATA_DIR_NAME = path.join(_user_dir, 'empf', 'data')
DEFAULT_CHUNK_SIZE = 4096  # 4Kb.


def _init_data_dir(path):
    """Auxiliary method for create the default data dir if not exists."""
    try:
        os.makedirs(path)
    except PermissionError:
        raise Exception("Your user don't have write permission.")

    except FileExistsError:
        pass


def to_local_file(obj, filename, to_save_path=None):
    """
    Save an object as file.

    Params
    ------
    obj : object
        Serialized object or any object which be built-in Python types.

    root_dir : str
        Root directory where the object will be saved.

    filename : str
        Object filename.

    """
    if to_save_path is None:
        _init_data_dir()
        to_save_path = DEFAULT_DATA_DIR_NAME

    fpath = path.join(to_save_path, filename)

    try:
        with open(fpath, mode='wb') as file:
            mpk.dump(obj, file)
    except TypeError:
        raise Exception('`obj` could not be saved because it ' +
                        'is not serialized.')


def load_local_file(filename, to_save_path=None):
    """
    Load a object saved as file to memory.

    Params
    ------

    root_dir : str
        Root directory where the object was saved.

    filename : str
        Object filename.

    Returns
    -------
    object : saved type
        In-memory object reference.

    """
    if to_save_path is None:
        _init_data_dir()
        to_save_path = DEFAULT_DATA_DIR_NAME

    fpath = path.join(to_save_path, filename)

    try:
        obj = None
        with open(fpath, mode='rb') as file:
            obj = mpk.load(file)
        return obj

    except mpk.exceptions.FormatError:
        raise Exception('Incompatible file.')

    except (mpk.exceptions.ExtraData, mpk.exceptions.ValueError):
        raise Exception('The file is corrupted.')


def download_files(url, filenames, to_save_path=None, chunk_size=None):
    """
    Obtain files from remote location.

    Parameters
    ----------
    url : str
        url where is the file.

    filenames : iterator<str>
        The wanted files.

    to_save_path : str, optional
        The path where the files will be saved. The default is None.

    chunk_size : int
        The number of bytes used in the main memory allocated to download
        the files. Larger files will be divided into parts of this length.

        AVOID TO LOAD HUGE FILES INTO MAIN MEMORY.

    """
    if to_save_path is None:
        to_save_path = DEFAULT_DATA_DIR_NAME

    _init_data_dir(to_save_path)

    print(f'\nData path: {to_save_path}')

    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    chunk_size = int(chunk_size)

    for filename in filenames:

        full_path = path.join(to_save_path, filename)
        full_url = path.join(url, filename)

        with open(full_path, mode='wb') as file:

            with requests.get(full_url, stream=True) as response:

                if response.status_code == requests.codes.OK:

                    total_length = response.headers.get('content-length')

                    if total_length is None:
                        file.write(response.content)

                    else:

                        total_length = int(total_length)

                        with tqdm(total=total_length, unit='B') as pbar:  # Show a progress bar.

                            for chunk_data in response.iter_content(chunk_size):
                                file.write(chunk_data)
                                pbar.update(chunk_size)
                else:
                    print('File not found.')
