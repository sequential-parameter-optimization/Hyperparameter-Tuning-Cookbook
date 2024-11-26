import os


def get_data_folder_path():
    """Returns the absolute path to the data folder located in the package."""
    # Assume the 'data' directory is within the same package directory
    current_file_path = os.path.abspath(__file__)
    package_dir = os.path.dirname(current_file_path)
    data_folder_path = os.path.join(package_dir, "data")
    return data_folder_path
