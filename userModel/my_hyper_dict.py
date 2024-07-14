import json
from spotPython.data import base
import pathlib


class MyHyperDict(base.FileConfig):
    """User specified hyperparameter dictionary.

    This class extends the FileConfig class to provide a dictionary for storing hyperparameters.

    Attributes:
        filename (str):
            The name of the file where the hyperparameters are stored.
    """

    def __init__(
        self,
        filename: str = "my_hyper_dict.json",
        directory: None = None,
    ) -> None:
        super().__init__(filename=filename, directory=directory)
        self.filename = filename
        self.directory = directory
        self.hyper_dict = self.load()

    @property
    def path(self):
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    def load(self) -> dict:
        """Load the hyperparameters from the file.

        Returns:
            dict: A dictionary containing the hyperparameters.

        Examples:
            # Assume the user specified file `my_hyper_dict.json` is in the `./hyperdict/` directory.
            >>> user_lhd = MyHyperDict(filename='my_hyper_dict.json', directory='./hyperdict/')
        """
        with open(self.path, "r") as f:
            d = json.load(f)
        return d
