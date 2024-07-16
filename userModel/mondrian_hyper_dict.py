import json
from spotRiver.data import base
import pathlib


class MondrianHyperDict(base.FileConfig):
    """River hyperparameter dictionary."""

    def __init__(
        self,
        filename: str = "mondrian_hyper_dict.json",
        directory: None = None,
    ) -> None:
        super().__init__(filename=filename, directory=directory)
        self.filename = filename
        self.directory = directory
        self.hyper_dict = self.load()
        self.scenario = "river"

    @property
    def path(self):
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    def get_scenario(self):
        return self.scenario

    def load(self) -> dict:
        """Load the hyperparameters from the file.

        Returns:
            dict: A dictionary containing the hyperparameters.

        """
        with open(self.path, "r") as f:
            d = json.load(f)
        return d
