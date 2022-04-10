import pickle
import os
import shutil
from datetime import datetime
from src import utils
import pandas as pd

class FileOperations:
    def __init__(self,model_path) -> None:
        self.model_directory=model_path

    def save_model(self, model, filename):
        """
        Saves the model file to directory
        """
        try:
            path = os.path.join(
                self.model_directory, filename
            )  # create seperate directory for each cluster
            if os.path.isdir(
                path
            ):  # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)  #
            with open(path + "/" + filename + ".sav", "wb") as f:
                pickle.dump(model, f)  # save the model to file
            #self.logger.info("Model File " + filename + " saved.")
            return "success"
        except Exception as e:
            #self.logger.error("Exception while saving the model" + str(e))
            raise Exception()
