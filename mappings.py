import json
import numpy as np

class Mappings:
    def __init__(self, json_path):
        """

        """
        with open(json_path) as json_file:
            self.data = json.load(json_file)

    def all_captionID(self):
        """Return list of all caption IDs"""

        return [i["id"] for i in self.data["annotations"]]

    def all_captions(self):
        """Return list of all captions"""

        return [i["caption"] for i in self.data["annotations"]]

    def all_imageID(self):
        """Returns list of all image IDs"""

        return [i["id"] for i in self.data["images"]]
                          
    def get_imageID(self, captionID):
        """Return Image ID corresponding to given caption ID
        
        Parameters:
        -----------
        captionID: int
        
        Returns:
        --------
        imageID: int
                 Will return 0 if caption ID is not found
        """

        cap_dict = next((i for i in self.data["annotations"] if self.data["annotations"][i]["id"] == captionID), None)

        if cap_dict is None:
            return 0
        else:
            return cap_dict["image_id"]


    def get_captionIDs(self, imageID):
        """Return caption IDs corresponding to given image ID
        
        Parameters:
        -----------
        imageID: int
        
        Returns:
        --------
        captionIDs: List[int]
                    Will return [0] if image ID is not found
        """

        caption_dicts = next((i for i in self.data["annotations"] if self.data["annotations"][i]["image_id"] == imageID), None)
        
        pass

    def get_captions(self, imageID):
        """Return captions corresponding to given image ID

        Parameters:
        -----------
        imageID: int

        Returns:
        --------
        captions: List[str]
        """
        pass

    def get_imageURL(self, imageID):
        """Returns image URLs for given image ID

        Parameters:
        -----------
        imageID: int

        Returns:
        --------
        URLs: List[str]
        """
        pass

    def get_cap_vector(self, captionID):
        """Returns unit vector given caption ID

        Parameters:
        -----------
        captionID: int

        Returns: 
        --------
        unit_vector: np.array(50,)
        """
        pass
    
