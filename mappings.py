import json
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

class Mappings:
    def __init__(self, json_path = r"./dat/captions_train2014.json"):
        """Set of convenience functions to pull information from dataset easily

        Parameters:
        -----------
        json_path (Optional): str
                              Path to json file containing dataset information
                              Default: "./dat/captions_train2014.json"
        """
        with open(json_path) as json_file:
            self.data = json.load(json_file)

        self.captions = [i["caption"] for i in self.data["annotations"]]
        self.captionID = [i["id"] for i in self.data["annotations"]]
        self.imgID = [i["id"] for i in self.data["images"]]
        self.capID2img = {self.data["annotations"][i]["id"] : self.data["annotations"][i]["image_id"] for i, _ in enumerate(self.data["annotations"])}
        self.id2caption = {self.data["annotations"][i]["id"] : self.data["annotations"][i]["caption"] for i, _ in enumerate(self.data["annotations"])}
        self.imgURL = {self.data["images"][i]["id"] : self.data["images"][i]["coco_url"] for i, _ in enumerate(self.data["images"])}

    # def all_captionID(self):
    #     """Return list of all caption IDs"""

    #     return [i["id"] for i in self.data["annotations"]]

    # def all_captions(self):
    #     """Return list of all captions"""

    #     return [i["caption"] for i in self.data["annotations"]]

    # def all_imageID(self):
    #     """Returns list of all image IDs"""

    #     return [i["id"] for i in self.data["images"]]
                          
    # def get_imageID_capID(self, captionID):
    #     """Return Image ID corresponding to given caption ID
        
    #     Parameters:
    #     -----------
    #     captionID: int
        
    #     Returns:
    #     --------
    #     imageID: int  
    #              Will return 0 if caption ID is not found
    #     """

    #     cap_dict = next((dic for i, dic in enumerate(self.data["annotations"]) if self.data["annotations"][i]["id"] == captionID), None)
    #     return cap_dict["image_id"] if cap_dict is not None else 0


    # def get_captionIDs_imgID(self, imageID):
    #     """Return caption IDs corresponding to given image ID
        
    #     Parameters:
    #     -----------
    #     imageID: int
        
    #     Returns:
    #     --------
    #     captionIDs: List[int]
    #                 Will return [0] if image ID is not found
    #     """

    #     caption_dicts = [dic for i, dic in enumerate(self.data["annotations"]) if self.data["annotations"][i]["image_id"] == imageID]
    #     return [j["id"] for j in caption_dicts] if caption_dicts is not None else [0]

    # def get_captions_imgID(self, imageID):
    #     """Return captions corresponding to given image ID

    #     Parameters:
    #     -----------
    #     imageID: int

    #     Returns:
    #     --------
    #     captions: List[str]
    #     """
    #     caption_dicts = [dic for i, dic in enumerate(self.data["annotations"]) if self.data["annotations"][i]["image_id"] == imageID]
    #     return [j["caption"] for j in caption_dicts] if caption_dicts is not None else [0]

    # def get_imageURL(self, imageID):
    #     """Returns image URLs for given image ID

    #     Parameters:
    #     -----------
    #     imageID: int

    #     Returns:
    #     --------
    #     URLs: str
    #           Returns 'None' if image ID is not found
    #     """
    #     img_dict = next((dic for i, dic in enumerate(self.data["images"]) if self.data["images"][i]["id"] == imageID), None)
    #     return img_dict["coco_url"] if img_dict is not None else 'None'

    # def get_caption_capID(self, captionID):
    #     """Returns caption for given caption ID
        
    #     Parameters:
    #     -----------
    #     captionID: int

    #     Returns:
    #     --------
    #     caption: str

    #     """
    #     caption_dict = next((dic for i, dic in enumerate(self.data["annotations"]) if self.data["annotations"][i]["id"] == captionID), None)
    #     return caption_dict["caption"] if caption_dict is not None else 'None'

    def get_capID_vector(self, captionID, glove, caption_tokens):
        import text_embedding
        """Returns unit vector given caption ID

        Parameters:
        -----------
        captionID: int
        glove_path: loaded glove file
        caption_tokens: all captions tokenized

        Returns: 
        --------
        unit_vector: np.array(50,)
        """
        caption = self.id2caption[captionID]
        return text_embedding.text_embed(caption, glove, self.captions, caption_tokens)       


# # Testing
# dataset = Mappings()
# print(dataset.capID2img)