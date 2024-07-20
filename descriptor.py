import numpy as np
from cosdistance import find_cos_dist
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from facenet_models import FacenetModel

model = FacenetModel()

def descriptorMatch(database, descriptor: np.ndarray, threshold: float) -> str:
    vectorDist = []
    possiblePeople = []
    #finding cosine distances

    for i in range(len(list(database.data.keys()))):
        dist = find_cos_dist(descriptor, np.mean(list(database.data.values())[i].descriptors, axis=0))
        possiblePeople.append(list(database.data.keys())[i])
        vectorDist.append(dist)
    # {"key": "values", "key1":"value1"}
    """for i, d in enumerate(list(database.data.values()[0]).descriptors):
        dist = find_cos_dist(descriptor, np.mean(d, axis=0))
        possiblePeople.append(list(database.data.keys())[0])
        vectorDist.append(dist)"""
    #position of least possible cosine distance
    pos = np.argmin(vectorDist)
    print(vectorDist[pos])
    if vectorDist[pos] < threshold:
        #database.add() from Profile class
        return possiblePeople[pos] #to be used in the final output graph with name
    return 'No match found'

def displayFinalPicture(array_pic, name):
    boxes, probabilities, landmarks = model.detect(array_pic)
    fig, ax = plt.subplots()
    ax.imshow(array_pic)

    for box, prob, landmark in zip(boxes, probabilities, landmarks):
        # draw the box on the screen
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))
        plt.text(box[:2][0], box[:2][1], name, color="red")
    plt.show()