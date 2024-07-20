import numpy as np
from cosdistance import find_cos_dist
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from facenet_models import FacenetModel

model = FacenetModel()
cosDist = []

def descriptorMatch(database, descriptor: np.ndarray, *, threshold=0.475):
    vectorDist = []
    possiblePeople = []
    #finding cosine distances

    for i in range(len(list(database.data.keys()))):
        meanDescriptor = np.mean(list(database.data.values())[i].descriptors, axis=0)
        meanDescriptor = meanDescriptor.reshape(1, meanDescriptor.shape[0])
        dist = find_cos_dist(descriptor, meanDescriptor)
        possiblePeople.append(list(database.data.keys())[i])
        vectorDist.append(dist)
    # {"key": "values", "key1":"value1"}
    """for i, d in enumerate(list(database.data.values()[0]).descriptors):
        dist = find_cos_dist(descriptor, np.mean(d, axis=0))
        possiblePeople.append(list(database.data.keys())[0])
        vectorDist.append(dist)"""
    #position of least possible cosine distance
    pos = np.argmin(vectorDist)
    cosDist.append(vectorDist[pos])
    if vectorDist[pos] < threshold:
        #database.add() from Profile class
        return possiblePeople[pos] #to be used in the final output graph with name
    return 'No match found'

def findThreshold():
    #print(np.array(cosDist).shape)
    cDist = np.array(cosDist).reshape(1, np.array(cosDist).shape[0])
    #speeds_np[speeds_np>0].mean()
    #return np.mean(cosDist, axis=1)
    return cDist[cDist < 0.5].mean(), cDist[cDist > 0.5].mean()

def displayFinalPicture(array_pic, names):
    boxes, probabilities, landmarks = model.detect(array_pic)
    fig, ax = plt.subplots()
    ax.imshow(array_pic)

    i = 0
    for box, prob, _ in zip(boxes, probabilities, landmarks):
        # draw the box on the screen
        names[i] = names[i].replace('_', ' ')
        if prob > 0.99:
            ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))
            plt.text(box[:2][0], box[:2][1], names[i], color="red", backgroundcolor="white", fontsize="x-small")
        i+=1
    plt.show()