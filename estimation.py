%matplotlib notebook
import matplotlib.pyplot as plt
from camera import take_picture
import numpy as np
from facenet_models import FacenetModel

model = FacenetModel()

def detect_face_prob(img):
    """
    Function to return features from image, probability of a face
    Take in generic image and return tuple of information from image
    """
    boxes, probabilities, landmarks = model.detect(img)
    return boxes, probabilities, landmarks

def extract_descriptors(img, boxes):
    """
    Function extracting all descriptors from a given image and the boxes drawn on the image
    """
    descriptors = model.compute_descriptors(img, boxes)
    return descriptors

def cosine_distane(descriptors, labels):
    """
    Determine cosine distance for each descriptor and its label
    Append to a list for whether it matches or does not match
    """
    distances = find_cost_dist(descriptors)
    matches = []
    non_matches = []
    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                matches.append(distances[i, j])
            else:
                non_matches.append(distances[i, j])
                
    return matches, non_matches


def plot_histograms(true_probs, false_probs, match_distances, non_match_distances):
    """
    Plot histogram to determine optimal threshold for detection probability and cosine-distances
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(true_probs, bins=50, alpha=0.5, label='True Positives')
    plt.hist(false_probs, bins=50, alpha=0.5, label='False Positives')
    plt.xlabel('Detection Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Detection Probability Distribution')

    plt.subplot(1, 2, 2)
    plt.hist(match_distances, bins=50, alpha=0.5, label='Matches')
    plt.hist(non_match_distances, bins=50, alpha=0.5, label='Non-Matches')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Cosine Distance Distribution')

    plt.tight_layout()
    plt.show()


images = []
labels = []

detections = [detect_faces_and_probabilities(img) for img in images]
descriptors = [extract_face_descriptors(img, boxes) for img, (boxes, _, _) in zip(images, detections)]
descriptors = np.concatenate(descriptors, axis=0)
labels = np.concatenate([[label] * len(boxes) for label, (boxes, _, _) in zip(labels, detections)], axis=0)

match_distances, non_match_distances = cosine_distances(descriptors, labels)
true_positives = [prob for _, probs in detections for prob in probs if is_true_positive(prob, labels)]
false_negatives = [prob for _, probs in detections for prob in probs if not is_true_positive(prob, labels)]

plot_histograms(true_positives, false_negatives, match_distances, non_match_distances)