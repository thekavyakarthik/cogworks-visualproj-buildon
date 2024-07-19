import matplotlib.pyplot as plt
from camera import take_picture
from facenet_models import FacenetModel
from cosdistance import find_cos_dist

def detect_face_prob(img, model):
    """
    Function to return features from image, probability of a face
    Take in generic image and return tuple of information from image
    """
    boxes, probabilities, landmarks = model.detect(img)
    return boxes, probabilities, landmarks

def extract_descriptors(img, boxes, model):
    """
    Function extracting all descriptors from a given image and the boxes drawn on the image
    """
    descriptors = model.compute_descriptors(img, boxes)
    return descriptors




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

def is_true_positive(prob, threshold):
    """
    Determine if the detection probability is a true positive.

    Parameters:
    - prob: The detection probability
    - threshold: The probability threshold for a true positive

    Returns:
    - True if the detection is a true positive, False otherwise
    """
    return prob >= threshold