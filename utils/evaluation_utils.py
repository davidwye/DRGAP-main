import statistics
class Scores(object):

  def __init__(self):
    self.true_positives = 0
    self.false_positives = 0
    self.true_negatives = 0
    self.false_negatives = 0

  def recall(self):

    numerator = self.true_positives
    denominator = self.true_positives + self.false_negatives
    return 100.0 * numerator / denominator if denominator else 0.0

  def precision(self):

    numerator = self.true_positives
    denominator = self.true_positives + self.false_positives
    return 100.0 * numerator / denominator if denominator else 0.0

  def f1(self):

    recall = self.recall()
    precision = self.precision()

    numerator = 2 * precision * recall
    denominator = precision + recall
    return numerator / denominator if denominator else 0.0

def get_distribution(data):
  mean = statistics.mean(data)
  stdev = statistics.stdev(data)

  return min(data), max(data), mean, stdev

