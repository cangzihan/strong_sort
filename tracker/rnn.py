import numpy as np
import scipy
import matplotlib.pylab as plt
from .kalman_filter import KalmanFilter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RNN(KalmanFilter):
    def __init__(self, img_size, model_path):
        super().__init__()
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size

    def initiate(self, measurement):
      mean_pos = measurement
      mean_vel = np.zeros_like(mean_pos)
      mean = measurement
        
      return mean

    def rnn_predict(self, x_pred, return_seq=False):
      time_step = len(x_pred)
      x_pred = np.array(x_pred)
      x_pred = np.reshape(x_pred, (1, time_step, 4))

      if return_seq:
        y_pred = self.model.predict(x_pred)
      else:
        y_pred = self.model.predict(x_pred)[0]

      return np.array([y_pred[0], y_pred[1], y_pred[2], y_pred[3]])

    def update(self, mean, measurement):
        # Normalize
        if self.img_size == '1080p':
          x_pred = [[mean[0]/1920,mean[1]/1080,mean[2],mean[3]/1080],
          [measurement[0]/1920,measurement[1]/1080,measurement[2],measurement[3]/1080]]
        elif self.img_size == 'VGA':
          x_pred = [[mean[0]/640,mean[1]/480,mean[2],mean[3]/480],
          [measurement[0]/640,measurement[1]/480,measurement[2],measurement[3]/480]]
        
        # Predict
        rnn_pred = self.rnn_predict(x_pred)
        
        # Normalize
        if self.img_size == '1080p':
          if rnn_pred[3] < 0:
            rnn_pred[3] = 0.001
          new_bbox = np.array([rnn_pred[0]*1920,rnn_pred[1]*1080,rnn_pred[2],rnn_pred[3]*1080])
        elif self.img_size == 'VGA':
          if rnn_pred[3] < 0:
            rnn_pred[3] = 0.001
          new_bbox = np.array([rnn_pred[0]*640,rnn_pred[1]*480,rnn_pred[2],rnn_pred[3]*480])
        #print("IIIU",mean)
        #print("OOOU",new_bbox)
        #new_bbox = mean
        return new_bbox

    def predict(self, mean):
      #print(mean)
      return mean
      # Normalize
      if self.img_size == '1080p':
        x_pred = [[mean[0]/1920,mean[1]/1080,mean[2],mean[3]/1080]]
      elif self.img_size == 'VGA':
        x_pred = [[mean[0]/640,mean[1]/480,mean[2],mean[3]/480]]
      
      # Predict
      rnn_pred = self.rnn_predict(x_pred)
      
      # Normalize
      if self.img_size == '1080p':
        if rnn_pred[3] < 0:
          rnn_pred[3] = 0.001
        new_bbox = np.array([rnn_pred[0]*1920,rnn_pred[1]*1080,rnn_pred[2],rnn_pred[3]*1080])
      elif self.img_size == 'VGA':
        if rnn_pred[3] < 0:
          rnn_pred[3] = 0.001
        new_bbox = np.array([rnn_pred[0]*640,rnn_pred[1]*480,rnn_pred[2],rnn_pred[3]*480])
      #print(new_bbox)
      return new_bbox


    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
      d = measurements - mean
      d = d.T
      squared_euro = np.sum(d * d, axis=0)
      #print("###",squared_euro)
      #print("CCC",mean)
      return squared_euro

if __name__ == "__main__":
    pass
