import numpy as np

# training
class DataInput:
  def __init__(self, data, batch_size, k):
    self.k = k
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    self.i += 1

    u, r_emb, i, y, sl, new_sl, c = [], [], [], [], [], [], []
    for t in ts:
      u.append(t[0])
      r_emb.append(t[6])
      i.append(t[7])
      y.append(t[8])
      c.append(t[9])
      sl.append(min(len(t[2]), self.k))
      new_sl.append(len(t[4]))
    max_new_sl = max(new_sl)

    hist_i = np.zeros([len(ts), self.k], np.int64)
    hist_t = np.zeros([len(ts), self.k], np.float32)
    hist_i_new = np.zeros([len(ts), max_new_sl], np.int64)
    hist_r_emb = np.zeros([len(ts), self.k, 384], np.float32)
    hist_r_new_emb = np.zeros([len(ts), max_new_sl, 384], np.float32)

    kk = 0
    for t in ts:
      length = len(t[2])
      if length > self.k:
        for l in range(self.k):
          hist_i[kk][l] = t[2][length-self.k+l]
          vec = t[1][length - self.k + l]
          hist_r_emb[kk][l] = vec if isinstance(vec, (list, np.ndarray)) else np.zeros(384, dtype=np.float32)
          hist_t[kk][l] = t[5][length-self.k+l]
      else:
        for l in range(length):
          hist_i[kk][l] = t[2][l]
          vec = t[1][l]
          hist_r_emb[kk][l] = vec if isinstance(vec, (list, np.ndarray)) else np.zeros(384, dtype=np.float32)
          hist_t[kk][l] = t[5][l]
      for l in range(len(t[4])):
        hist_i_new[kk][l] = t[4][l]
        vec = t[3][l]
        hist_r_new_emb[kk][l] = vec if isinstance(vec, (list, np.ndarray)) else np.zeros(384, dtype=np.float32)
      kk += 1

    return self.i, (u, r_emb, i, y, hist_r_emb, hist_i, hist_r_new_emb, hist_i_new, hist_t, sl, new_sl, c)


# testing
class DataInputTest:
  def __init__(self, data, batch_size, k):
    self.k = k
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    self.i += 1

    u, ri_emb, i, rj_emb, j, sl, new_sl, c = [], [], [], [], [], [], [], []
    for t in ts:
      u.append(t[0])
      ri_emb.append(t[6][0])
      i.append(t[7][0])
      rj_emb.append(t[6][1])
      j.append(t[7][1])
      c.append(t[8])
      sl.append(min(len(t[2]), self.k))
      new_sl.append(len(t[4]))
    max_new_sl = max(new_sl)

    hist_i = np.zeros([len(ts), self.k], np.int64)
    hist_t = np.zeros([len(ts), self.k], np.float32)
    hist_i_new = np.zeros([len(ts), max_new_sl], np.int64)
    hist_r_emb = np.zeros([len(ts), self.k, 384], np.float32)
    hist_r_new_emb = np.zeros([len(ts), max_new_sl, 384], np.float32)

    kk = 0
    for t in ts:
      length = len(t[2])
      if length > self.k:
        for l in range(self.k):
          hist_i[kk][l] = t[2][length-self.k+l]
          vec = t[1][length - self.k + l]
          hist_r_emb[kk][l] = vec if isinstance(vec, (list, np.ndarray)) else np.zeros(384, dtype=np.float32)
          hist_t[kk][l] = t[5][length-self.k+l]
      else:
        for l in range(length):
          hist_i[kk][l] = t[2][l]
          vec = t[1][l]
          hist_r_emb[kk][l] = vec if isinstance(vec, (list, np.ndarray)) else np.zeros(384, dtype=np.float32)
          hist_t[kk][l] = t[5][l]
      for l in range(len(t[4])):
        hist_i_new[kk][l] = t[4][l]
        vec = t[3][l]
        hist_r_new_emb[kk][l] = vec if isinstance(vec, (list, np.ndarray)) else np.zeros(384, dtype=np.float32)
      kk += 1

    return self.i, (u, ri_emb, i, rj_emb, j, hist_r_emb, hist_i, hist_r_new_emb, hist_i_new, hist_t, sl, new_sl, c)
