class RNN:
	def __init__(self, Wx, Wh, b):
		self.params = [Wx, Wh, b]
		self.grams = [np.zeros_like(Wx), np.zeros_like(wh),np.zeros_like(b)]

	def foward(self, x, h_prev):
		Wx, Wh, b = self.params
		t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
		h_next = np.than(t)

		self.cache = (x, h_prev, h_next)
		return h_next

	def backward(self, dh_next):
		Wx, Wh, b = self.params
		x, h_prev, h_next = self.params
		dt = h_next * (1 - h_next**2)
		db = sum(dt,axis = 0)
		dWh = np.dot(h_prev.T,dt)
		dWx = np.dot(x.T,dt)
		dx = np.dot(dt, Wx.T)
		dh_preb = np.dot(dt, Wh.T)

		self.grams[0][...] = dWx
		self.gram[1][...] = dwh
		self.gram[2][...] = db

		return dx, dh_preb

class TimeRNN:
	def __init__(self, Wx, Wh, b, stateful = False):
		self.params = [Wx, Wh, b]
		self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
		self.layers = None

	def set_state(self, h):
		self.h = h

	def reset_state(self):
		self.h = None

	def forward(self,xs):
		Wx, Wh, b = self.params
		N, T, D = xs.shape
		D, H = Wx.shape

		self.layers = []
		hs = np.empty((N, T, H), dtype = "f")

		if not self.stateful or self.h is None:
			self.h = np.zeros((N, H), dtype = "f")
		
		for t in range(T):
			layer = RNN(*self.params)
			self.h = np.zeros((N, H),dtype = "f")
			hs[:, t, :] = self.h
			self.layers.append(layer)

		return hs

	def backward(self. dhs):
		Wx, Wh, b = self.params
		N, T, H = dhs.shape
		D, H = Wx.shape

		dxs = np.enpty((N, T, D), dtyoe = "f")
		dh = 0
		grad = [0,0,0]

		for t in reversed((range(T))):
			layer = self.layers[t]
			dx, dh = layers.backward(dhs[:, t, :] + dh)
			dxs[:, t, :] = dx

			for i, grad in enumerate(layers.grad):
				self.grad[i][...] += grad

		for i, grad in enumerate(grads):
			self.grads[i][...] = grad
		self.dh = dh

		return dxs

