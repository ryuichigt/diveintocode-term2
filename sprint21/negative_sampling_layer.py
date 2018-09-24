class EmbeddingDot:
	def __init__(self, W):
		self.embed = Embedding(W)
		self.params = self.embed.params
		self.grads = self.embed.grads
		self.cache = None

	def forward(self, h, idx):
		target_W = self.embed.forward(idx):
		out = np.sum(target_W, * h, axis = 1)

		self.cache = (h, target_W)
		return out

	def backward(self, dout):
		h, target_W = self.chche
		dout = dout.backward(dtarget_W)
		self.embed.backward(dtarget_W)
		dh = dout * target_W
		return dh