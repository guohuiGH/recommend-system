import tensorflow as tf
from tensorflow.layers import Layer, Dense
from tensorflow import keras
from tensorflow.keras.backend import repeat_elements, sum
class MLP(Layer):
	def __init__(self):
		"""
			ts多任务样例的改写
			init里定义layer及变量
		"""
		super(MLP, self).__init__()

		self.ctr_dense1 = Dense(500, activation='relu')
		self.ctr_dense2 = Dense(200, activation='relu')
		self.ctr_dense3 = Dense(100, activation='relu')
		self.ctr_out = Dense(1)

		self.cvr_out = Dense(1)

		self.flatten = tf.layers.Flatten()
	
	def build(self, input_shape):
		"""
			变量的size定义，也可以在__init__完成，tf.layers.Layer需要继承改函数
		"""
		pass

	def call(self, inputx):
		"""
			inputx:
				x: s, l , m, dense四个维度向量的拼接
				show_index: 展示pos位置向量
				st: sparse table
			return:
				ctr_logits: ctr的预测值
				cvr_logits: cvr的预测值
			网络结构：
				step1: input->500-200->100  变量共享
				step2: step1 + show_index 共122维 变量共享
				step3: ctr: 122->1->sigmoid
				step4: cvr: 122->1->sigmoid
		"""
		x, show_index, st = inputx 
		#ctr_tmp = self.ctr_net(x)
		t_d1 = self.ctr_dense1(x)
		t_d2 = self.ctr_dense2(t_d1)
		t_d3 = self.ctr_dense3(t_d2)
		#cur_in = tf.concat([t_d3, show_index], axis=-1)
		cur_in = t_d3 
		ctr_value = self.ctr_out(cur_in)
		ctr = self.flatten(ctr_value)
		ctr_logits = tf.sigmoid(ctr)

		cvr_value = self.cvr_out(cur_in)
		cvr = self.flatten(cvr_value)
		cvr_logits = tf.sigmoid(cvr)

		#ctcvr = tf.multiply(ctr_logits, cvr_logits)
		return [ctr_logits, cvr_logits]


class ESMM(Layer):
	"""
		在mlp基础上的ESMMmodel实现
	"""
	def __init__(self):
		"""
			ts多任务样例的改写
			init里定义layer及变量
		"""
		super(ESMM, self).__init__()


		self.ctr_dense1 = Dense(500, activation='relu')
		self.ctr_dense2 = Dense(200, activation='relu')
		self.ctr_dense3 = Dense(100, activation='relu')
		self.ctr_out = Dense(1)
		self.ctr_drop = tf.layers.Dropout(0.6)

		self.cvr_dense1 = Dense(500, activation='relu')
		self.cvr_dense2 = Dense(200, activation='relu')
		self.cvr_dense3 = Dense(100, activation='relu')
		self.cvr_out = Dense(1)
		self.cvr_drop = tf.layers.Dropout(0.6)

		self.flatten = tf.layers.Flatten()
		
	
	def build(self, input_shape):
		"""
			变量的size定义，也可以在__init__完成，tf.layers.Layer需要继承改函数
		"""
		pass

	def call(self, inputx):
		"""
			inputx:
				x: s, l , m, dense四个维度向量的拼接
				show_index: 展示pos位置向量
				st: sparse table
			return:
				ctr_logits: ctr的预测值
				ctcvr_logits: ctr*cvr的预测值
			网络结构：
				step1: ctr: input->500-200->100, dropout
				step2: ctr: step1 + show_index 共122维
				step3: ctr: 122->1->sigmoid
				step4: cvr: 做step1-step3的操作
				step5: pctcvr: ctr*cvr
		"""
		x, show_index, st = inputx 

		t_d1 = self.ctr_dense1(x)
		t_d2 = self.ctr_dense2(t_d1)
		t_d3 = self.ctr_dense3(t_d2)
		ctr_in = tf.concat([t_d3, show_index], axis=-1)

		ctr_value = self.ctr_out(ctr_in)
		ctr = self.flatten(ctr_value)
		ctr_logits = tf.sigmoid(ctr)


		v_d1 = self.cvr_dense1(x)
		v_d2 = self.cvr_dense2(v_d1)
		v_d3 = self.cvr_dense3(v_d2)
		cvr_in = tf.concat([t_d3, show_index], axis=-1)
		cvr_value = self.cvr_out(cvr_in)
		cvr = self.flatten(cvr_value)
		cvr_logits = tf.sigmoid(cvr)

		ctcvr_logits = tf.multiply(ctr_logits, cvr_logits)
		return [ctr_logits, ctcvr_logits]

class LR(Layer):
	"""
		在mlp基础上的ESMMmodel实现
	"""
	def __init__(self):
		"""
			ts多任务样例的改写
			init里定义layer及变量
		"""
		super(LR, self).__init__()
		self.ctr_out = Dense(1, activation='sigmoid')
		self.cvr_out = Dense(1, activation='sigmoid')
		
	
	def build(self, input_shape):
		"""
			变量的size定义，也可以在__init__完成，tf.layers.Layer需要继承改函数
		"""
		pass

	def call(self, inputx):
		"""
			inputx:
				x: s, l , m, dense四个维度向量的拼接
				show_index: 展示pos位置向量
				st: sparse table
			return:
				ctr_logits: ctr的预测值
				ctcvr_logits: ctr*cvr的预测值
			网络结构：
				step1: ctr: input->1, sigmoid
				step1: cvr: input->1, sigmoid

		"""
		x, show_index, st = inputx
		ctr_in = tf.concat([x, show_index], axis=-1)
		ctr_logits = self.ctr_out(ctr_in)
		cvr_logits = self.cvr_out(ctr_in)
		return [ctr_logits, cvr_logits]


class FM(Layer):
	"""
		在mlp基础上的ESMMmodel实现
	"""
	def __init__(self):
		"""
			ts多任务样例的改写
			init里定义layer及变量
		"""
		super(FM, self).__init__()

		self.last_layer = Dense(1, activation='sigmoid')
		
	
	def build(self, input_shape):
		"""
			变量的size定义，也可以在__init__完成，tf.layers.Layer需要继承改函数
		"""
		pass

	def call(self, inputx):
		"""
			inputx:
				x: s, l , m, dense四个维度向量的拼接
				show_index: 展示pos位置向量
				st: sparse table
			return:
				ctr_logits: ctr的预测值
				ctcvr_logits: ctr*cvr的预测值
			网络结构：
				step1: : first order 输入
				step2: fm 值计算

				step5: pctcvr: ctr*cvr
		"""
		x, show_index, st = inputx
		predicts = []
		for i in range(0, 2):
			#first order input
			first_in = tf.concat([x, show_index], axis=-1)

			#scecond order input
			sum_feature = tf.reduce_sum(st, 1)
			sum_feature_square = tf.square(sum_feature)

			square_feature = tf.square(st)
			square_feature_sum = tf.reduce_sum(square_feature, 1)
			cross = 0.5 * tf.subtract(sum_feature_square, square_feature_sum)

			mid_out = tf.concat([first_in, cross], axis=-1)
			predict = self.last_layer(mid_out)
			predicts.append(predict)

		return predicts


class DeepFM(Layer):
	"""
		在mlp基础上的ESMMmodel实现
	"""
	def __init__(self):
		"""
			ts多任务样例的改写
			init里定义layer及变量
		"""
		super(DeepFM, self).__init__()

		self.first_layer = Dense(1)
		self.last_layer = Dense(1, activation='sigmoid')


		self.deep_dense1 = Dense(500, activation='relu')
		self.deep_dense2 = Dense(200, activation='relu')
		self.deep_dense3 = Dense(100, activation='relu')

		self.flatten = tf.layers.Flatten()
		
	
	def build(self, input_shape):
		"""
			变量的size定义，也可以在__init__完成，tf.layers.Layer需要继承改函数
		"""
		pass

	def call(self, inputx):
		"""
			inputx:
				x: s, l , m, dense四个维度向量的拼接
				show_index: 展示pos位置向量
				st: sparse table
			return:
				ctr_logits: ctr的预测值
				cvr_logits: cvr的预测值
			网络结构：

				step5: pctcvr: ctr*cvr
		"""
		x, show_index, st = inputx 
		predicts = []
		for i in range(0, 2):
			#first order input
			first_in = tf.concat([x, show_index], axis=-1)

			#scecond order input
			sum_feature = tf.reduce_sum(st, 1)
			sum_feature_square = tf.square(sum_feature)

			square_feature = tf.square(st)
			square_feature_sum = tf.reduce_sum(square_feature, 1)
			cross = 0.5 * tf.subtract(sum_feature_square, square_feature_sum)
			
			first_order = self.first_layer(first_in)
			d_1 = self.deep_dense1(first_in)
			d_2 = self.deep_dense2(d_1)
			d_3 = self.deep_dense3(d_2)

			mid_out = tf.concat([first_order, cross, d_3], axis=-1)
			predict = self.last_layer(mid_out)
			predicts.append(predict)

		return predicts


class DCNModel(Layer):
	"""
		deep cross network
	"""
	def __init__(self):
		"""
			ts多任务样例的改写
			init里定义layer及变量
		"""
		super(DCNModel, self).__init__()


		self.last_layer = Dense(1, activation='sigmoid')

		self.deep_dense1 = Dense(500, activation='relu')
		self.deep_dense2 = Dense(200, activation='relu')
		self.deep_dense3 = Dense(200, activation='relu')
	
	def build(self, input_shape):
		"""
			变量的size定义，也可以在__init__完成，tf.layers.Layer需要继承改函数
		"""
		pass
	def cross_net(self, x_in, emb, layer_num):
		"""
			input:
				x: 原始输入向量
				emb: 第n层cross网络结果
				layer_num: 层数
		"""
		dim = x_in.get_shape().as_list()[1]
		'''
		w = tf.get_variable("cross_layer_W{}".format(layer_num),
            [1, dim],
            initializer = tf.contrib.layers.xavier_initializer(),
			dtype=tf.float32)
		b = tf.get_variable("cross_layer_b{}".format(layer_num),
            [1, dim],
			initializer = tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32)
		'''
		w = tf.Variable(tf.random_normal([1, dim], stddev=0.01))
		b = tf.Variable(tf.random_normal([1, dim], stddev=0.01))
		mid1 = keras.backend.batch_dot(tf.reshape(x_in, (-1, dim, 1)), tf.reshape(emb, (-1, 1, dim)))
		mid2 = keras.backend.sum(w * mid1, 1, keepdims=True)
		mid = tf.reshape(mid2, (-1, dim))
		out = mid + b
		out = out + x_in
		return out

	def call(self, inputx):
		"""
			inputx:
				x: s, l , m, dense四个维度向量的拼接
				show_index: 展示pos位置向量
				st: sparse table
			return:
				ctr_logits: ctr的预测值
				cvr_logits: cvr的预测值
			网络结构：
				step1: deep 部分计算
				step2: 根据cross层数叠加计算向量
				step3: 计算ctr, cvr
		"""
		x, show_index, st = inputx 

		predicts = []
		for i in range(0, 2):

			d_1 = self.deep_dense1(x)
			d_2 = self.deep_dense2(d_1)
			d_3 = self.deep_dense3(d_2)
			
			emb = x 
			for j in range(0, 3):
				name = str(i) + str(j)
				emb = self.cross_net(x, emb, name)

			mid_out = tf.concat([emb, d_3], axis=-1)
			predict = self.last_layer(mid_out)
			predicts.append(predict)

		return predicts


def product(query, key):
	"""
	din 的attention网络结构
		input:
			query: [B,H] product_id
			key: : [B, T, H]
	"""
	len_key = key.get_shape().as_list()[1]
	len_query = query.get_shape().as_list()[-1]
	queries = tf.tile(query, [1, len_key])
	queries = tf.reshape(queries, [-1, len_key, len_query])
	#print(queries.shape, key.shape)
	v = queries * key
	#print(v.shape)
	return v


def din_attention(query, key):
	"""
	din 的attention网络结构
		input:
			query: [B,H] product_id
			key: : [B, T, H]
	"""
	len_key = key.get_shape().as_list()[1]
	len_query = query.get_shape().as_list()[-1]
	batch = [key.get_shape().as_list()[0]]
	queries = query#tf.tile(query, [1, len_key])
	queries = tf.reshape(queries, [-1, len_key, len_query])
	din_all = tf.concat([queries, key, queries*key], axis=-1)
	din_all = tf.layers.batch_normalization(inputs = din_all)
	#mlp sim calculate
	a_layer_1 = Dense(100, activation='relu')(din_all)
	a_layer_2 = Dense(50, activation='relu')(a_layer_1)
	a_layer_3 = Dense(1)(a_layer_2)
	outputs = tf.reshape(a_layer_3, [-1,1,len_key])	
	#mask
	#key_masks = tf.sequence_mask(batch, len_key)   # [B, T]
  	#print(key_masks.shape)
	#key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  	#print(key_masks.shape)
	#paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
	#print(key_masks.shape, outputs.shape, paddings.shape)
	#outputs = tf.where(key_masks, outputs, paddings)
	#######

	#scale
	outputs = outputs / (len_query ** 0.5)
	outputs = tf.nn.softmax(outputs)
	#print(outputs.shape)
	outputs = tf.matmul(outputs, key)
	#print(outputs)
	#outputs = tf.reshape(outputs, [-1, len_query])
	#print(outputs.shape)
	return outputs


def scaled_dot_product_attention(q, k, v, mask):
	"""计算注意力权重。
	q, k, v 必须具有匹配的前置维度。
	k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
	虽然 mask 根据其类型（填充或前瞻）有不同的形状，
	但是 mask 必须能进行广播转换以便求和。

	参数:
	q: 请求的形状 == (..., seq_len_q, depth)
	k: 主键的形状 == (..., seq_len_k, depth)
	v: 数值的形状 == (..., seq_len_v, depth_v)
	mask: Float 张量，其形状能转换成
		  (..., seq_len_q, seq_len_k)。默认为None。

	返回值:
	输出，注意力权重
	"""

	matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

	# 缩放 matmul_qk
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	# 将 mask 加入到缩放的张量上。
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)  

	# softmax 在最后一个轴（seq_len_k）上归一化，因此分数
	# 相加等于1。
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

	output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

	return output, attention_weights


class MultiHeadAttention(Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = Dense(d_model)
		self.wk = Dense(d_model)
		self.wv = Dense(d_model)

		self.dense = Dense(d_model)

	def split_heads(self, x, batch_size):
		"""
		分拆最后一个维度到 (num_heads, depth).
		转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]

		q = self.wq(q)  # (batch_size, seq_len, d_model)
		k = self.wk(k)  # (batch_size, seq_len, d_model)
		v = self.wv(v)  # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(
			q, k, v, mask)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention, 
									  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

		return output


class MMOE(Layer):
	def __init__(self):
		"""
			ts多任务样例的改写
			init里定义layer及变量
		"""
		super(MMOE, self).__init__()

		self.ctr_out = Dense(1)
		self.cvr_out = Dense(1)
		self.experts_num = 10
		self.task_num = 2

		self.flatten = tf.layers.Flatten()
	
	def build(self, input_shape):
		"""
			变量的size定义，也可以在__init__完成，tf.layers.Layer需要继承改函数
		"""
		pass

	def call(self, inputx):
		"""
			inputx:
				x: s, l , m, dense四个维度向量的拼接
				show_index: 展示pos位置向量
				st: sparse table
			return:
				ctr_logits: ctr的预测值
				cvr_logits: cvr的预测值
			网络结构：
				step1: experts 网络构建 batch * embedding * unit * experts_num 
				step2: gates 网络构建 batch * experts_num * task_num
				step3: gate * expert expert网络权重相加
				step4: cvr, cvr 专有网络输出
		"""
		x, show_index, st = inputx
		experts = []
		gates = []
		final_output = []
		for i in range(0, self.experts_num):
			#t_d1 = Dense(500, activation='relu')(x)
			#t_d2 = Dense(200, activation='relu')(t_d2)
			t_d3 = Dense(10, activation='relu')(x)
			experts.append(tf.expand_dims(t_d3, axis=2))
		experts_output = tf.concat(experts, 2)
		for i in range(0, self.task_num):
			g_d1 = Dense(10, activation='softmax')(x)
			gates.append(tf.expand_dims(g_d1, axis=1))

		for gate in gates:
			weighted_expert_output = experts_output * repeat_elements(gate, 10, axis=1)
			final_output.append(sum(weighted_expert_output, axis=2))

		ctr_logits = tf.sigmoid(self.ctr_out(final_output[0]))
		cvr_logits = tf.sigmoid(self.cvr_out(final_output[1]))
		return ctr_logits, cvr_logits


class PLE(Layer):
	def __init__(self):
		"""
			ts多任务样例的改写
			init里定义layer及变量
			single level
		"""
		super(PLE, self).__init__()

		self.ctr_out = Dense(1)
		self.cvr_out = Dense(1)
		self.experts_num = 10
		self.task_num = 2

		self.flatten = tf.layers.Flatten()
	
	def build(self, input_shape):
		"""
			变量的size定义，也可以在__init__完成，tf.layers.Layer需要继承改函数
		"""
		pass

	def call(self, inputx):
		"""
			inputx:
				x: s, l , m, dense四个维度向量的拼接
				show_index: 展示pos位置向量
				st: sparse table
			return:
				ctr_logits: ctr的预测值
				cvr_logits: cvr的预测值
			网络结构：
				step1: share_experts 网络构建 batch  * unit * experts_num 
				step2: gates 网络构建 batch * experts_num * task_num
				step3: task_experts 网络构建 batch * unit * experts_num * task_num
				step4: gate * (shapre_expert + task_expert) 网络权重相加
				step5: cvr, cvr 专有网络输出
		"""
		x, show_index, st = inputx
		share_experts = []
		task_experts_output = []
		gates = []
		final_output = []
		for i in range(0, self.experts_num):
			#t_d1 = Dense(500, activation='relu')(x)
			#t_d2 = Dense(200, activation='relu')(t_d2)
			t_d3 = Dense(10, activation='relu')(x)
			share_experts.append(tf.expand_dims(t_d3, axis=2))
		share_experts_output = tf.concat(share_experts, 2)

		for i in range(0, self.task_num):
			g_d1 = Dense(20, activation='softmax')(x)
			gates.append(tf.expand_dims(g_d1, axis=1))

			task_exp = []
			for i in range(0, self.experts_num):
				t_d = Dense(10, activation='relu')(x)
				task_exp.append(tf.expand_dims(t_d, axis=2))
			task_experts_output.append(tf.concat(task_exp, 2))

		for gate, task_expert in zip(gates, task_experts_output):
			merge_expert = tf.concat([task_expert, share_experts_output], 2)

			weighted_expert_output = merge_expert * repeat_elements(gate, 10, axis=1)

			final_output.append(sum(weighted_expert_output, axis=2))

		ctr_logits = tf.sigmoid(self.ctr_out(final_output[0]))
		cvr_logits = tf.sigmoid(self.cvr_out(final_output[1]))
		return ctr_logits, cvr_logits


def co_action(key, query):
	"""
		CAN: Revisiting Feature Co-Action for Click-Through Rate Prediction
	"""
	actions = []
	key_2 = key * key * key
	key = tf.expand_dims(key, axis=2)
	#key_2 = tf.expand_dims(key_2, axis=2)
	for i in range(0, 30):
		item = query[:,i,:]

		item = tf.expand_dims(item, axis=1)
		#b = tf.Variable(tf.random_normal([1, dim], stddev=0.001))
		mid_value = tf.tanh(tf.matmul(item, key))
		#mid_value_2 = tf.tanh(tf.matmul(item, key_2))
		actions.append(mid_value)
		#actions.append(mid_value_2)
	action_concat = tf.reshape(tf.concat(actions, 1), [-1, 30])

	return action_concat
