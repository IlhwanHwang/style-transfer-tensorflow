import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
from argparse import Namespace

meta = Namespace()
meta.ALPHA = 3e-2
meta.BETA = 1e0
meta.LEARNING_RATE = 1e1
meta.TRAINING_STEP = 1000
meta.INTERVAL_SAVE = 10000
meta.INTERVAL_RESULT = 100
meta.INTERVAL_PRINT = 10
meta.MAX_WIDTH = 512
meta.MAX_HEIGHT = 512

meta.STYLEFILE = './resource/starry_night.jpg'
meta.CONTENTSFILE = './resource/rose.jpg'
meta.OUTFILE = './result/result_step_{}.png'
meta.SAVEFILE = './model/main.ckpt'
meta.WEIGHTFILE = 'imagenet-vgg-verydeep-19'

weight_file = 'imagenet-vgg-verydeep-19'
net = scipy.io.loadmat(weight_file)

# Helper function to load MatConvNet network.
def generate(image, breakpoint=None, verbose=False):
	graph = image
	graph_dic = {}

	for i, layer in enumerate(net['layers'][0]):
		layer = layer[0][0]

		kind = layer['type'][0]
		name = layer['name'][0]
		if verbose:
			print("{}: {} {}".format(i + 1, kind, name))

		# conv and pool have things in common.
		if kind == 'conv' or kind == 'pool':
			# First of all, add padding layer. To apply padding sophiscatedly, we should go through this layer. Tensorflow's native conv2d only supports simple padding schemes.
			paddings = layer['pad'][0].astype(np.int_)
			# And we don't need to pad when padding size is 0.
			if np.any(paddings != [0, 0, 0, 0]):
				# But I'm not sure it's correct dimension order...
				paddings = [[0, 0], [paddings[0], paddings[1]], [paddings[2], paddings[3]], [0, 0]]
				graph = tf.pad(graph, paddings=paddings, mode='REFLECT', name=(name + '_padding'))

			# Only two provided, but Tensorflow needs 4 arguments. In most cases, the rests are filled with 1s.
			strides = [1, 1, 1, 1]
			strides[1] = layer['stride'][0][0]
			strides[2] = layer['stride'][0][1]

			if kind == 'conv':
				# Align axis accordingly. Matconvnet uses order (W, H, Ci, Co) but Tensorflow uses order (H, W, Ci, Co).
				kernel = np.transpose(layer['weights'][0][0], (1, 0, 2, 3))
				# Deference bias one more time to shift column to row.
				bias = layer['weights'][0][1][0]
				
				graph = tf.nn.conv2d(graph, tf.constant(kernel), strides=strides, padding='VALID', name=(name + '_kernel'))
				graph = tf.nn.bias_add(graph, tf.constant(bias), name=(name + '_bias'))

			if kind == 'pool':
				# The same as the case of strides
				ksize = [1, 1, 1, 1]
				ksize[1] = layer['pool'][0][0]
				ksize[2] = layer['pool'][0][1]

				if layer['method'][0].upper() == 'MAX':
					method = tf.nn.max_pool
				elif layer['method'][0].upper() == 'AVG':
					method = tf.nn.avg_pool
				else:
					print("\tInvalid pooling method specified")
					continue

				graph = method(graph, name=name, strides=strides, ksize=ksize, padding='VALID')

		elif kind == 'relu':
			graph = tf.nn.relu(graph, name=name)

		elif kind == 'softmax':
			graph = tf.nn.softmax(graph, name=name)

		else:
			print('\tUnprocessible kind of layer found')

		graph_dic.update({name : graph})

		if verbose:
			print("\t{}".format(graph))

		if breakpoint:
			if name == breakpoint:
				if verbose:
					print("Breakpoint reached. Stop building graph.")
				break

	return graph_dic

# Session
sess = tf.Session()

# Actual images to learn
image_style = scipy.misc.imread(meta.STYLEFILE)
image_contents = scipy.misc.imread(meta.CONTENTSFILE)
image_mean = net['normalization'][0]['averageImage'][0].astype(np.uint8)

# Adjust size not too big
if image_contents.shape[0] > meta.MAX_HEIGHT:
	image_contents = scipy.misc.imresize(image_contents, meta.MAX_HEIGHT / image_contents.shape[0])

if image_contents.shape[1] > meta.MAX_WIDTH:
	image_contents= scipy.misc.imresize(image_contents, meta.MAX_WIDTH / image_contents.shape[1])

image_style = scipy.misc.imresize(image_style, image_contents.shape[0:2])
image_mean = scipy.misc.imresize(image_mean, image_contents.shape[0:2])

image_mean = image_mean.astype(np.float32)
image_style = image_style.astype(np.float32)
image_contents = image_contents.astype(np.float32)

dim = [1, image_contents.shape[0], image_contents.shape[1], 3]

print('Size destination is {} x {}'.format(dim[2], dim[1]))

image_mean = np.reshape(image_mean, dim)
image_style = np.reshape(image_style, dim) - image_mean
image_contents = np.reshape(image_contents, dim) - image_mean

# List of layers involved
list_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
list_contents = ['conv4_2', 'conv5_2']

style_loss_weight = [0.3, 0.25, 0.2, 0.15, 0.1]
contents_loss_weight = [0.3, 0.7]

# Image source to be style.
src_style_image = tf.constant(image_style)
src_style_layer_dic = generate(src_style_image, breakpoint=list_style[-1])

# Image source to be contents.
src_contents_image = tf.constant(image_contents)
src_contents_layer_dic = generate(src_contents_image, breakpoint=list_contents[-1])

# Source layer will be rasterized. There are constants.
src_style_layer_list = list(map(lambda x: src_style_layer_dic[x], list_style))
src_contents_layer_list = list(map(lambda x: src_contents_layer_dic[x], list_contents))

src_style_layer_list = [tf.constant(feature) for feature in sess.run(src_style_layer_list)]
src_contents_layer_list = [tf.constant(feature) for feature in sess.run(src_contents_layer_list)]

# And the target.
initializer = tf.constant(np.zeros_like(image_mean))
# initializer = tf.constant((image_style + image_contents) / 2.)
# initializer = tf.constant(image_mean) + tf.random_normal(dim, stddev=np.std(image_style))
target = tf.get_variable('generated_image', initializer=initializer)
dest_layer_dic = generate(target, breakpoint=list_contents[-1])

dest_style_layer_list = list(map(lambda x: dest_layer_dic[x], list_style))
dest_contents_layer_list = list(map(lambda x: dest_layer_dic[x], list_contents))

# Calculate loss for contents layer by layer.
contents_loss_layer_list = []
for i, (src_contents_layer, dest_contents_layer) in enumerate(zip(src_contents_layer_list, dest_contents_layer_list)):
	# Quite straight forward.
	dist = tf.square(tf.subtract(src_contents_layer, dest_contents_layer))
	dist = tf.reduce_sum(dist) * 0.5

	contents_loss_layer_list.append(contents_loss_weight[i] * dist)

contents_loss = tf.reduce_sum(tf.stack(contents_loss_layer_list))

# Calculate loss for style layer by layer.
style_loss_layer_list = []
for i, (src_style_layer, dest_style_layer) in enumerate(zip(src_style_layer_list, dest_style_layer_list)):

	shape = src_style_layer.get_shape().as_list()
	# Width and height are merged into 'position'
	flatten = tf.reshape(src_style_layer, [-1, shape[3]])
	# We will iterate over position.
	# The resulting tensor is 2-rank (Co, Co).
	src_style_gram = tf.matmul(tf.transpose(flatten), flatten)

	# Things are exactly same as source.
	shape = dest_style_layer.get_shape().as_list()
	flatten = tf.reshape(dest_style_layer, [-1, shape[3]])
	dest_style_gram = tf.matmul(tf.transpose(flatten), flatten)
	
	# We need to calculate the layer coefficient. Coefficient is given by N^-2M^-2/4, where N = number of filters, M = size of feature map.
	M, N = flatten.get_shape().as_list()
	coefficient = 1. / 4. / (tf.size(flatten, out_type=tf.float32) ** 2)

	# Then finally calculate loss by layer (E_l)
	dist = tf.square(tf.subtract(src_style_gram, dest_style_gram))
	dist = tf.reduce_sum(dist)

	# This is E_l
	style_loss_layer_list.append(style_loss_weight[i] * coefficient * dist)

# List to tensor
style_loss = tf.reduce_sum(tf.stack(style_loss_layer_list))

total_loss = meta.ALPHA * contents_loss + meta.BETA * style_loss
global_step = tf.Variable(0, trainable=False, name='global_step')
train = tf.train.AdamOptimizer(meta.LEARNING_RATE).minimize(total_loss, global_step=global_step)

# Let's start!
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	saver.restore(sess, ckpt.model_checkpoint_path)
else:
	sess.run(tf.global_variables_initializer())

for _ in range(meta.TRAINING_STEP):
	step = sess.run(global_step)

	if step % meta.INTERVAL_PRINT == 0:
		loss, loss_c, loss_s = sess.run([total_loss, contents_loss, style_loss], feed_dict={})
		loss_c = loss_c * meta.ALPHA
		loss_s = loss_s * meta.BETA
		print('{}: Total loss {:g}, Content loss {:g}, Style loss {:g}'.format(step, loss, loss_c, loss_s))

	if step % meta.INTERVAL_RESULT == 0:
		print('Printing result...')
		result = sess.run(target, feed_dict={})
		result = result[0] + image_mean[0]
		result = np.clip(result, 0, 255)
		result = result.astype(np.uint8)
		scipy.misc.imsave(meta.OUTFILE.format(step), result)

	if (step + 1) % meta.INTERVAL_SAVE == 0:
		print('Saving status...')
		saver.save(sess, meta.SAVEFILE, global_step=global_step)

	sess.run(train, feed_dict={})