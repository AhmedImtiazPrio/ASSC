from __future__ import print_function, division, absolute_import
import numpy as np
# from keras.preprocessing.image import Iterator
from scipy import linalg
from scipy.signal import resample
import keras.backend as K
import warnings
from scipy.ndimage.interpolation import shift
import threading
# from six.moves import range

class Iterator(object):
    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen) ####### SEEEEEED চেঞ্জ হচ্ছে নতুন ব্যপার স্যপারের জন্য ##########
            if self.batch_index == 0:      ###############
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n ####### উপর থেকে কত নিচে আছে সেটার হিসাব। যদি একবার পুরোটা ঘুরে আবার উপরে উঠে যায়, সেটাও হিসাবে ধরা হচ্ছে। কেন এমন টা হবে দেখতে হবে #############
            if n >= current_index + batch_size: ###### এটা সমান তখন ই হবে যখন একেবারে শেষ হয়ে গেছে। যে আরেকটু আগালেই ডেড এন্ড #######
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index ######## একেবারে তলায় চলে আসার পরে, যেটুক বাকি আছে, সেটুক নিয়েই ব্যাচ ####################
                self.batch_index = 0 ############# যেহেতু ব্যাচ ফুরিয়ে গেছে, তাই ব্যাচ ইন্ডেক্স গোড়ায় নিয়ে গিয়ে শেষ করা, পরের বারের জন্য। ######################
            self.total_batches_seen += 1  ######## কতগুলা ব্যাচ ইল্ড করে করে বের হল সেইটার কাউন্টার ##########################
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)        ####  একেবারে শেষ বিন্দু পর্যন্ত ডেটা টে ইউজ করার জন্য কারেন্ট ব্যাচ সাইজ ও পাঠানো হচ্ছে, কারেন্ট ইন্ডেক্স এর সাথে।
                                                            ##### আর কাটা হচ্ছে, কারেন্ট পজিশন থেকে শুরু করে কারেন্ট ব্যাচ সাইজ পরিমান ##########

###################  YIELD এর থেকে প্রতিবার নেক্সট নেক্সট করে একটা করে করে বের হবে ###################



    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class NumpyArrayIterator(Iterator):
    def __init__(self, x, y, audio_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 subset=None):

            split_idx = int(len(x) * audio_data_generator._validation_split)
            if subset == 'validation':
                x = x[:split_idx]
                if y is not None:
                    y = y[:split_idx]
            else:
                x = x[split_idx:]
                if y is not None:
                    y = y[split_idx:]
        if data_format is None:
            data_format = 'channels_last'
        self.x = np.asarray(x, dtype=K.floatx())

        channels_axis = 2 if data_format == 'channels_last' else 1

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.audio_data_generator = audio_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.audio_data_generator.random_transform(x.astype(K.floatx()))
            x = self.audio_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            raise NotImplementedError

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)


class AudioDataGenerator(object):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 roll_range=0.,
                 brightness_range=None,
                 zoom_range=0.,
                 shift=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 noise=None,
                 validation_split=0.0):
        if data_format is None:
            data_format = 'channels_last'
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.roll_range = roll_range
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.fill_mode = fill_mode
        self.cval = cval
        self.shift = shift
        self.noise = noise

        if data_format not in {'channels_last', 'channels_first'}:
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
        if data_format == 'channels_last':
            self.channel_axis = 2
            self.row_axis = 1
        if validation_split and not 0 < validation_split < 1:

        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False

        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True

        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', subset=None):
        if self.noise:
            shuffle = True
            warnings.warn('This AudioDataGenerator specifies '
                          '`noise`, which overrides the setting of'
                          '`shuffle` as True'
                          )
        return NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        raise NotImplementedError

    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())


        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-2:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)

    def random_transform(self, x, seed=None):
        img_row_axis = self.row_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if not (self.zoom_range[0] == 1 and self.zoom_range[1] == 1):
            zx = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            input_length = x.shape[img_row_axis]
            x = resample(x, num=int(zx * x.shape[img_row_axis]), axis=img_row_axis)
            if x.shape[img_row_axis] >= input_length:
                x = x[:input_length]
            else:
                x = np.pad(x, ((0, input_length - x.shape[img_row_axis]), (0, 0)),
                           'constant', constant_values=(0, np.mean(x)))

        if shift:
            hx = np.random.uniform(-self.shift, self.shift)
            x = shift(x, (int(hx * x.shape[img_row_axis]), 0), mode=self.fill_mode, cval=self.cval)

        if self.roll_range:
            tx = np.random.uniform(-self.roll_range, self.roll_range)
            if self.roll_range < 1:
                tx *= x.shape[img_row_axis]
            x = np.roll(x, int(tx), axis=(img_row_axis))

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = np.flip(x, axis=img_row_axis)

        if (self.noise):
            if np.random.random() < 0.5:
                if self.noise[-1] == 'Uniform':
                    x = x + np.random.uniform(self.noise[0], self.noise[1], size=x.shape)
                elif self.noise[-1] == 'Normal':
                    x = x + np.random.normal(self.noise[0], self.noise[1], size=x.shape)

        if self.brightness_range is not None:
            x = random_brightness(x, self.brightness_range)

        return x

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):

        x = np.asarray(x, dtype=K.floatx())
        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            raise NotImplementedError

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis))
            broadcast_shape = [1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis))
            broadcast_shape = [1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)


def random_brightness(x, brightness_range):

    if len(brightness_range) != 2:
        raise ValueError('`brightness_range should be tuple or list of two floats. '
                         'Received arg: ', brightness_range)
    u = np.random.uniform(brightness_range[0], brightness_range[1])
    x = u * x
    return x