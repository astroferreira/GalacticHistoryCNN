def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(fits_string, number_of_mm_10, number_of_mm_25, fracs, last_snapshot):
    feature = {
      'number_of_mm_10': _int64_feature(number_of_mm_10),
      'number_of_mm_25': _int64_feature(number_of_mm_25),
      'fracs': _float_feature(fracs),
      'last_snapshot': _int64_feature(last_snapshot),
      'image_raw': _bytes_feature(fits_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def generate_tfrecords(dataframe, destination, fits_lookup):
    record_file = 'mergenet/data/REGRESSION/train.tfrecords'
    with tf.io.TFRecordWriter(destination) as writer:
        for idx, row in dataframe.iterrows():
            fits_bytes = fits.getdata(f'{fits_lookup}/{row.rootname}.fits').astype(np.float64).tobytes()
            tf_example = image_example(fits_bytes, int(row.number_of_mm_10), int(row.number_of_mm_25), row.fracs, int(row.last_snapshot))
            writer.write(tf_example.SerializeToString())