# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:41:19 2019

@author: xuyan
"""

import os
import sys
sys.path.append('models/research/audioset/')

import vggish_slim
import vggish_params
import vggish_input
import vggish_postprocess
import soundfile as sf
import numpy as np
import tensorflow as tf
#from scipy import signal

tf.reset_default_graph()
sess = tf.Session()

def CreateVGGishNetwork(hop_size=0.96):   # Hop size is in seconds.
  """Define VGGish model, load the checkpoint, and return a dictionary that points
  to the different tensors defined by the model.
  """
  vggish_slim.define_vggish_slim()
  checkpoint_path = 'models/research/audioset/vggish_model.ckpt'
  vggish_params.EXAMPLE_HOP_SECONDS = hop_size
  
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)

  layers = {'conv1': 'vggish/conv1/Relu',
            'pool1': 'vggish/pool1/MaxPool',
            'conv2': 'vggish/conv2/Relu',
            'pool2': 'vggish/pool2/MaxPool',
            'conv3': 'vggish/conv3/conv3_2/Relu',
            'pool3': 'vggish/pool3/MaxPool',
            'conv4': 'vggish/conv4/conv4_2/Relu',
            'pool4': 'vggish/pool4/MaxPool',
            'fc1': 'vggish/fc1/fc1_2/Relu',
            'fc2': 'vggish/fc2/Relu',
            'embedding': 'vggish/embedding',
            'features': 'vggish/input_features',
         }
  g = tf.get_default_graph()
  for k in layers:
    layers[k] = g.get_tensor_by_name( layers[k] + ':0')
    
  return {'features': features_tensor,
          'embedding': embedding_tensor,
          'layers': layers,
         }

def ProcessWithVGGish(vgg, x, sr):
  '''Run the VGGish model, starting with a sound (x) at sample rate
  (sr). Return a whitened version of the embeddings. Sound must be scaled to be
  floats between -1 and +1.'''

  # Produce a batch of log mel spectrogram examples.
  input_batch = vggish_input.waveform_to_examples(x, sr)
  # print('Log Mel Spectrogram example: ', input_batch[0])

  [embedding_batch] = sess.run([vgg['embedding']],
                               feed_dict={vgg['features']: input_batch})

  # Postprocess the results to produce whitened quantized embeddings.
  pca_params_path = 'models/research/audioset/vggish_pca_params.npz'

  pproc = vggish_postprocess.Postprocessor(pca_params_path)
  postprocessed_batch = pproc.postprocess(embedding_batch)
  # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
  return postprocessed_batch[0]

vgg = CreateVGGishNetwork(0.01)

#num_secs = 3
#freq = 1000
#sr = 44100
#t = np.linspace(0, num_secs, int(num_secs * sr))
#x = np.sin(2 * np.pi * freq * t)

#wav_data, sr = sf.read("audio_train/0a6dbf2c.wav", dtype='int16')
#maxs=float(max(abs(np.max(wav_data)),abs(np.min(wav_data))))
#assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
#samples = wav_data / maxs  # Convert to [-1.0, +1.0]

audio=os.listdir("audio_train/")
for i in range(len(audio)):
    f="audio_train/"+audio[i]
    wav_data, sr = sf.read(f, dtype='int16')
    if len(wav_data)<=60000:
        continue
    else:
    #maxs=float(max(abs(np.max(wav_data)),abs(np.min(wav_data))))
    #assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    #samples = wav_data / maxs  # Convert to [-1.0, +1.0]
        samples = wav_data / 32768.0
        #if len(samples)>132300:
        #    samples= signal.resample(samples,132300)
    
    
        postprocessed_batch = ProcessWithVGGish(vgg, samples[:132300], sr)
        out= "Downloads/freesound-audio-tagging/audio_train/embedding/" +audio[i][:-4]
        np.savetxt(out,postprocessed_batch,delimiter="\t")
    