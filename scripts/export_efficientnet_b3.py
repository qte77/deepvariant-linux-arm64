#!/usr/bin/env python3
"""Export trained EfficientNet-B3 checkpoint to SavedModel format.

Usage:
  python3 scripts/export_efficientnet_b3.py \
    --experiment_dir=$HOME/dv-training-data/output/efficientnet_b3_experiment \
    --output_dir=$HOME/dv-training-data/output/efficientnet_b3_savedmodel
"""

import json
import os
import sys

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

# Add repo root to path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from deepvariant import keras_modeling

FLAGS = flags.FLAGS

flags.DEFINE_string('experiment_dir', None, 'Path to training experiment directory.')
flags.DEFINE_string('output_dir', None, 'Path to write SavedModel. Default: experiment_dir/saved_model')
flags.DEFINE_string('checkpoint', None, 'Specific checkpoint to export. Default: best EMA checkpoint.')


def find_best_checkpoint(experiment_dir):
  """Find the best EMA checkpoint by F1 score from checkpoint filenames."""
  ema_dir = os.path.join(experiment_dir, 'checkpoints', 'ema')
  if not os.path.exists(ema_dir):
    raise FileNotFoundError(f'EMA checkpoint directory not found: {ema_dir}')

  best_ckpt = None
  best_f1 = -1.0
  for entry in os.listdir(ema_dir):
    ckpt_path = os.path.join(ema_dir, entry)
    if os.path.isdir(ckpt_path) and 'checkpoint-' in entry:
      # Format: checkpoint-{step}-{f1_score}
      parts = entry.split('-')
      if len(parts) >= 3:
        try:
          f1 = float(parts[-1])
          if f1 > best_f1:
            best_f1 = f1
            best_ckpt = ckpt_path
        except ValueError:
          continue

  if best_ckpt is None:
    # Fall back to latest checkpoint
    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
      return latest
    raise FileNotFoundError(f'No checkpoints found in {experiment_dir}')

  logging.info('Best EMA checkpoint: %s (F1=%.5f)', best_ckpt, best_f1)
  return best_ckpt


def main(argv):
  del argv

  experiment_dir = FLAGS.experiment_dir
  if not experiment_dir:
    raise ValueError('--experiment_dir is required')

  output_dir = FLAGS.output_dir or os.path.join(experiment_dir, 'saved_model')
  os.makedirs(output_dir, exist_ok=True)

  # Load example_info.json for input shape
  example_info_path = os.path.join(experiment_dir, 'checkpoints', 'example_info.json')
  if not os.path.exists(example_info_path):
    raise FileNotFoundError(f'example_info.json not found: {example_info_path}')

  with open(example_info_path) as f:
    example_info = json.load(f)

  input_shape = example_info['shape']
  logging.info('Input shape: %s', input_shape)

  # Build model
  model = keras_modeling.efficientnetb3(
      input_shape=input_shape,
      init_backbone_with_imagenet=False,
  )
  logging.info('Model built with %d parameters.', model.count_params())

  # Load checkpoint weights
  checkpoint_path = FLAGS.checkpoint or find_best_checkpoint(experiment_dir)
  logging.info('Loading checkpoint: %s', checkpoint_path)

  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(checkpoint_path)
  status.expect_partial()
  logging.info('Checkpoint loaded.')

  # Define serving signature
  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None] + input_shape, dtype=tf.float32, name='input')
  ])
  def serving_fn(input_tensor):
    predictions = model(input_tensor, training=False)
    return {'classification': predictions}

  # Save as SavedModel
  tf.saved_model.save(
      model,
      output_dir,
      signatures={'serving_default': serving_fn},
  )
  logging.info('SavedModel saved to: %s', output_dir)

  # Copy example_info.json alongside the model
  output_info_path = os.path.join(output_dir, 'example_info.json')
  with open(output_info_path, 'w') as f:
    json.dump(example_info, f)
  logging.info('example_info.json copied to: %s', output_info_path)

  # Verify the saved model
  loaded = tf.saved_model.load(output_dir)
  test_input = tf.zeros([1] + input_shape, dtype=tf.float32)
  result = loaded.signatures['serving_default'](test_input)
  logging.info('Verification: output shape = %s', result['classification'].shape)
  logging.info('Verification: output sum = %s (should be ~1.0)', tf.reduce_sum(result['classification']).numpy())

  print(f'\nSavedModel exported to: {output_dir}')
  print(f'Copy this directory to the ARM64 Docker image for inference.')


if __name__ == '__main__':
  app.run(main)
