# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""A test script for frame interpolation for a video.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_video_test \
   --video <filepath of the video> \
   --model_path <The filepath of the TF2 saved model to use> \
   --times_to_interpolate <Number of times to interpolate>

The output is saved to <the directory of the input video>/output_video.mp4.
"""
import os
from typing import Sequence

from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
from absl import logging
import numpy as np
import mediapy as media
import cv2

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


_VIDEO = flags.DEFINE_string(
    name='video',
    default=None,
    help='The filepath of the input video.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=6,
    help='The number of times to run recursive midpoint interpolation. '
    'The number of output frames will be 2^times_to_interpolate+1.')
_FPS = flags.DEFINE_integer(
    name='fps',
    default=0,
    help='Frames per second to play interpolated videos in slow motion.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=64,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='An int >= 1, number of patches along height, '
    'patch_height = height//block_height, should be evenly divisible.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='An int >= 1, number of patches along width, '
    'patch_width = width//block_width, should be evenly divisible.')


def _run_interpolator() -> None:
  """Writes interpolated mid frame from a given two input frame filepaths."""

  interpolator = interpolator_lib.Interpolator(
      model_path=_MODEL_PATH.value,
      align=_ALIGN.value,
      block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])

  vidcap = cv2.VideoCapture(_VIDEO.value)
  success, image = vidcap.read()
  fps = _FPS.value
  if fps == 0:
      fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))

  output_video_frames_list = []
  last_frame = None
  count = 0
  while success:
    # First batched image.
    img_rgb_1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
    image_1 = np.float32(img_rgb_1 / 255.0)

    success, image = vidcap.read()

    if success:
      # Second batched image.
      img_rgb_2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
      image_2 = np.float32(img_rgb_2 / 255.0)

      # Invoke the model for frame interpolation.
      frames = list(util.interpolate_recursively_from_memory([image_1, image_2],
                                         _TIMES_TO_INTERPOLATE.value, interpolator))

      output_video_frames_list.extend(frames[:-1])
      last_frame = frames[-1]
      logging.info('interpolator frame number %s.', str(count))
      count += 1

  if last_frame is not None:
      output_video_frames_list.append(last_frame)
  vidcap.release()

  output_dir_name = os.path.dirname(_VIDEO.value)
  output_basename = os.path.basename(_VIDEO.value)
  output_name_wo_ext = os.path.splitext(output_basename)[0]
  output_video_path = os.path.join(output_dir_name, output_name_wo_ext + "_output_video.mp4")
  media.write_video(output_video_path, output_video_frames_list, fps= fps)
  logging.info('Output video saved at %s.', output_video_path)

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  ffmpeg_path = util.get_ffmpeg_path()
  media.set_ffmpeg(ffmpeg_path)

  _run_interpolator()


if __name__ == '__main__':
  app.run(main)
