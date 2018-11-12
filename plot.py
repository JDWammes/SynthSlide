
# import packages
from __future__ import print_function, division
from psychopy import visual, event, core, logging, gui, data
from psychopy.tools.filetools import fromFile, toFile
from SynthSlider import *
import numpy as np
import pandas as pd
import random
import csv
import math
import itertools
from PIL import Image
import PIL.ImageOps
import scipy
import os


StartUp = gui.Dlg(title="Fractal Experiment")
StartUp.addField('Subject Number:', 'test_subject_id')  # deliberately choose a longer name so that the text field is larger
StartUp.addField('Duration:', 15)  # trial length
StartUp.show()
# what happens when you click "OK" or "Cancel"
if StartUp.OK:
    starters = StartUp.data
    subid, sDur = starters
else:
    core.quit()

files=[i for i in os.listdir('Data') if i.split('-')[0] == subid]
sTrials = len(files)
print('Subject {}, {} trials, {} s in length'.format(subid, sTrials, sDur))

plot_trials(subid, sTrials, sDur)
