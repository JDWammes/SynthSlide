
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

# filenames (without .tiff extension) of stimuli that we want to test
# the 1st line of (8) pairs is visually similar
# the 2nd line of (8) pairs is visually dissimilar
# all other stimuli are used as distractors
set1 = [('c83-101_100A', 'c83-101_100B'), ('c79-97_86A', 'c79-97_86B'),
        ('c11-5_71A', 'c11-5_71B'), ('c102-109_57A', 'c102-109_57B'),
        ('c40-20_43A', 'c40-20_43B'), ('c47-34_29A', 'c47-34_29B'),
        ('c17-85_14A', 'c17-85_14B'), ('c56-68_0A', 'c56-68_0B')]
set2 = [(23, '23i'), (34, '34i'), (36, '36i'), (51, '51i'),
        (58, 60), (78, 92), (3, 4), (5, 6)]
set3 = [(58, '58i'), (60, '60i'), (78, '78i'), (92, '92i'),
        (23, 34), (36, 51), (3, 4), (5, 6)]

stem, chanLists, superIm, numOptions, allTrials, simDict, answerDict, nums1 = gen_trials()
set4 = []
for x, channel in enumerate(chanLists):
    makeSet = superIm[x]
    imA = makeSet[answerDict[channel]].split('.')[0]
    imBind = ((len(nums1) * 2) - 1 - answerDict[channel])
    imB = makeSet[imBind].split('.')[0]
    set4.append((imA, imB))


sets = [set1, set2, set3, set4]

# backup = np.unique([i[:-5] for i in os.listdir('../Synth') if i[:2] == 'bu'])


refresh = 60
# define custom green color
green = (51, 204, 51)

Parts = gui.Dlg(title="Type of Experiment")
Parts.addField('Type:', choices=["One Day", "Part 1", "Part 2"])
Parts.show()
if Parts.OK:
    parts = Parts.data[0]
    print(parts)
    if parts != "Part 2":
        # Build and show GUI to set parameters, i.e. choose tasks and their settings.
        # Values after the comma are the default values with which the GUI is presented.
        StartUp = gui.Dlg(title="Fractal Experiment")
        StartUp.addField('Subject Number:', 'test_subject_id')  # deliberately choose a longer name so that the text field is larger
        StartUp.addField('Actual Data (use full screen):', True)
        StartUp.addField('Image Set:', 4)
        #StartUp.addText('')
        StartUp.addText('Choose outcome task:')
        StartUp.addField('Numerosity?:', False)
        StartUp.addField('Associative Inference?:', False)
        StartUp.addField('New Associates?:', False)
        StartUp.addField('Visual Search?:', False)
        StartUp.addField('Slider?:', True)
        if parts == "One Day":
            #StartUp.addText('')
            StartUp.addText('Outcome task before AND after?:')
            StartUp.addField('Numerosity?:', False)
            StartUp.addField('Associative Inference?:', False)
            StartUp.addField('New Associates?:', False)
            StartUp.addField('Visual Search?:', False)
            StartUp.addField('Slider?:', False)
        #StartUp.addText('')
        StartUp.addField('# Template Runs?:', 0)
        StartUp.addField('# Stat Learning Runs?:', 8)
        StartUp.addField('# Outcome Runs?:', 1)
        StartUp.addField('# Pair Exposures (Stat/Template):', 5)
        StartUp.addField('# Frequency Judgments (Numerosity):', 12)
        StartUp.addField('# Repetitions (Associative/Search):', 3)
        StartUp.addField('% Targets (Visual Search):', 100)
        StartUp.addField('% Rectangles (Stat/Template):', 10)
        StartUp.addField('# Trials (Slider):', 64)
        StartUp.addField('Duration in s (Slider):', 15)


        StartUp.show()

        # what happens when you click "OK" or "Cancel"
        if StartUp.OK:
            starters = StartUp.data

            if parts == "One Day":
                # Assign the values that were input into the GUI to variables
                subid, check, set, \
                numer, associnf, newassoc, visua, slide, \
                numer1, associnf1, newassoc1, visua1, slide1, \
                templates, stats, outcomes, exposures, nums, encodings, targets, rectangles, \
                sTrials, sDur = starters
            else:
                subid, check, set, \
                numer, associnf, newassoc, visua, slide, \
                templates, stats, outcomes, exposures, nums, encodings, targets, rectangles, \
                sTrials, sDur = starters
                numer1, associnf1, newassoc1, visua1, slide1 = numer, associnf, newassoc, visua, slide

            # choose image set
            basepairs = sets[set-1]
            if set < 4:
                stem = '../Fractals/Synth/'
                chans = ['c83-101', 'c79-97', 'c11-5', 'c102-109',
                         'c40-20', 'c47-34', 'c17-85', 'c56-68']
            else:
                chans = ['x83-101', 'x79-97', 'x11-5', 'x102-109',
                         'x40-20', 'x47-34', 'x17-85', 'x56-68']

            # same thing (same order) as a list without tuples or nested list
            stim_list = [item for sublist in basepairs for item in sublist]

            # Build arrays and lists for the tasks
            check_tasks = np.array([numer, associnf, newassoc, visua, slide])
            task_names = np.array(['numerosity', 'associative_inf', 'new_associates', 'visual_search', 'slider'])
            tasks = list(task_names[check_tasks])
            _bef = np.array([numer1, associnf1, newassoc1, visua1, slide1])
            bef = list(_bef[check_tasks])
            aft = tasks if parts == "One Day" else [False, False, False, False, False]

            # calculate how long the whole experiment will take
            # How long does one run take in seconds, when settings are set to 1?
            _times = np.array([58, 43, 69, 97, 69, 97, 64, 64, int(sDur+1)]).astype(int)
            multipliers = np.array([exposures, nums, encodings, 1, encodings, 1, encodings, 1, int(sTrials)]).astype(int)
            __total_time = _times * multipliers  # array with total times for each task
            _total_time = np.array([__total_time[0],
                                    __total_time[1],
                                    np.sum(__total_time[2:4]),
                                    np.sum(__total_time[4:6]),
                                    np.sum(__total_time[6:8]),
                                    __total_time[8]
                                    ])
            trim = np.insert(np.copy(check_tasks), 0, True)
            total_time = _total_time * trim  # only take the time for tasks that were run
            total_time[0] *= (stats + (templates * 2))
            total_time[1:] *= outcomes
            prepost = np.insert(np.array([2 if i else 1 for i in list(_bef)]), 0, 1)
            total_time *= prepost
            time = np.sum(total_time)/60

            out = np.array((basepairs, outcomes, tasks, nums, bef))
            np.save('Data/{}_day2.npy'.format(subid), out)

            # Print summary of selected tasks and settings to the console.
            print("{} templating runs, {} stat runs, {} outcome runs: {}".format(templates, stats, outcomes, tasks))  # @tobias: outcomes is used for every task, because it dictates how many runs of the task will be done.
            print("Stat learning exposures per run: {}".format(exposures)) if stats > 0 else print('', end='')
            print("Association task encoding reps per item: {}".format(encodings)) if any(t for t in check_tasks[1:3]) else print('', end='')
            print("Numerosity trials per item: {}".format(nums)) if check_tasks[0] else print('', end='')
            print("Visual search study phase reps per run: {}".format(encodings)) if check_tasks[3] else print('', end='')
            print("Estimated time to completion: {} minutes".format(np.around(time, 2)))

            # Build a new GUI that summarized the choices of the previous GUI as a safeguard
            Confirm = gui.Dlg(title="Confirmation")
            Confirm.addText('')
            Confirm.addText("Tasks:", "{}".format(tasks))
            Confirm.addFixedField("Templating runs:", "{}".format(templates))
            Confirm.addFixedField("Stat Learning runs:", "{}".format(stats))
            Confirm.addFixedField("Outcome runs:", "{}".format(outcomes))
            Confirm.addText('')
            Confirm.addFixedField("Stat learning exposures per run:", "{}".format(exposures)) if stats > 0 else Confirm.addText('')
            Confirm.addFixedField("Association task encoding reps per item:", "{}".format(encodings)) if any(t for t in check_tasks[1:3]) else Confirm.addText('')
            Confirm.addFixedField("Numerosity trials per item:", "{}".format(nums)) if check_tasks[0] else Confirm.addText('')
            Confirm.addFixedField("Visual search study phase reps per run:", "{}".format(encodings)) if check_tasks[3] else Confirm.addText('')
            Confirm.addText('')
            Confirm.addText("Estimated time to completion: {} minutes, plus instructions".format(np.around(time, 2)))
            Confirm.show()

            if Confirm.OK:  # What happens when you click "OK" or "Cancel" in the confirmation GUI
                pass
            else:
                core.quit()  # Abort the program

        else:
            core.quit()  # Abort the program
    # If its day 2
    else:

        StartUp = gui.Dlg(title="Fractal Experiment")
        StartUp.addField('Subject Number:', 'test_subject_id')
        StartUp.addField('Actual Data (use full screen):', True)
        StartUp.show()
        if StartUp.OK:
            starters = StartUp.data
            subid, check = starters
            day2 = np.load("../Data/{}_day2.npy".format(subid))
            basepairs, outcomes, tasks, nums, aft = day2
            templates = stats = rectangles = encodings = exposures = targets = 0
            bef = [False, False, False, False]


rawTimer = core.Clock()
trialTimer = core.Clock()

# If the experiment is not collecting actual data (check=False)
# we will not use the full screen and the mouse will be visible.
mywin = visual.Window([1000, 750], color=(-1, -1, -1), monitor="testMonitor", units="pix", fullscr=check, waitBlanking=False)
m = event.Mouse(win=mywin)
m.setVisible(not check)

# define Esc as the exit key to abort the whole experiment
def quick_exit():
    if 'escape' in event.getKeys():
        core.quit()

# call Instruction screen with two messages and proceed with a delay if SPACE is pressed
def instruction_screen(text, width=800, delay=0.25, last=False, done=False):
    quick_exit()
    mywin.flip()
    # message 1
    instruct = visual.TextStim(mywin, text=text, wrapWidth=width,
                               alignHoriz='left', pos=(-(width/2), 100))
    # message 2
    if done:
        wrap = ''
    else:
        wrap = 'When you are ready to begin, press SPACE' if last else 'Press SPACE to continue'
    color = green if last else 'white'
    bold = True if last else False
    space = visual.TextStim(mywin, text=wrap, wrapWidth=width, colorSpace='rgb255',
                            alignHoriz='center', color=color, pos=(0, -100), bold=bold)
    # show messages
    instruct.draw()
    space.draw()
    mywin.update()
    # proceed with a delay if SPACE is pressed
    NoKey = True
    while NoKey:
        allKeys = event.getKeys()
        if len(allKeys) > 0:
            resp = allKeys[0]
            if resp == 'space':
                NoKey = False
    mywin.flip()
    core.wait(delay)


# Generates an ISI list made up of one, three, and five seconds
def gen_isi(exposures):
    num_trials = exposures * len(stim_list)
    modifier = int(math.ceil(num_trials / 5))
    ones = [1] * modifier * 3
    threes = [2] * modifier * 2
    # fives = [5] * modifier
    # ISIs = itertools.chain(ones, threes, fives)
    ISIs = itertools.chain(ones, threes)
    isi_list = list(ISIs)
    np.random.shuffle(isi_list)
    return isi_list


# generate trial list for new associations task
def gen_trial_new_assoc(exposures, pairset):
    stims = []
    pairs = [i for i in pairset]
    for i in pairs:
        stims.extend(i)
    # all_stim = np.setdiff1d(np.array(range(1, 102)), np.array([i for i in stims if type(i) is int]))  # only use stimuli that are not basepairs
    # new_assoc = np.random.choice(backup, 24, replace=False)  # randomly choose 24 of those stimuli
    _new_assoc = [i for i in backup]
    AB = ['A', 'B'] * 15
    np.random.shuffle(AB)
    np.random.shuffle(_new_assoc)
    new_assoc = [str(_new_assoc[addon]) + str(AB[addon]) for addon in range(len(_new_assoc))]
    new_pairs = [(new_assoc[i - 1], new_assoc[i]) for i in range(1, 17, 2)]  # build new pairs based on the random sequence
    inf_pairs = list(np.roll(np.copy(new_pairs), 2))
    lures = [(new_assoc[i - 1], new_assoc[i]) for i in range(17, 25, 2)]
    lures2 = list(np.roll(np.copy(lures), 1))
    lures2 = [(i[0], i[1]) for i in lures2]
    lures.extend(lures2)
    _test_list = []
    enc_list = []
    for i in range(len(pairs)):
        _test_list.append((pairs[i][0], new_pairs[i][0], new_pairs[i][1],
                          lures[i][0], inf_pairs[i][0], pairs[i], new_pairs[i], lures[i]))
        _test_list.append((pairs[i][1], new_pairs[i][1], new_pairs[i][0],
                          lures[i][1], inf_pairs[i][1], pairs[i], new_pairs[i], lures[i]))
    test_list = _test_list
    np.random.shuffle(test_list)
    for j in range(exposures):
        shuffled = _test_list
        np.random.shuffle(shuffled)
        if j != 0:
            while shuffled[0] == lastitem:
                np.random.shuffle(shuffled)
        enc_list.extend(shuffled)
        lastitem = shuffled[-1]
    vis_sim = ['sim' if 'i' in str(item[5]) else 'not' for item in test_list]
    return enc_list, test_list, vis_sim


# Generates trial list for numerosity task
def gen_trial_num(trials_per_pair, pairset):
    new_list = []
    pairs = [i for i in pairset]
    probe_sel = [0, 1] * int(math.ceil(trials_per_pair * len(pairs) / 2))  # needs to multiply with an integer
    hi_freq = ['targ', 'targ', 'fill', 'fill'] * int(math.ceil(trials_per_pair * len(pairs) / 4))
    correct_side = ['left', 'right', 'left', 'right'] * int(math.ceil(trials_per_pair * len(pairs) / 4))
    for j in pairs:
        j1, j2 = j
        per = j1.split('_')[-1].split('A')[0]
        chan = j1.split('_')[0]
        fills = [x for x in chans if x != chan]
        all = []
        num_lists = int(math.ceil(trials_per_pair / len(fills)))
        for w in range(num_lists):
            _all = list(range(0, len(fills)))
            np.random.shuffle(_all)
            all.extend(_all)
        for k in all[0:trials_per_pair]:
            fill_select = fills[np.random.randint(0, 7)]
            item1 = str(fill_select) + '_' + str(per) + 'A'
            item2 = str(fill_select) + '_' + str(per) + 'B'
            items = [item1, item2]
            pmnum = [0, 1]
            np.random.shuffle(pmnum)
            k1 = items[pmnum[0]]
            k2 = items[pmnum[1]]
            new_list.append([j1, j2, k1, k2])
    zipper = list(zip(probe_sel, hi_freq, correct_side, new_list))
    while any(i[3][0] == j[3][0] for i, j in list(zip(zipper, zipper[1:]))):
        np.random.shuffle(zipper)
    vis_sim = [str(item[3][0]).split('_')[-1].split('A')[0] for item in zipper]
    return zipper, vis_sim


# Generates trial sequence for stat learning
def gen_trial_stat(exposures, pairset):
    pairs = [i for i in pairset]
    _trial_list = []
    trial_list = []
    pair_list = []
    for j in range(exposures):
        shuffled = pairs
        np.random.shuffle(shuffled)
        if j != 0:
            while shuffled[0] == lastitem:
                np.random.shuffle(shuffled)
        _trial_list.extend(shuffled)
        lastitem = shuffled[-1]
    while any(i == j for i, j in list(zip(_trial_list, _trial_list[1:]))):
        assert False
    for i in _trial_list:
        trial_list.extend(i)
        pair_list.append(i)
        pair_list.append(i)
    return trial_list, pair_list


# Generates trials for perceptual templating
def gen_trial_prepost(exposures, pairset):
    pairs = [i for i in pairset]
    __trial_list = pairs
    _trial_list = []
    trial_list = []
    _pair_list = []
    pair_list = []
    lastitem = ""
    for i in __trial_list:
        _trial_list.extend(i)
        _pair_list.append(i)
        _pair_list.append(i)
    for j in range(exposures):
        shuffled = list(zip(_trial_list, _pair_list))
        np.random.shuffle(shuffled)
        while shuffled[0][1] == lastitem or any(i[1] == j[1] for i, j in list(zip(shuffled, shuffled[1:]))):
            # ensure that two consecutive stimuli are not identical
            np.random.shuffle(shuffled)
        t_l, p_l = list(zip(*shuffled))
        trial_list.extend(t_l)
        pair_list.extend(p_l)
        lastitem = shuffled[-1][1]
    while any(i == j for i, j in list(zip(pair_list, pair_list[1:]))):
        assert False
    return trial_list, pair_list


# Generates correct vectors of where a letter "T" will appear (boolean), the correct answer vector
def gen_letterT_list(trial_list, targets):
    num_Ts = int(math.floor(len(trial_list) / 100 * targets))
    T_indexes = np.random.choice(len(trial_list), size=num_Ts, replace=False)
    letterT_list = np.array([False] * len(trial_list))
    letterT_list[T_indexes] = True
    answers = ['left' if i else 'right' for i in letterT_list]

    return answers, letterT_list


# Generates trial sequence for visual search (based on perceptual templating function)
def gen_trial_vis(encodings, pairset, targets):
    basecoord_list = []
    pairlist = []
    angles = np.arange(-10, 350, 360/len(pairset))
    np.random.shuffle(angles)
    distances = [110]*len(pairset)
    targ_pos = list(zip(pairset, angles, distances))
    _trial_list = []
    for j in range(encodings+1):
        shuffled = targ_pos
        np.random.shuffle(shuffled)
        if j != 0:
            while shuffled[0] == lastitem:  # ensure that two consecutive stimuli are not identical
                np.random.shuffle(shuffled)
        _trial_list.extend(shuffled)
        lastitem = shuffled[-1]

    while any(i == j for i, j in list(zip(_trial_list, _trial_list[1:]))):
        assert False

    for pair, angle, distance in _trial_list:
        jit_angle = np.random.uniform(-5, 5)  # introduce some variation
        jit_distance = np.random.uniform(-5, 5)  # introduce some variation
        angle += jit_angle
        distance += jit_distance
        newx = int(np.round(math.cos(angle * math.pi / 180) * distance, 0))
        newy = int(np.round(math.sin(angle * math.pi / 180) * distance, 0))
        basecoord_list.append(tuple([newx, newy]))
        pairlist.append(pair)

    answers, letters_list = gen_letterT_list(_trial_list, targets)
    vis_sim = ['sim' if 'i' in str(item) else 'not' for item in pairlist]  # Is the current item an inverse of another item and therefore similar?
    trial_list = list(zip(pairlist, basecoord_list, vis_sim, answers, letters_list))
    learn_list = trial_list[:-len(pairset)]
    transfer_list = trial_list[-len(pairset):]

    return learn_list, transfer_list


# Generates vectors of where a patch will appear (boolean) 
# and the correct answer vector for the cover task. 
# In this task, the pairings are randomly shuffled.
def gen_cover(exposures, percent):
    num_trials = exposures * len(stim_list)
    num_patches = int(math.floor(num_trials / 100 * percent))
    num_free = num_trials - num_patches
    rect_list = [False] * num_free
    rect_list2 = [True] * num_patches
    rect_list.extend(rect_list2)
    np.random.shuffle(rect_list)
    answers = ['left' if i else 'right' for i in rect_list]
    np.random.shuffle(stim_list)
    return answers, rect_list


# Produces color-inverted fractal
# This function is not run during the task but needed to create stimuli with inverted RGB colors
def invert_image(image):
    im = Image.open('../Synth/' + str(image) + '.tiff')
    r, g, b, a = im.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    inverted = PIL.ImageOps.invert(rgb_image)
    r2, g2, b2 = inverted.split()
    final = Image.merge('RGBA', (r2, g2, b2, a))
    final.save('../Synth/' + str(image) + 'i.tiff')


# Takes an image number and generates the path, then draws it
def preload_image(image, mask, position=[0, 0], size=[350, 350]):
    curr_stim = visual.ImageStim(mywin,
                                 image=str(stem) + str(image) + '.png', # @jeff: delete '/fractals' before running
                                 pos=position, size=size)
    mask_stim = visual.ImageStim(mywin,
                                 image=str(stem) + str(mask) + '.png',
                                 pos=position, size=size)
    return curr_stim, mask_stim



# Takes an image number and generates the path, then draws it
def gen_image(image, mask, position=[0, 0], size=[350, 350]):
    curr_stim = visual.ImageStim(mywin,
                                 image=str(stem) + str(image) + '.png', # @jeff: delete '/fractals' before running
                                 pos=position, size=size)
    mask_stim = visual.ImageStim(mywin,
                                 image=str(stem) + str(mask) + '.png',
                                 pos=position, size=size)
    curr_stim.draw()
    mask_stim.draw()


# Generates a rectangular patch in the bound of the fractal
def gen_patch():
    angle = np.random.uniform(0, 360)
    distance = np.random.uniform(0, 100)
    newx = np.round(math.cos(angle * math.pi / 180) * distance, 0)
    newy = np.round(math.sin(angle * math.pi / 180) * distance, 0)
    patch = visual.Rect(mywin, width=45, height=45, pos=(newx, newy),
                        fillColor=(0.3, 0.3, 0.3), opacity=0.7, lineWidth=0)
    patch.draw()


# instructions for the associative inference task
def instruction_static(text, images, width=800, delay=0.25, last=False, double=True):
    quick_exit()
    mywin.flip()
    instruct = visual.TextStim(mywin, text=text, wrapWidth=width,
                               alignHoriz='left', pos=(-(width/2), 300))
    wrap = 'When you are ready to begin, press SPACE' if last else 'Press SPACE to continue'
    color = green if last else 'white'
    bold = True if last else False
    space = visual.TextStim(mywin, text=wrap, wrapWidth=width, colorSpace='rgb255',
                            alignHoriz='center', color=color, pos=(0, -300), bold=bold)
    if double:
        gen_image(images[0], position=[-150, 0])
        gen_image(images[1], position=[150, 0])
    else:
        gen_image(images[0], position=[0, -50], size=[790, 600])
    instruct.draw()
    space.draw()
    mywin.update()

    # Continue with SPACE key
    NoKey = True
    while NoKey:
        allKeys = event.getKeys()
        if len(allKeys) > 0:
            resp = allKeys[0]
            if resp == 'space':
                NoKey = False
    core.wait(delay)


# instructions for the associative inference task
def associative_inf_instruct():
    instruction_screen('The next task is a memory test. You will be presented with a cue image, and three possible images as answers. You will need to choose the image which was associated with the same initial pair as the cue.',
                       delay=0.0)
    instruction_static('To clarify, during the square detection task, a set of two images like these might always have been paired together...',
                       ['Greyed.001']*2, delay=0.0, double=False)
    instruction_static('In the last task, you studied pairs where the both of these images were newly paired with an alternate image...',
                       ['Greyed.002']*2, delay=0.0, double=False)
    instruction_static('For this upcoming task, you will be presented with the cue image on the left, and the correct answer would be the image on the right, because it was previously associated with the same pair. That pair serves as an intermediate link.',
                       ['Greyed.003']*2, delay=0.0, double=False)
    instruction_screen('On each trial, you will be presented with a cue image, and three possible answers below it. Press the left arrow to choose the image on the lefthand side, the down arrow to choose the image in the middle, or the right arrow to choose the image on the right.',
                       delay=0.0)
    instruction_screen('If you have any questions about this, please ask the experimenter now.', delay=2, last=True)


# instructions for the new associations task
def new_assoc_instruct():
    instruction_screen('The next task is a memory test. You will be presented with a cue image, and three possible images as answers. You will need to choose the image which was associated the cue.')
    instruction_screen('On each trial, you will be presented with a cue image, and three possible answers below it. Press the left arrow to choose the image on the lefthand side, the down arrow to choose the image in the middle, or the right arrow to choose the image on the right.')
    instruction_screen('If you have any questions about this, please ask the experimenter now.', delay=2, last=True)


# Writes the data for a given stat learning or pre/post trial
def write_data(task, run, trialn, pair, image, rect, answer, resp, resp_time, acc, sta):
    sta = sta.append({'Task': task, 'Run': run, 'Trial Number': trialn,
                      'Pair': pair, 'Item': image, 'Rect': rect,
                      'CorrResp': answer, 'Resp': resp, 'RT': resp_time,
                      'Acc': acc}, ignore_index=True)
    return sta

# Writes the data for a given new associations encoding trial
def write_newEncdata(task, run, trialn, pair, new_pair, lure_pair, correct,
                     cue, intrude, lure, newEnc):
    newEnc = newEnc.append({'Task': task, 'Run': run, 'Trial Number': trialn,
                            'CritPair': pair, 'SubPair': new_pair, 'LurePair': lure_pair,
                            'Cue': cue, 'Correct': correct, 'pmLure': intrude,
                            'uLure': lure}, ignore_index=True)
    return newEnc

# Writes the data for a given new associations retrieval trial
def write_newRetdata(task, run, trialn, pair, new_pair, lure_pair, cue, correct, intrude, lure,
                     critpos, pmpos, upos, corresp, resp, resp_time, acc, data):
    data = data.append({'Task': task, 'Run': run, 'Trial Number': trialn, 'CritPair': pair, 'SubPair': new_pair,
                        'LurePair': lure_pair, 'Cue': cue, 'Correct': correct, 'pmLure': intrude, 'uLure': lure,
                        'CorrPos': critpos, 'pmPos': pmpos, 'uPos': upos, 'CorrResp': corresp, 'Resp': resp,
                        'RT': resp_time, 'Acc': acc}, ignore_index=True)
    return data


# Writes the data for a given numerosity task trial
def write_numdata(task, run, trialn, items, critpair, critpos, critprobe,
                  fillprobe, corritem, corrresp, resp, resp_time, acc, num):
    num = num.append({'Task': task, 'Run': run, 'Trial Number': trialn, 'Items': items,
                      'CritPair': critpair, 'CritPos': critpos, 'CritProbe': critprobe,
                      'FillProbe': fillprobe, 'CorrItem': corritem, 'CorrResp': corrresp,
                      'Resp': resp, 'RT': resp_time, 'Acc': acc}, ignore_index=True)
    return num


# Writes the data for a given visual search trial
def write_vis_search(task, run, trialn, pair, image, 
                    letterT, answer, resp, resp_time, acc, vis):
    vis = vis.append({'Task': task, 'Run': run, 'Trial Number': trialn,
                      'Pair': pair, 'Item': image,
                      'letterT': letterT, 'CorrResp': answer, 'Resp': resp, 
                      'RT': resp_time, 'Acc': acc}, ignore_index=True)
    return vis


# Runs an entire stat learning or pre/post trial
def trial_run(image, answer, trialn, rect, data, pair, run, rects=False, task='stat'):
    quick_exit()
    resp = 'miss'
    resp_time = None
    acc = None
    masks = list(np.random.choice(50, 1, replace=True))
    maskfiles = ['mask{}'.format(mask) for mask in masks]
    gen_image(image, maskfiles[0])
    if rects and rect:
        gen_patch()
    event.clearEvents()
    mywin.update()
    trialTimer.reset()
    noKey = True
    while trialTimer.getTime() < 1.0:
        allKeys=event.getKeys(timeStamped=trialTimer)
        if len(allKeys) > 0 and noKey:
            resp = allKeys[0][0]
            if resp == answer:
                acc = 1
            else:
                acc = 0
            resp_time = allKeys[0][1]
            noKey = False
    data = write_data(task, run, trialn, pair, image, rect, answer, resp, resp_time,
                      acc, data)
    return data

# Runs an entire new association encoding trial
def newEnc_trial_run(cue, correct, intrude, lure, pair, new_pair, lure_pair,
                     trialn, run, data, enc_time):
    quick_exit()
    gen_image(cue)
    mywin.update()
    core.wait(enc_time)
    mywin.flip()
    core.wait(0.25)
    gen_image(correct)
    mywin.update()
    core.wait(enc_time)
    data = write_newEncdata('newEnc', run, trialn, pair, new_pair, lure_pair, cue,
                            correct, intrude, lure, data)
    return data

# Runs an entire new association retrieval trial
def newRet_trial_run(cue, correct, intrude, lure, pair, new_pair, lure_pair, trialn, run, data, ret_time):
    quick_exit()
    resp = 'miss'
    resp_time = None
    acc = None
    positions = ['left', 'down', 'right']
    pos_dict = dict(zip(positions, [[-350, -200], [-0, -200], [350, -200]]))
    np.random.shuffle(positions)
    gen_image(cue, position=[0, 200])
    gen_image(correct, position=pos_dict[positions[0]])
    gen_image(intrude, position=pos_dict[positions[1]])
    gen_image(lure, position=pos_dict[positions[2]])
    critpos = positions[0]
    pmpos = positions[1]
    upos = positions[2]
    answer = critpos
    mywin.update()
    event.clearEvents()
    trialTimer.reset()
    noKey = True
    while trialTimer.getTime() < ret_time:
        allKeys = event.getKeys(timeStamped=trialTimer)
        if len(allKeys) > 0 and noKey:
            resp = allKeys[0][0]
            if resp == answer:
                acc = 1
            else:
                acc = 0
            resp_time = allKeys[0][1]
            noKey = False
    data = write_newRetdata('newRet', run, trialn, pair, new_pair, lure_pair, cue,
                            correct, intrude, lure, critpos, pmpos, upos, answer,
                            resp, resp_time, acc, data)
    return data

# Runs an entire numerosity trial
def numer_trial_run(probe_stim, winner, corr_side, items, trialn, run, num, rsvp_time=0.1, probe_lag=0.25, num_resp=1.5):
    quick_exit()
    resp = 'miss'
    resp_time = None
    acc = None
    if winner == 'targ':
        rsvp = items[0:2] * 6
        rsvp2 = items[2:4] * 4
        if corr_side == 'left':
            critpos = "left"
            im1 = preload_image(items[probe_stim], position=[-190, 0])
            im2 = preload_image(items[probe_stim + 2], position=[190, 0])
        else:
            critpos = "right"
            im1 = preload_image(items[probe_stim], position=[190, 0])
            im2 = preload_image(items[probe_stim + 2], position=[-190, 0])
    else:
        rsvp = items[2:4] * 6
        rsvp2 = items[0:2] * 4
        if corr_side == 'left':
            critpos = "right"
            im1 = preload_image(items[probe_stim], position=[190, 0])
            im2 = preload_image(items[probe_stim + 2], position=[-190, 0])
        else:
            critpos = "left"
            im1 = preload_image(items[probe_stim], position=[-190, 0])
            im2 = preload_image(items[probe_stim + 2], position=[190, 0])
    answer = corr_side
    rsvp.extend(rsvp2)
    np.random.shuffle(rsvp)
    while any(i == j for i, j in list(zip(rsvp, rsvp[1:]))):
        np.random.shuffle(rsvp)
    for image in rsvp:
        curr_image = preload_image(image)
        for frame in range(int(rsvp_time * refresh)):
            curr_image.draw()
            mywin.flip()
    mywin.flip()
    im1.draw()
    im2.draw()
    core.wait(probe_lag)
    event.clearEvents()
    mywin.update()
    trialTimer.reset()
    noKey = True
    while trialTimer.getTime() < num_resp:
        allKeys = event.getKeys(timeStamped=trialTimer)
        if len(allKeys) > 0 and noKey:
            resp = allKeys[0][0]
            if resp == answer:
                acc = 1
            else:
                acc = 0
            resp_time = allKeys[0][1]
            noKey = False
    core.wait(1.0)
    num = write_numdata('numerosity', run, trialn, items, items[0:2], critpos,
                        items[probe_stim], items[probe_stim+2], winner, answer,
                        resp, resp_time, acc, num)
    return num


# @tobias, played around with different possible targets, including gratings
# Runs an entire visual search trial
def vis_trial_run(image, answer, trialn, letterT, data, pair, run, coord, letterTs=False, task='vis', height=18, color='white', op_start=0.0, op_stop=0.3, max_search_dur=10):
    quick_exit()
    resp = 'miss'
    resp_time = None
    acc = None

    # generate and show the fractal, start the stopwatch, record keypress and fade in target
    event.clearEvents()
    trialTimer.reset()
    noKey = True
    while (trialTimer.getTime() <= max_search_dur) and noKey: # the fractal stays on until a response or until the maximal search duration is over
        allKeys = event.getKeys(timeStamped=trialTimer)
        gen_image(image)
        grate1 = visual.GratingStim(mywin, mask='circle', pos=coord, size=height, sf=0.25, ori=355)
        grate2 = visual.GratingStim(mywin, mask='circle', pos=coord, size=height, sf=0.25, ori=85)
        op = ((trialTimer.getTime() / max_search_dur) * (op_stop - op_start)) + op_start
        grate1.opacity = op
        grate2.opacity = op
        grate1.draw()
        grate2.draw()
        mywin.update()
        if len(allKeys) > 0 and noKey:  # @jeff: Had no time to think about where to put this part so that it works.
            resp = allKeys[0][0]
            if resp == answer:
                acc = 1
            else:
                acc = 0
            resp_time = allKeys[0][1]
            noKey = False

    data = write_vis_search(task, run, trialn, pair, image, 
                            letterT, answer, resp, resp_time, acc, data)
    return data


# Runs the ISI depending on the time input
def set_isi(time):
    mywin.flip()
    core.wait(time)


# run an associative inference block
def associative_inf(IDnum, repetitions, pairset, trialn, run, enc_isi=2, enc_time=1, ret_time=5, ret_isi=1, interval=60):

    # predefine dataframe columns and generate trials
    infRet = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'CritPair', 'SubPair',
                                   'LurePair', 'Cue', 'Correct', 'pmLure', 'uLure',
                                   'CorrPos', 'pmPos', 'uPos', 'CorrResp', 'Resp',
                                   'RT', 'Acc'])
    infEnc = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'CritPair', 'SubPair',
                                   'LurePair', 'Cue', 'Correct', 'pmLure', 'uLure'])
    enc_list, test_list, vis_sim = gen_trial_new_assoc(repetitions, pairset)

    # show instructions
    instruction_screen('In this next task, you will view pairs of images, one after the other. Please try to remember which images were paired together for a later memory test. You will see each pair of images '+ str(repetitions) + ' time(s).')
    instruction_screen('The first member of a pair will be presented for ' + str(enc_time) + ' second(s), followed closely by the second member of the pair. You do not need to press any buttons, just do your best to remember the pairs.')
    instruction_screen('Once all the pairs have been presented, there will be a ' + str(
        interval) + ' second waiting period before the test.',
                       delay=2, last=True)

    # run the encoding task
    rawTimer.reset()
    for initial, correct, other, lure, fam_lure, pair, new_pair, lure_pair in enc_list:
        trialn += 1
        infEnc = newEnc_trial_run(initial, correct, fam_lure, lure,  pair, new_pair,
                                  lure_pair, trialn, run, infEnc, enc_time)
        set_isi(enc_isi)
    print(rawTimer.getTime())

    # write data to file
    infEnc.to_csv("Data/" + str(IDnum) + "_infEnc_" + str(run) + ".csv")

    trialn = 0  # reset trial number
    set_isi(interval)

    associative_inf_instruct() # show instructions

    # run the retrieval task
    rawTimer.reset()
    for _, cue, correct, lure, fam_lure, pair, new_pair, lure_pair in test_list:
        trialn += 1
        infRet = newRet_trial_run(cue, correct, fam_lure, lure, pair, new_pair,
                                  lure_pair, trialn, run, infRet, ret_time)
        set_isi(ret_isi)
    print(rawTimer.getTime())
    infRet['condition'] = vis_sim

    # write data to file
    infRet.to_csv("Data/" + str(IDnum) + "_infRet_" + str(run) + ".csv")


# Run new associate task
def new_associate(IDnum, repetitions, pairset, trialn, run, enc_isi=2, enc_time=1, ret_time=5, ret_isi=1, interval=60):

    # predefine dataframe columns and generate trials
    newRet= pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'CritPair', 'SubPair',
                                  'LurePair', 'Cue', 'Correct', 'pmLure', 'uLure',
                                  'CorrPos', 'pmPos', 'uPos', 'CorrResp', 'Resp',
                                  'RT', 'Acc'])
    newEnc = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'CritPair', 'SubPair',
                                   'LurePair', 'Cue', 'Correct', 'pmLure', 'uLure'])
    enc_list, test_list, vis_sim = gen_trial_new_assoc(repetitions, pairset)

    # show instructions
    instruction_screen('In this next task, you will view pairs of images, one after the other. Please try to remember which images were paired together for a later memory test. You will see each pair of images '+ str(repetitions) + ' time(s).')
    instruction_screen('The first member of a pair will be presented for ' + str(enc_time) + ' second(s), followed closely by the second member of the pair. You do not need to press any buttons, just do your best to remember the pairs.')
    instruction_screen('Once all the pairs have been presented, there will be a ' + str(interval) + ' second waiting period before the test.',
                       delay=2, last=True)

    # run the encoding task
    rawTimer.reset()
    for cue, correct, intrude, lure, _, pair, new_pair, lure_pair in enc_list:
        trialn += 1
        newEnc = newEnc_trial_run(cue, correct, intrude, lure, pair, new_pair,
                                  lure_pair, trialn, run, newEnc, enc_time)
        set_isi(enc_isi)
    print(rawTimer.getTime())
    newEnc.to_csv("Data/" + str(IDnum) + "_newEnc_" + str(run) + ".csv")  # write data to file

    trialn = 0  # reset trial number
    set_isi(interval)
    new_assoc_instruct() # show instructions

    # run the retrieval task
    rawTimer.reset()
    for cue, correct, intrude, lure, _, pair, new_pair, lure_pair in test_list:
        trialn += 1
        newRet = newRet_trial_run(cue, correct, intrude, lure, pair, new_pair,
                                  lure_pair, trialn, run, newRet, ret_time)
        set_isi(ret_isi)
    print(rawTimer.getTime())
    newRet['condition'] = vis_sim
    newRet.to_csv("Data/" + str(IDnum) + "_newRet_" + str(run) + ".csv")  # write data to file


# run a numerosity task block
def numerosity(IDnum, trials_per_pair, pairset, trialn, run, isi=0.5, rsvp_time=0.1, probe_lag=0.25):

    # predefine dataframe columns and generate trials
    num = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'Items', 'CritPair',
                                'CritPos', 'CritProbe', 'FillProbe', 'CorrItem',
                                'CorrResp', 'Resp', 'RT', 'Acc'])
    zipped_trials, vis_sim = gen_trial_num(trials_per_pair, pairset)

    # show instructions
    instruction_screen('In this task, you will view a very rapid series of images. Following this, two of the images will appear on the left and right-hand side of the screen. You will need to choose which of these two images appeared most often in the rapid stream presented just before they appeared. If you think the image on the left appeared more than the one on the right, press the left arrow. If you think the image on the right appeared more, press the right arrow.')
    instruction_screen('If you have any questions, please ask the experimenter now.',delay=2, last=True)

    # run the task
    rawTimer.reset()
    for probe_stim, winner, corr_side, items in zipped_trials:
        trialn += 1
        num = numer_trial_run(probe_stim, winner, corr_side, items, trialn, run,
                              num, rsvp_time, probe_lag)
        set_isi(isi)
    print(rawTimer.getTime())
    num['condition'] = vis_sim

    # write data to file
    num.to_csv("Data/" + str(IDnum) + "_Num_" + str(run) + ".csv")


# stat learning blocks
def stat_learning(IDnum, exposures, pairset, percent, trialn, run):

    # predefine dataframe columns and generate trials
    sta = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'Pair', 'Item',
                                'Rect', 'CorrResp', 'Resp', 'RT', 'Acc'])
    trial_list, pair_list = gen_trial_stat(exposures, pairset)
    isi_list = gen_isi(exposures)
    answers, rect_list = gen_cover(exposures, percent)
    trials = list(zip(trial_list, pair_list, isi_list, answers, rect_list))

    # show instructions
    if run == 1:
        instruction_screen('In this square-detection task, you will view a series of images, one after the other. Occasionally, one of these images will have a small grey square over top of it. If this happens, press the LEFT arrow. Otherwise, press the RIGHT arrow.')
        instruction_screen('If you have any questions, please ask the experimenter now.', last=True)
    else:
        instruction_screen('You will now complete the same task again.', delay=2, last=True)

    # run the task
    set_isi(2)
    rawTimer.reset()
    for stim, pair, isi, answer, rect in trials[:]:
        trialn += 1
        sta = trial_run(stim, answer, trialn, rect, sta, pair, run, rects=True, task='stat')
        set_isi(isi)
    print(rawTimer.getTime())

    # write data to file
    sta.to_csv("Data/" + str(IDnum) + "_Stat_" + str(run) + ".csv")


# pre and post perceptual templating
def pre_post(IDnum, exposures, pairset, percent, trialn, run):

    # predefine dataframe columns and generate trials
    prepost = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'Pair', 'Item',
                                    'Rect', 'CorrResp', 'Resp', 'RT', 'Acc'])
    trial_list, pair_list = gen_trial_prepost(exposures, pairset)  # generate the trials and pairs
    isi_list = gen_isi(exposures)  # generate the ISIs
    answers, rect_list = gen_cover(exposures, percent)
    trials = list(zip(trial_list, pair_list, isi_list, answers, rect_list))  # combine in one dataframe

    # show instructions
    if run == 1:
        instruction_screen('In this square-detection task, you will view a series of images, one after the other. Occasionally, one of these images will have a small grey square over top of it. If this happens, press the LEFT arrow. Otherwise, do not press anything.')
        instruction_screen('If you have any questions, please ask the experimenter now.',
                           delay=2, last=True)
    else:
        instruction_screen('You will now complete the same task again.',
                           delay=2, last=True)

    # run the task
    set_isi(2.0)
    for stim, pair, isi, answer, rect in trials[:]:
        trialn += 1
        prepost = trial_run(stim, answer, trialn, rect, prepost, pair, run, rects=True, task='prepost')
        set_isi(isi)

    # write data to file
    prepost.to_csv("Data/" + str(IDnum) + "_PrePost_" + str(run) + ".csv")


# visual search run
def vis_search(IDnum, encodings, pairset, targets, trialn, run, isi_study=1, isi_test=1):

    # predefine dataframe columns and assign A and B
    visStudy = pd.DataFrame(columns=['Task', 'Run', 'Trial Number', 'Pair', 'Item',
                                    'letterT', 'CorrResp', 'Resp', 'RT', 'Acc'])
    visTest = visStudy[:]

    # show instructions
    if run == 1:
        instruction_screen('In this visual search task, you will view a series of images, one after the other. At some point, a small checkerboard will appear at a particular location over the image. When you see the checkerboard, please press the LEFT arrow.')
        instruction_screen('If you have any questions about this, please ask the experimenter now.', delay=2, last=True)
    else:
        pass
        instruction_screen('You will now complete the same task again. Press the LEFT arrow, whenever a checkerboard appears on top of the image', delay=2, last=True)

    # generate trials for study and test phase
    study_list, transfer_list = gen_trial_vis(encodings, pairset, targets)

    # run learning trials
    set_isi(2)
    rawTimer.reset()
    for pair, coord, vis_sim, answer, letterT in study_list:
        trialn += 1
        stim = pair[0]
        visStudy = vis_trial_run(stim, answer, trialn, letterT, visStudy, pair, run, coord, letterTs=True, task='visStudy')
        set_isi(isi_study)
    print(rawTimer.getTime())

    # run transfer trials
    trialn = 0  # reset trial number
    rawTimer.reset()
    for pair, coord, vis_sim, answer, letterT in transfer_list:
        trialn += 1
        stim = pair[1]
        visTest = vis_trial_run(stim, answer, trialn, letterT, visTest, pair, run, coord, letterTs=True, task='visTest')
        set_isi(isi_test)
    print(rawTimer.getTime())
    visStudy.to_csv("Data/" + str(IDnum) + "_visStudy_" + str(run) + ".csv")  # write data to file
    visTest.to_csv("Data/" + str(IDnum) + "_visTest_" + str(run) + ".csv")  # write data to file


# run a full experiment with all selected tasks
def full_exp(pairset, n_prepost, n_stat, n_differint, differint,
             num_per=5, t_p_p=10, rect_per=10, encodings=1, targets=100,
             pres=[False], posts=[False], IDnum='test_subject_id', fullornot=False):
    prenum = 0
    # do pre-outcome tasks, if outcome tasks should be done before AND after
    for task, pre in list(zip(differint, pres)):
        if pre:
            for i in range(n_differint):
                if task == 'numerosity':
                    numerosity(IDnum, trials_per_pair=t_p_p, pairset=pairset, trialn=0, run=i+1)
                elif task == 'new_associates':
                    new_associate(IDnum, repetitions=encodings, pairset=pairset, trialn=0, run=i+1)
                elif task == 'associative_inf':
                    associative_inf(IDnum, repetitions=encodings, pairset=pairset, trialn=0, run=i+1)
                elif task == 'visual_search':
                    vis_search(IDnum, encodings=encodings, pairset=pairset, targets=targets, trialn=0, run=i+1)
                elif task == 'slider':
                    run_slider(sTrials, sDur, True, subject, stem, numOptions, allTrials, simDict,
                               answerDict, chanLists, superIm, nums1, fullornot)
            prenum = n_differint  # keep track of the number of outcome task runs that were done before

    # do the statistical learning task with pre & post tasks
    for i in range(n_prepost):
        pre_post(IDnum, exposures=num_per, pairset=pairset, percent=rect_per, trialn=0, run=i+1)
    for i in range(n_stat):
        stat_learning(IDnum, exposures=num_per, pairset=pairset, percent=rect_per, trialn=0, run=i+1)
    for i in range(n_prepost):
        pre_post(IDnum, exposures=num_per, pairset=pairset, percent=rect_per, trialn=0, run=i+1+n_prepost)
    if parts == "Part 1":
        instruction_screen('Congratulations, Part 1 of the experiment is finished! Please find the experimenter.',
                           delay=5, last=False, done=True)

    # do the outcome tasks
    for task, post in list(zip(differint, posts)):
        if post:
            for i in range(n_differint):
                if parts == "Part 2":
                    prenum = n_differint
                if task == 'numerosity':
                    numerosity(IDnum, trials_per_pair=t_p_p, pairset=pairset, trialn=0, run=i+prenum+1, rsvp_time=0.1)
                elif task == 'new_associates':
                    new_associate(IDnum, repetitions=encodings, pairset=pairset, trialn=0, run=i+prenum+1)
                elif task == 'associative_inf':
                    associative_inf(IDnum, repetitions=encodings, pairset=pairset, trialn=0, run=i+prenum+1)
                elif task == 'visual_search':
                    vis_search(IDnum, encodings=encodings, pairset=pairset, targets=targets, trialn=0, run=i+prenum+1)
                elif task == 'slider':
                    run_slider(3, 10, True, IDnum, stem, numOptions, allTrials, simDict,
                               answerDict, chanLists, superIm, nums1, fullornot, practise=True)
                    run_slider(sTrials, sDur, True, IDnum, stem, numOptions, allTrials, simDict,
                               answerDict, chanLists, superIm, nums1, fullornot)
                    plotTrials=True
    if parts != "Part 1":
        instruction_screen('Congratulations, you have finished! Please find the experimenter.', delay=5, last=False, done=True)
    return plotTrials


# execute full experiment
plotTrials = full_exp(pairset=basepairs, n_prepost=templates, n_stat=stats, n_differint=outcomes, differint=tasks,
                      num_per=exposures, t_p_p=nums, rect_per=rectangles, encodings=encodings, targets=targets,
                      pres=bef, posts=aft, IDnum=subid, fullornot=check)

# close the window and quit the program
m.setVisible(True)
mywin.close()

if plotTrials:
    plot_trials(subid, sTrials, sDur)


core.quit()
