# import packages
from __future__ import print_function, division
from psychopy import visual, event, core, logging, gui, data
from psychopy.tools.filetools import fromFile, toFile
from random import shuffle
import numpy as np
import pandas as pd
import random
import csv
import math
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import PIL.ImageOps
import scipy
import sys
import os


# define Esc as the exit key to abort the whole experiment
def fast_exit():
    if 'escape' in event.getKeys():
        core.quit()


# call Instruction screen with two messages and proceed with a delay if SPACE is pressed
def instruction_present(winwin, text, width=800, delay=0.25, last=False, done=False):
    fast_exit()
    winwin.flip()
    green = (51, 204, 51)
    # message 1
    instruct = visual.TextStim(winwin, text=text, wrapWidth=width,
                               alignHoriz='left', pos=(-(width/2), 100))
    # message 2
    if done:
        wrap = ''
    else:
        wrap = 'When you are ready to begin, press SPACE' if last else 'Press SPACE to continue'
    color = green if last else 'white'
    bold = True if last else False
    space = visual.TextStim(winwin, text=wrap, wrapWidth=width, colorSpace='rgb255',
                            alignHoriz='center', color=color, pos=(0, -100), bold=bold)
    # show messages
    instruct.draw()
    space.draw()
    winwin.update()
    # proceed with a delay if SPACE is pressed
    NoKey = True
    while NoKey:
        allKeys = event.getKeys()
        if len(allKeys) > 0:
            resp = allKeys[0]
            if resp == 'space':
                NoKey = False
    winwin.flip()
    core.wait(delay)


# generate lists for each set of axes
def gen_trials():
    chanLists = ["17-85", "47-34", "56-68", "79-97", "40-20", "83-101", "11-5", "102-109"]
    shifts = [1, 2] * 4
    dirs = ['away', 'away', 'toward', 'toward'] * 2
    mates = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    correct = [8, 8, 17, 17, 26, 26, 35, 35]
    _correct = [8, 79, 17, 70, 26, 61, 35, 52]
    sim = [4, 4, 3, 3, 2, 2, 1, 1]
    simDict = dict(zip(_correct, sim))
    np.random.shuffle(correct)
    answerDict = dict(zip(chanLists, correct))
    nums1 = [0, 2, 4, 7, 9, 11, 14, 16, 18, 20, 22, 25, 27, 30, 32, 34, 36, 39, 41, 43, 46, 48,
             50, 52, 55, 57, 59, 61, 64, 66, 68, 71, 73, 77, 80, 82, 84, 86, 89, 91, 93, 96, 98, 1]
    nums2 = nums1[::-1]
    # stem = 'Synth/'
    # stem = '../synthesis/Slider_Pairs/'
    stem = 'Images/'

    # number of image options
    numOptions = len(nums1) * 2

    superIm = []
    for chan in chanLists:
        tempList = []
        for num in nums1:
            tempList.append("x{}_{}A.png".format(chan, num))
        for num in nums2:
            tempList.append("x{}_{}B.png".format(chan, num))
        superIm.append(tempList)

    Chans, Shifts, Mates, Dirs = [], [], [], []
    for chan in chanLists:
        chanRep = [chan] * 8
        Chans.extend(chanRep)
        Shifts.extend(shifts)
        Mates.extend(mates)
        Dirs.extend(dirs)
    allTrials = list(zip(Chans, Shifts, Mates, Dirs))
    shuffle(allTrials)
    return stem, chanLists, superIm, numOptions, allTrials, simDict, answerDict, nums1


def run_slider(Trials, trialTime, limitScale, subject, stem, numOptions, allTrials,
               simDict, answerDict, chanLists, superIm, nums1, fullornot, practise=False):
    # build window
    slwin = visual.Window([1000, 750], color=(-1, -1, -1), monitor="testMonitor",
                          units="pix", fullscr=fullornot, waitBlanking=False)
    m = event.Mouse(win=slwin)
    m.setVisible(True)

    # color for when slider is free and locked
    moveColor = (0.22, -0.88, -0.6)
    stayColor = (-0.66, 0.54, -0.04)

    # build clock and display
    rawTimer = core.Clock()
    # actual image
    synthIm = visual.ImageStim(slwin, pos=(0, 100), size=(300, 300))
    synthMask = visual.ImageStim(slwin, pos=(0, 100), size=(300, 300))
    # slider components
    flatline = visual.Rect(slwin, pos=(0, -150), width=700, height=8, units='pix', fillColor=(1,1,1))
    edge1 = visual.Rect(slwin, pos=(-350, -150), width=8, height=30, units='pix', fillColor=(1, 1, 1))
    edge2 = visual.Rect(slwin, pos=(350, -150), width=8, height=30, units='pix', fillColor=(1, 1, 1))
    circle = visual.Circle(slwin, radius=14, units='pix', fillColor=moveColor, lineWidth=2, lineColor=(1, 1, 1))
    # timer components
    hGheight = 250
    hourGlass = visual.Rect(slwin, pos=(-300, 100), width=25, height=hGheight, units='pix',
                            fillColor=(-1, -1, -1), lineColor=(1, 1, 1), lineWidth=4)
    sand = visual.Rect(slwin, pos=(-300, 100), width=21, height=hGheight-4, units='pix', fillColor=(-1, -1, 1))
    # confirmation components
    ConfBox = visual.Rect(slwin, pos=(0, -250), width=100, height=30, units='pix', fillColor=stayColor, lineWidth=4,
                          lineColor=(1, 1, 1))
    confirm = visual.TextStim(slwin, text='  Submit? ', font='Arial', pos=(0, -250), color=(1.0, 1.0, 1.0),
                              height=20, antialias=True)
    mouse = event.Mouse(visible=True, win=slwin)

    yloc = -150

    for i, (channel, shift, PM, AT) in enumerate(allTrials):
        if i >= Trials:
            break
        if practise and i == 0:
            instructions = 'In this next task, a slider will appear at the bottom of the screen, with an image presented' \
                            ' above it. We ask that you click (but DO NOT HOLD DOWN) the mouse on the green dot, which' \
                            ' is located centrally on the slider. When you click and release the slider, the central dot will turn red, and will follow the' \
                            ' mouse as you move it. As you are moving the dot, the image above the slider will change' \
                            ' subtly. We ask that you please use the slider to carefully adjust the image until it' \
                            ' resembles one that you saw in the first phase. When you are satisfied, you can click' \
                            ' the mouse to freeze the slider. The dot will turn green again, and you will be given the' \
                            ' option to SUBMIT, by clicking the button below the slider. We will try a trial now with' \
                            ' NO IMAGES present, to get you accustomed to the mechanics of the slider.'
        elif practise and i == 1:
            instructions = 'A few things to notice. The first, is that there is a graphical timer positioned to the' \
                            ' left of the slider. It will lower and change color as your time runs out. Once your time' \
                            ' has fully run out, we will assume that your current mouse position is your choice. Also,' \
                            ' note that if you are not satisfied after freezing the slider, you can click and' \
                            ' release the dot to reposition the slider. We will try a trial now with a few' \
                            ' arbitrary animal images.'
        elif practise and i == 2:
            instructions = 'If you have any questions, please ask the experimenter now. Otherwise, we will do one last' \
                            ' trial with the animals, and you will begin the task. Note that the trials for the actual' \
                            ' task will be a little bit slower.'
        if practise:
            if i == 2:
                instruction_present(slwin, instructions, delay=2, last=True)
            else:
                instruction_present(slwin, instructions, delay=2)
        submitted = False  # no response locked
        first = True
        state = 'stay'  # slider active
        xloc = 0
        hourGlass.lineColor = (1, 1, 1)  # reset color for timer border
        sand.lineColor = (1, 1, 1)  # reset color for timer border
        circle.fillColor = moveColor  # reset color for slider handle
        TrialDat = pd.DataFrame(columns=['Trial', 'Submitted', 'Options', 'Frame', 'Position', 'Sim', 'Index',
                                         'CorrectImage', 'Image', 'State', 'Final', 'Correct', 'Pair', 'Shift',
                                         'RealLo', 'RealHi'])
        ind = chanLists.index(channel)  # choose random number to select index for random axis
        synths = superIm[ind]  # choose random axis
        # assign a correct response, depending on whether slider covers whole range, or limited scale
        if limitScale:
            eitherSide = 6
            aInd = answerDict[channel]
            bInd = ((len(nums1) * 2) - 1 - answerDict[channel])
            if PM == "A":
                ABScorrInd = aInd
                ABSpairInd = bInd
            else:
                ABScorrInd = bInd
                ABSpairInd = aInd
            simLev = simDict[ABScorrInd]
            diff = ABSpairInd - ABScorrInd
            moving = shift * -1 if ABScorrInd < ABSpairInd else shift
            if AT == 'away':
                corrInd = eitherSide - moving
                lo = ((ABScorrInd + moving) - eitherSide)
                hi = ((ABScorrInd + moving) + eitherSide + 1)
            else:
                corrInd = eitherSide + moving
                lo = ((ABScorrInd - moving) - eitherSide)
                hi = ((ABScorrInd - moving) + eitherSide + 1)
            synths = synths[lo:hi]
            numOptions = len(synths)
        else:
            corrInd = answerDict[channel]
        masks = list(np.random.choice(50, numOptions, replace=True))
        masks = ['mask{}.png'.format(msk) for msk in masks]
        frame = 0  # reset frame
        counter = 0  # reset delay
        rawTimer.reset()
        while rawTimer.getTime() < trialTime and not submitted:  # kill trial if response locked, or time over
            fast_exit()
            sandHeight = ((trialTime - rawTimer.getTime()) / trialTime) * (hGheight-4)  # update sand left in hourglass
            # modify color of timer based on time left
            if sandHeight < (0.25 * hGheight):
                sand.fillColor = moveColor
                sand.lineColor = moveColor
                hourGlass.lineColor = moveColor
            elif sandHeight < (0.4 * hGheight):
                sand.fillColor = (1.0, 0.2, -0.6)
            else:
                sand.fillColor = (-1, -1, 1)
            sandPos = (-300, (100 - (hGheight / 2)) + (sandHeight / 2))
            # draw all the stuff
            flatline.draw()
            edge1.draw()
            edge2.draw()
            hourGlass.draw()
            sand.pos = sandPos
            sand.height = sandHeight
            sand.draw()
            # introduces a delay so that click doesn't span frames
            if counter > 0:
                counter -= 1
            else:
                if state == 'move':  # set position based on mouse position
                    if mouse.getPressed() == [0, 0, 0]:
                        xloc = mouse.getPos()[0]
                        xloc = -350 if xloc < -350 else xloc
                        xloc = 350 if xloc > 350 else xloc
                        lastx = xloc
                    else:  # if clicked, change state to 'stay', change slider handle color, lock position
                        xloc = lastx
                        circle.fillColor = stayColor
                        state = 'stay'
                        counter = 20
                elif state == 'stay':  # draw the confirmation box, slider handle still locked, color still changed
                    if mouse.getPressed() == [0, 0, 0]:
                        circle.fillColor = stayColor
                        if not first:
                            ConfBox.draw()
                            confirm.draw()
                    elif abs(mouse.getPos()[0] - xloc) < 15 and abs(mouse.getPos()[1] + 150) < 15:  # revert to 'move'
                        circle.fillColor = moveColor
                        state = 'move'
                        first = False
                        counter = 20
                        lastx = 0 if first else xloc
                    elif abs(mouse.getPos()[0]) < 105 and abs(mouse.getPos()[1] + 250) < 20:  # finalize response
                        if not first:
                            submitted = True
                    else:
                        if not first:
                            ConfBox.draw()
                            confirm.draw()
            circle.setPos((xloc, yloc))  # draw the circle
            normPos = int((((xloc + 350) / 700) * numOptions) - 0.01)  # transform x coordinate into list index
            circle.draw()
            synthim = str(stem) + str(synths[normPos])  # pull image
            synthmask = str(stem) + str(masks[normPos])
            synthIm.image = synthim
            synthMask.image = synthmask
            if not practise:
                synthIm.draw()
            elif practise and i > 0:
                pracSet = [str(stem) + 'cat.jpg', str(stem) + 'dog.jpg', str(stem) + 'pig.jpeg']
                normPos = int((((xloc + 350) / 700) * 3) - 0.01)
                synthim = pracSet[normPos]
                synthIm.image = synthim
                synthIm.draw()
            synthMask.draw()
            # write out data
            TrialDat = TrialDat.append({'Trial': i, 'Submitted': submitted, 'Frame': frame, 'Position': xloc,
                                        'Index': normPos, 'Image': synths[normPos], 'State': state}, ignore_index=True)
            frame += 1
            slwin.update()
        TrialDat['Final'] = normPos  # write final response
        TrialDat['Correct'] = corrInd
        TrialDat['Options'] = numOptions
        TrialDat['Pair'] = diff
        TrialDat['Shift'] = shift
        TrialDat['RealLo'] = lo
        TrialDat['RealHi'] = hi
        TrialDat['CorrectImage'] = synths[corrInd]
        TrialDat['Sim'] = simLev
        TrialDat.to_csv('Data/{}-{}.csv'.format(subject, i))
        event.clearEvents()
        superIm[ind] = superIm[ind][::-1]
        answerDict[channel] = ((len(nums1) * 2) - 1 - answerDict[channel])
        slwin.flip()
        core.wait(1.00)  # pause until next trial
    slwin.close()
    if not practise:
        core.quit()


def plot_trials(subject, Trials, trialTime):
    nColumns = int(np.ceil(Trials/8))
    fig, axes = plt.subplots(8, nColumns, sharex=True, sharey=True, figsize=(nColumns * 3, 8))
    sim1, sim2, sim3, sim4 = [], [], [], []
    sims = [sim1, sim2, sim3, sim4]
    EndDict = {}
    for i in range(Trials):
        fname = 'Data/{}-{}.csv'.format(subject, i)
        reduce = pd.read_csv(fname)
        simLevel = reduce['Sim'].iloc[0]
        EndDict[reduce['CorrectImage'].iloc[0]] = simLevel
        sims[int(simLevel)-1].append(fname)
    sim1.extend(sim2)
    sim1.extend(sim3)
    sim1.extend(sim4)
    StimNames = list(EndDict.keys())
    StimSims = list(EndDict.values())
    print(StimNames)
    print(StimSims)
    for i, ax in zip(sim1, axes.flat):
        reduce = pd.read_csv(i)
        corr = reduce['Correct'].iloc[0]
        opts = reduce['Options'].iloc[0]
        corrBase = ((int(corr) / opts) * 700) - 350
        corrTops = (((int(corr) + 1) / opts) * 700) - 350
        ax.fill_between(np.arange(0, (60 * trialTime)), corrBase, corrTops, color=(0.17, 0.75, 0.48, 0.5))
        if reduce['Pair'].iloc[0] > 0:
            intLo = corrTops
            intHi = 360
            diffLo = -360
            diffHi = corrBase
        else:
            intLo = -360
            intHi = corrBase
            diffLo = corrTops
            diffHi = 360
        ax.fill_between(np.arange(0, (60 * trialTime)), diffLo, diffHi, color=(1, 0.31, 0.31, 0.5))
        ax.fill_between(np.arange(0, (60 * trialTime)), intLo, intHi, color=(0, 0.4, 1, 0.5))
        ax.text(0, -350, 'sim = {}'.format(reduce['Sim'].iloc[0]))
        ax.plot(reduce['Frame'], reduce['Position'], color=(0, 0, 0, 1))
    plt.ylim((-360, 360))
    plt.xlim((0, (60 * trialTime)))
    fig.text(0.5, 0.04, 'time', ha='center')
    fig.text(0.04, 0.5, 'distance from center', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    subject = sys.argv[1]
    stem, numOptions, allTrials, simDict, answerDict, nums1 = gen_trials()
    run_slider(10, 15, True, subject, stem, numOptions, allTrials, simDict, answerDict, chanLists, superIm, nums1)
    plot_trials(10, 15)
    core.quit()
