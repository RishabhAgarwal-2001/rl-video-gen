from math import cos, sin, radians
from random import random, choice
import numpy as np
import cv2
import os

# Generated images will be saved to this directory
directory = 'hand_raising_images'
# Size of a frame
size = [32, 32]
# Initial value of angle of the leg from the vertical axis
initial_leg_theta = 30.0
# Initial value of angle of the hand from the vertical axis
initial_hand_theta = 60.0
# Minimum angle of the hand from vertical axis
max_hand_theta = 120.0
# Rate of change of hand theta
delta_hand_theta = 8
# Number of frames to generate
number_of_frames = 8

# Different parameters to create a varied dataset
face_radii = [2, 3]
body_lengths = [5, 6]
hand_lengths = [4, 6]
leg_lengths = [6, 8]
roots = [[22, 8], [22, 16], [22, 24]]

def draw_face (im, center, radius, to_draw=1):
    pixel_value = to_draw
    angle = 0
    while (angle < 360):
        x = round(radius*cos(radians(angle))) + center[0]
        y = round(radius*sin(radians(angle))) + center[1]
        # print(x,' ',y)
        im[x][y] = pixel_value
        angle += 1

def draw_leg (im, root, theta, length, to_draw=1):
    pixel_value = to_draw
    theta = radians(theta)
    for i in range (1, length):      
        x = round(i*cos(theta)) + root[0]
        y = round(i*sin(theta)) + root[1]
        im[x][y] = pixel_value

def draw_hand (im, root, theta, length, to_draw=1):
    draw_leg(im, root, theta, length, to_draw)

def draw_body (im, root, length, face_radius, hand_length, hand_angle, leg_length, leg_angle, to_draw=1):
    pixel_value = to_draw
    for i in range (0, length):
        im[root[0]-i][root[1]] = pixel_value
    draw_hand(im, (root[0]-length//2, root[1]), initial_hand_theta, hand_length, to_draw)
    draw_hand(im, (root[0]-length//2, root[1]), -hand_angle, hand_length, to_draw)
    draw_leg(im, root, leg_angle, leg_length, to_draw)
    draw_leg(im, root, -leg_angle, leg_length, to_draw)
    draw_face(im, (root[0] - length - face_radius, root[1]), face_radius, to_draw)

def erase_body (im, root, length, face_radius, hand_length, hand_angle, leg_length, leg_angle):
    draw_body(im, root, length, face_radius, hand_length, hand_angle, leg_length, leg_angle, 0)

def generate_frames (im, directory, number_of_frames, root, initial_hand_theta, delta_hand_theta, face_radius, body_length, hand_length, leg_length, test=False):
    if (not os.path.isdir(directory)):
        os.mkdir(directory)   
    folder_name = str(root[1])+'_'+str(face_radius)+'_'+str(body_length)+'_'+str(hand_length)+'_'+str(leg_length)
    if test:
        folder_name = directory
    else:
        folder_name = os.path.join(directory, folder_name)
    if (not os.path.isdir(folder_name)):
        os.mkdir(folder_name)
    theta_hand = initial_hand_theta

    # Generates frames
    for i in range (0, number_of_frames):
        draw_body(im, root, body_length, face_radius, hand_length, theta_hand, leg_length, initial_leg_theta)
        filename = os.path.join(folder_name, str(i)+'.jpg')
        cv2.imwrite(filename, im*255)
        erase_body(im, root, body_length, face_radius, hand_length, theta_hand, leg_length, initial_leg_theta)

        theta_hand += delta_hand_theta

        # Hand angle
        if (theta_hand > max_hand_theta):
            theta_hand = initial_hand_theta

def createDataset ():
    im = np.zeros((size[0], size[1]))

    for root in roots:
        for face_radius in face_radii:
            for body_length in body_lengths:
                for hand_length in hand_lengths:
                    for leg_length in leg_lengths:
                        generate_frames(im, directory, number_of_frames, root, initial_hand_theta, delta_hand_theta, face_radius, body_length, hand_length, leg_length)

createDataset()