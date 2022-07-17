from math import cos, sin, radians
from random import random, choice
import numpy as np
import cv2
import os

# Generated images with straight hands will be saved to this directory
directory1 = 'walking_images_straight_hands'
# Generated images with angled hands will be saved to this directory
directory2 = 'walking_images_angled_hands'
# Size of a frame
size = [32, 32]
# Initial position of the stick man
root = [20, 8]
# Initial value of angle of the leg from the vertical axis
initial_theta = -30.0
# Maximum angle of the leg from vertical axis
max_theta = 30.0
# Rate of change of theta
delta_theta = 10
# Speed of stick man
speed = 2
# Number of frames to generate. Stick man moves speed units to the right every frame
number_of_frames = 8

# Different parameters to create a varied dataset
face_radii = [2, 4]
body_lengths = [6, 8]
hand_lengths = [4, 6]
leg_lengths = [6, 8]

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
    draw_hand(im, (root[0]-length//2, root[1]), hand_angle, hand_length, to_draw)
    draw_hand(im, (root[0]-length//2, root[1]), -hand_angle, hand_length, to_draw)
    draw_leg(im, root, leg_angle, leg_length, to_draw)
    draw_leg(im, root, -leg_angle, leg_length, to_draw)
    draw_face(im, (root[0] - length - face_radius, root[1]), face_radius, to_draw)

def erase_body (im, root, length, face_radius, hand_length, hand_angle, leg_length, leg_angle):
    draw_body(im, root, length, face_radius, hand_length, hand_angle, leg_length, leg_angle, 0)

def generate_frames (im, directory, number_of_frames, root, initial_theta, delta_theta, speed, face_radius, body_length, hand_length, leg_length, hand_angle=90, test=False):
    if (not os.path.isdir(directory)):
        os.mkdir(directory)   
    folder_name = str(face_radius)+'_'+str(body_length)+'_'+str(hand_length)+'_'+str(leg_length)
    if test:
        folder_name = directory
    else:
        folder_name = os.path.join(directory, folder_name)
    if (not os.path.isdir(folder_name)):
        os.mkdir(folder_name)
    current_position = root.copy()
    theta = initial_theta

    # Generates frames
    for i in range (0, number_of_frames):
        draw_body(im, current_position, body_length, face_radius, hand_length, hand_angle, leg_length, theta)
        filename = os.path.join(folder_name, str(i)+'.jpg')
        cv2.imwrite(filename, im*255)
        erase_body(im, current_position, body_length, face_radius, hand_length, hand_angle, leg_length, theta)
        theta += delta_theta
        difference = abs(theta) - max_theta
        if (theta > max_theta):
            delta_theta = -delta_theta
            theta = max_theta - difference
        elif (theta < -max_theta):
            delta_theta = -delta_theta
            theta = -max_theta + difference
        current_position[1] += speed

# Program starts
def createDataset ():
    im = np.zeros((size[0], size[1]))

    for face_radius in face_radii:
        for body_length in body_lengths:
            for hand_length in hand_lengths:
                for leg_length in leg_lengths:
                    generate_frames(im, directory1, number_of_frames, root, initial_theta, delta_theta, speed, face_radius, body_length, hand_length, leg_length)
                    generate_frames(im, directory2, number_of_frames, root, initial_theta, delta_theta, speed, face_radius, body_length, hand_length, leg_length, 45)

def createTestFrame ():
    im = np.zeros((size[0], size[1]))
    new_root = root
    new_root[1] = new_root[1] + 8
    generate_frames(im, 'test_frame', 1, new_root, initial_theta, delta_theta, speed, choice([3, 5]), choice([7, 9]), choice([5, 7]), choice([7, 9]), 90, True)

createTestFrame()