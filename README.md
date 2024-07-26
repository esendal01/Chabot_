Gerekli kütüphaneler
import sys

import random

import json

import numpy as np

import torch

import re

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QSizePolicy

from PyQt5.QtCore import Qt, QTimer

from PyQt5.QtGui import QPixmap

import requests
from bs4 import BeautifulSoup


import nltk
from nltk.stem.porter import PorterStemmer


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet



