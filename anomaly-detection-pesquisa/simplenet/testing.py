import logging
import os
import sys

import click
import numpy as np
import torch

sys.path.append("src")
import backbones
import common
import metrics
import simplenet
import utils