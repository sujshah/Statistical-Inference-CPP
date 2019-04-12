# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:54:08 2019

@author: suraj
"""

import random

# Play with different concentrations
for concentration in [0.0, 0.5, 1.0]:

    # First customer always sits at the first table
    # To do otherwise would be insanity
    tables = [1]

    # n=1 is the first customer 
    for n in range(2,50):

        # Gen random number 0~1
        rand = random.random()

        p_total = 0
        existing_table = False

        for index, count in enumerate(tables):

            prob = count / (n + concentration)

            p_total += prob
            if rand < p_total:
                tables[index] += 1
                existing_table = True
                break

        # New table!!
        if not existing_table:
             tables.append(1)

    print(tables)