import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv

prefix = 'C:/Users/Dominic/Documents/astro/research/dev_code/mcmc/auto/'
progress_name = prefix+'progress.txt'
queue_name = prefix+'queue.csv'

def load_progress(progress_name=progress_name,queue_name=queue_name):
    '''
    incomplete
    '''
    with open(progress_name,'rb') as f:
        prog_string = f.readline()
    if prog_string == '':
        #No in-progress mcmc run
        #So load the queue and start a new one
        load_queue(queue_name=queue_name)
    else:
        #There *IS* an in-progress mcmc run
        #Is it literally currently running?
        #If so, let it
        #If not, resume it from the latest save
        #All necessary information to do this must be part of progstring

def load_queue(queue_name=queue_name,newq_name=queue_name):
    '''
    incomplete
    '''
    print 'Reading file, do not quit'
    with open(queue_name,'rb') as csvfile:
        reader = csv.reader(csvfile)
        rows=[]
        for row in reader: rows.append(row)
    print 'File closed, quit at will'
    num_queued_tasks = len(rows)
    if num_queued_tasks == 0:
        return
        
    else:
        #Split queue into top task and others
        first_task = rows[0]
        rest_of_tasks = rows[1:]
        
        #Save a new queue composed of rest_of_tasks
        print 'Saving files, do not quit'
        save_queue(rest_of_tasks,queue_name=newq_name)
        
        #Save a new progress file indicating we should start first_task from the beginning
        # save_progress(first_task)
        print 'Files saved and closed, quit at will'
        
        #Start first_task from the beginning
        # resume_from_progress(new_progress_file_that_has_no_name_yet)
        
def save_queue(rows,queue_name=queue_name):
    '''
    Helper function to write a csv queue.
    Best used within a script, where the 'rows' nested list can be assembled
    either from a pre-existing csv queue; or with another function returning
    such a nested list in the correct format.
    Arguments:
    rows :: list of lists of strings, in the correct format for csv queues.
    returns nothing; only saves the file
    '''
    if rows is not None:
        with open(queue_name,'wb') as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)

def resume_from_progress(progress_name):
    '''
    incomplete
    '''
    
    pass
    
    
    
'''
Elements needed for an mcmc run:
--csv file containing observations at epochs (possibly generated at run-time)
--eccentric anomaly equation solver
--keplerian orbit params -> data points as fn of time
--orbitclass (not actually necessary, but it's imported)
--constants file
--orbitfit fn
--plotting file + trianlge package
'''